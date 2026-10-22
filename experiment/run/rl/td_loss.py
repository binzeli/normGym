"""Read-only Bellman-error probes on fixed train/validation transitions.

These probes are diagnostics, never replay-buffer inputs. Their uniform-random
behavior policy is identical across splits and independent of learned policies.
The training probe measures the training distribution, not optimizer minibatches.
"""
from __future__ import annotations

import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiment.run.rl.env import InteractiveDQNEnv
from experiment.run.rl.provenance import digest, write_json
from experiment.run.rl.validation import apply_layout


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def dataset_hash(arrays, metadata):
    content = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                   "sha256": hashlib.sha256(v.tobytes()).hexdigest()}
               for k, v in sorted(arrays.items())}
    return digest({"arrays": content, "metadata": metadata})


def collect_probe(*, norm, capacity, layout, seeds, horizon, subject_position,
                  dense=False, magnitude=0.0, norm_penalty=None, collision_penalty=None,
                  task_goal_mode="benchmark"):

    """Collect the same training-environment protocol on each designated split."""
    if not seeds or horizon < 1:
        raise ValueError("A probe needs seeds and a positive horizon")
    rows = []
    with preserve_rng():
        env = InteractiveDQNEnv(norm_type=norm, capacity=capacity, max_steps=horizon,
                                world_type=layout["world_type"], subject_position=subject_position,
                                num_no_norm_agents=1, goal_progress_reward=dense,
                                goal_progress_magnitude=magnitude, training_penalties=True,
                                norm_penalty=norm_penalty, collision_penalty=collision_penalty,
                                task_goal_mode=task_goal_mode)
        try:
            apply_layout(env.env.world, layout)
            for seed in seeds:
                observation, _ = env.reset(seed=seed)
                action_rng = random.Random(seed ^ 0x54444C)
                for _ in range(horizon):
                    before = {k: v.copy() for k, v in observation.items()}
                    action = action_rng.randrange(env.action_space.n)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    rows.append((before, action, float(reward),
                                 {k: v.copy() for k, v in observation.items()},
                                 bool(terminated), bool(truncated), seed))
                    if terminated or truncated:
                        break
        finally:
            env.close()
    arrays = {"actions": np.array([r[1] for r in rows], dtype=np.int64),
              "rewards": np.array([r[2] for r in rows], dtype=np.float32),
              "terminated": np.array([r[4] for r in rows], dtype=np.bool_),
              "truncated": np.array([r[5] for r in rows], dtype=np.bool_),
              "episode_seeds": np.array([r[6] for r in rows], dtype=np.int64)}
    for key in rows[0][0]:
        arrays[f"obs__{key}"] = np.stack([r[0][key] for r in rows])
        arrays[f"next_obs__{key}"] = np.stack([r[3][key] for r in rows])
    metadata = {"norm": norm, "capacity": capacity, "layout_sha256": layout["layout_sha256"],
                "episode_seeds": list(seeds), "horizon": horizon,
                "subject_position": subject_position, "num_no_norm_agents": 1,
                "behavior_policy": "uniform all five physical actions, including blocked actions",
                "environment_protocol": "interactive training turnover on every probe split",
                "reward": "training reward including shaping" if dense else "sparse training reward",
                "goal_progress_magnitude": magnitude if dense else 0.0,
                "use_for_gradient_updates": False, "norm_penalty": norm_penalty,
                "collision_penalty": collision_penalty, "task_goal_mode": task_goal_mode}
    return arrays, metadata


def save_probe(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse implicit replacement of the fixed diagnostic data.
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    result = {**metadata, "dataset_sha256": dataset_hash(arrays, metadata),
              "transitions": len(arrays["actions"])}
    write_json(path.with_suffix(".json"), result, exclusive=True)
    return result


def load_probe(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    base = {k: v for k, v in metadata.items() if k not in {"dataset_sha256", "transitions"}}
    if dataset_hash(arrays, base) != metadata["dataset_sha256"]:
        raise ValueError("Fixed transition dataset hash mismatch")
    return arrays, metadata


def score_td_loss(model, arrays, *, batch_size=256):
    """Match one-step DQN's configured physical target mask and smooth-L1 (Huber) loss.

    Use a single checkpoint's frozen target network for all its probe splits.
    Targets may change between checkpoints; this is not fixed-label regression.
    Pure time-limit truncations bootstrap, true terminations do not.
    """
    count = len(arrays["actions"])
    if count == 0 or batch_size <= 0:
        raise ValueError("TD loss requires nonempty data and positive batch size")
    if getattr(model, "n_steps", 1) != 1 or model.get_vec_normalize_env() is not None:
        raise ValueError("Probe supports one-step, unnormalized DQN only")
    keys = [k.removeprefix("obs__") for k in arrays if k.startswith("obs__")]
    if set(keys) != set(model.observation_space.spaces):
        raise ValueError("Probe observation keys do not match policy")
    for key in keys:
        if tuple(arrays[f"obs__{key}"].shape[1:]) != model.observation_space[key].shape:
            raise ValueError("Probe observation shape does not match policy")
    modes = {module: module.training for module in model.policy.modules()}
    totals = np.zeros(4, dtype=np.float64)
    try:
        model.policy.set_training_mode(False)
        with torch.no_grad():
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                observation = {k: arrays[f"obs__{k}"][start:stop] for k in keys}
                next_observation = {k: arrays[f"next_obs__{k}"][start:stop] for k in keys}
                obs_tensor, _ = model.policy.obs_to_tensor(observation)
                next_tensor, _ = model.policy.obs_to_tensor(next_observation)
                actions = torch.as_tensor(arrays["actions"][start:stop], device=model.device).long()
                rewards = torch.as_tensor(arrays["rewards"][start:stop], device=model.device)
                terminal = torch.as_tensor(arrays["terminated"][start:stop], device=model.device).float()
                if getattr(model, "mask_td_target", False):
                    from experiment.run.rl.masked_dqn import masked_max
                    next_q = masked_max(model.q_net_target(next_tensor), next_tensor)
                else:
                    next_q = model.q_net_target(next_tensor).max(dim=1).values
                target = rewards + (1.0 - terminal) * model.gamma * next_q
                predicted = model.q_net(obs_tensor).gather(1, actions[:, None]).flatten()
                error = target - predicted
                if not torch.isfinite(error).all():
                    raise ValueError("Nonfinite TD error")
                totals += [F.smooth_l1_loss(predicted, target, reduction="sum").item(),
                           error.abs().sum().item(), predicted.sum().item(), target.sum().item()]
    finally:
        for module, training in modes.items():
            module.training = training
    return {"huber_td_loss": float(totals[0] / count),
            "mean_absolute_td_error": float(totals[1] / count),
            "mean_predicted_q": float(totals[2] / count), "mean_td_target": float(totals[3] / count),
            "transitions": count, "gamma": float(model.gamma),
            "target": "checkpoint target network; frozen within checkpoint; " + ("physically masked maximum" if getattr(model, "mask_td_target", False) else "unmasked action maximum"),
            "terminal_handling": "bootstrap time-limit truncations; do not bootstrap true terminations",
            "gradient_updates": 0}


def prepare_probes(manifest, task, run):
    split = manifest["splits"][task["norm"]]
    config = manifest["td_loss_diagnostics"]
    episodes = config["episodes_per_split"]
    # Only seeds already used by training, never future episode seeds.
    train_count = min(episodes, max(1, (task["total_timesteps"] + task["training_horizon"] - 1) // task["training_horizon"]))
    batches = [("train_distribution_probe", manifest["training_layout"],
                list(range(task["training_seed"], task["training_seed"] + train_count))),
               ("same_layout_validation_probe", manifest["training_layout"], split["train_protocol_validation"][:episodes])]
    batches += [(key, manifest["validation_layouts"][key], seeds[:episodes])
                for key, seeds in split["held_out_layout_validation"].items()]
    probes = {}
    for label, layout, seeds in batches:
        arrays, metadata = collect_probe(norm=task["norm"], capacity=task["capacity"], layout=layout,
                                        seeds=seeds, horizon=task["training_horizon"],
                                        subject_position=task["subject_position"], dense=task["reward"] == "dense",
                                        magnitude=task["goal_progress_magnitude"],
                                        norm_penalty=task.get("norm_penalty"), collision_penalty=task.get("collision_penalty"),
                                        task_goal_mode=task.get("task_goal_mode", "benchmark"))
        path = Path(run) / "td_loss_probes" / f"{label}.npz"
        save_probe(path, arrays, metadata)
        probes[label] = path
    return probes
