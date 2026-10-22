"""Deterministic provenance for RL runs, including non-Git cluster staging."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish complete documents atomically, including concurrent Slurm tasks.
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_identity(root=ROOT):
    # Include untracked source, not generated outputs, figures, logs or secrets.
    paths = []
    for directory in ("src", "experiment/run", "experiment/evaluation", "experiment/agents", "experiment/utils"):
        for p in (root / directory).rglob("*"):
            if (p.is_file() and p.suffix in {".py", ".sh", ".json"}
                    and not set(p.relative_to(root).parts).intersection(
                        {"outputs", "results", "figures", "raw_runs", "simulation_log", "__pycache__"})):
                paths.append(p)
    paths += [p for name in ("requirements.txt", "pyproject.toml", "uv.lock")
              if (p := root / name).is_file()]
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(paths)}
    def git(*args):
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    status = git("status", "--porcelain")
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(status) if status is not None else None,
            "content_sha256": digest(files), "source_files": files}


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("stable-baselines3", "gymnasium", "torch", "numpy", "tensorboard"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    import torch
    result.update(cuda_version=torch.version.cuda, cuda_available=torch.cuda.is_available())
    return result


def model_config(model, env):
    from src.env.rewards import GOAL_REWARD, STEP_COST, COLLISION_PENALTY, NORM_VIOLATION_PENALTY
    keys = ("learning_rate", "buffer_size", "learning_starts", "batch_size", "gamma", "tau",
            "gradient_steps", "target_update_interval", "exploration_fraction",
            "exploration_initial_eps", "exploration_final_eps", "max_grad_norm",
            "optimize_memory_usage", "n_steps", "stats_window_size", "seed")
    settings = {key: getattr(model, key, None) for key in keys}
    settings["learning_rate"] = {"kind": "constant" if isinstance(model.learning_rate, (int, float))
                                 else "callable", "initial_value": float(model.lr_schedule(1.0))}
    settings["train_freq"] = {"frequency": model.train_freq.frequency,
                              "unit": model.train_freq.unit.value}
    settings.update(
        device=str(model.device), algorithm=("normgym.MaskedTargetDQN_v1" if getattr(model, "mask_td_target", False) else "stable_baselines3.DQN"),
        policy_class=model.policy.__class__.__name__,
        feature_extractor=model.policy.q_net.features_extractor.__class__.__name__,
        features_dim=model.policy.q_net.features_extractor.features_dim,
        net_arch=model.policy.net_arch, network=str(model.policy.q_net),
        optimizer=model.policy.optimizer.__class__.__name__,
        optimizer_defaults={k: v for k, v in model.policy.optimizer.defaults.items()
                            if isinstance(v, (str, int, float, bool, tuple, type(None)))},
        action_mask={"greedy": "walls/bounds masked; WAIT allowed; occupied/norm actions allowed",
                     "exploration": "uniform over all five actions, including blocked moves",
                     "td_target": ("walls/bounds masked maximum" if getattr(model, "mask_td_target", False) else "unmasked maximum over all five actions")},
        observation_shapes={k: list(v.shape) for k, v in env.observation_space.spaces.items()},
        environment={k: getattr(env.unwrapped, k) for k in
                     ("norm_type", "world_type", "map_type", "capacity", "max_steps", "subject_position",
                      "ground_truth", "num_no_norm_agents", "goal_progress_reward",
                      "goal_progress_magnitude", "training_penalties", "norm_penalty", "collision_penalty", "task_goal_mode")},
        reward={"goal": GOAL_REWARD, "step_cost": STEP_COST, "collision_penalty": COLLISION_PENALTY,
                "norm_violation_penalty": NORM_VIOLATION_PENALTY,
                "learning_norm_penalty": NORM_VIOLATION_PENALTY if env.unwrapped.norm_penalty else 0.0,
                "learning_collision_penalty": COLLISION_PENALTY if env.unwrapped.collision_penalty else 0.0,
                "goal_semantics": "+10 if target index advanced, not per completed episode",
                "step_semantics": "active unfinished agent with action only",
                "collision_semantics": "blamed moving agents; excludes stationary/displaced victims",
                "shaping": "signed BFS distance change; not discounted potential-based shaping",
                "normalization": "none", "clipping": "none"})
    return settings


def check_storage(path, *, max_run_mib=256, min_free_gib=1):
    """Bound per-run artifacts; never remove checkpoints or earlier results."""
    import shutil
    path = Path(path)
    used = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    if used > max_run_mib * 1024**2:
        raise RuntimeError(f'Run output exceeds {max_run_mib} MiB: {path}')
    if shutil.disk_usage(path).free < min_free_gib * 1024**3:
        raise RuntimeError(f'Less than {min_free_gib} GiB free on output filesystem: {path}')
