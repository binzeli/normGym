"""MLP/CNN DQN with fixed physical-action identities.

Both encoders receive the same map and vector features. Greedy action selection
and the Bellman target mask walls and boundaries; epsilon exploration samples
all five physical actions. Replay-buffer snapshots are opt-in.
"""
from pathlib import Path
from typing import Optional

import gymnasium as gym

from experiment.run.rl.env import (
    InteractiveDQNEnv,
    _subject_reward_callback,
)


def _require_dqn():
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import BaseCallback
    except ModuleNotFoundError as error:
        if error.name == "stable_baselines3":
            raise RuntimeError(
                "Stable-Baselines3 is required for DQN training. Install the "
                "project dependencies (for example: pip install -r requirements.txt)."
            ) from error
        raise
    return DQN, BaseCallback


def masked_dqn_policy_class():
    """Mask greedy Q-values without changing the identity of an action."""
    import torch
    from stable_baselines3.dqn.policies import MultiInputPolicy

    class MaskedMultiInputPolicy(MultiInputPolicy):
        def _predict(self, obs, deterministic: bool = True) -> "torch.Tensor":
            q_values = self.q_net(obs)
            mask = obs.get("action_mask") if isinstance(obs, dict) else None
            if mask is not None:
                # An all-zero mask would make every action -inf; the env always
                # leaves WAIT legal, so guard only against a malformed batch.
                valid = mask > 0.5
                valid = torch.where(valid.any(dim=1, keepdim=True), valid,
                                    torch.ones_like(valid))
                q_values = q_values.masked_fill(~valid, -float("inf"))
            return q_values.argmax(dim=1).reshape(-1)

    return MaskedMultiInputPolicy


def grid_cnn_extractor_class():
    """Two padded 3x3 convolutions, fused with the unchanged vector features."""
    import numpy as np
    import torch
    from torch import nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class GridCNNExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            channels, height, width = observation_space.spaces["grid"].shape
            self.cnn = nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Flatten(),
            )
            # Sorted so the concatenation order is stable across processes; a
            # dict iteration order change would silently permute the features.
            self.vector_keys = sorted(
                key for key in observation_space.spaces if key != "grid"
            )
            vector_dim = sum(
                int(np.prod(observation_space.spaces[key].shape))
                for key in self.vector_keys
            )
            self.linear = nn.Sequential(
                nn.Linear(32 * height * width + vector_dim, features_dim), nn.ReLU()
            )

        def forward(self, observations):
            spatial = self.cnn(observations["grid"])
            flat = [torch.flatten(observations[key], start_dim=1)
                    for key in self.vector_keys]
            return self.linear(torch.cat([spatial, *flat], dim=1))

    return GridCNNExtractor


class DQNValidActionEnv(gym.Wrapper):
    """Record the subject's physical action for the TensorBoard callback.

    The DQN action index IS the physical action; no remapping happens here.
    The legality of the chosen action is logged rather than enforced, so a run
    shows directly whether the masked policy has stopped proposing illegal
    moves (epsilon-greedy exploration still proposes some by design).
    """

    def step(self, action):
        mask = self.env.action_masks()
        observation, reward, terminated, truncated, info = self.env.step(int(action))
        return observation, reward, terminated, truncated, {
            **info,
            "subject_policy_action": int(action) if info.get("subject_id") is not None else -1,
            "valid_action_count": int(mask.sum()),
            "subject_action_legal": bool(mask[int(action)]),
        }


def load_dqn_model(model_path: Path, *, device: str = "cpu"):
    """Load a DQN policy for rendered pipeline rollouts."""
    DQN, _ = _require_dqn()
    return DQN.load(str(model_path), device=device,
                    custom_objects={"policy_class": masked_dqn_policy_class()})


def _replay_buffer_path(model_path: Path) -> Path:
    """Return the replay-buffer filename paired with a DQN checkpoint."""
    return model_path.with_name(f"{model_path.stem}_replay_buffer.pkl")


def _dqn_checkpoint_callback(checkpoint_dir: Path, checkpoint_freq: int,
                             *, save_replay_buffer: bool = False,
                             checkpoint_steps=None):
    """Save policy-only snapshots by default; replay state is explicit opt-in."""
    if checkpoint_freq <= 0:
        raise ValueError("checkpoint_freq must be positive")
    scheduled = set(checkpoint_steps) if checkpoint_steps is not None else None
    _, BaseCallback = _require_dqn()

    class DQNCheckpointCallback(BaseCallback):
        def _on_step(self) -> bool:
            if self.n_calls % 10000 == 0:
                from experiment.run.rl.provenance import check_storage
                check_storage(checkpoint_dir.parent)
            if (self.num_timesteps in scheduled if scheduled is not None
                    else self.n_calls % checkpoint_freq == 0):
                checkpoint = checkpoint_dir / f"dqn_{self.num_timesteps}_steps.zip"
                self.model.save(str(checkpoint))
                if save_replay_buffer:
                    self.model.save_replay_buffer(str(_replay_buffer_path(checkpoint)))
                from experiment.run.rl.provenance import write_json
                write_json(checkpoint.with_suffix(".json"), {
                    "environment_transitions": self.num_timesteps,
                    "optimizer_updates": self.model._n_updates,
                    "training_td_loss_last_logged": self.model.logger.name_to_value.get("train/loss"),
                    "callback_phase": "after env.step, before replay insertion and current rollout update",
                    "replay_buffer_saved": save_replay_buffer,
                    "replay_buffer_size": self.model.replay_buffer.size(),
                    "exploration_rate": self.model.exploration_rate,
                })
                print(f"[OK] Saved DQN checkpoint: {checkpoint}")
            return True

    return DQNCheckpointCallback()


def train_or_load_dqn(
    *,
    env: InteractiveDQNEnv,
    model_path: Path,
    total_timesteps: int,
    seed: int,
    checkpoint_dir: Path,
    checkpoint_freq: int,
    learning_rate: float = 1e-4,
    buffer_size: int = 100_000,
    learning_starts: int = 1_000,
    exploration_fraction: float = 0.30,
    exploration_final_eps: float = 0.05,
    device: str = "cpu",
    resume_model: Optional[Path] = None,
    policy: str = "mlp",
    gamma: float = 0.99,
    batch_size: int = 32,
    train_freq: int = 4,
    gradient_steps: int = 1,
    target_update_interval: int = 10_000,
    tau: float = 1.0,
    exploration_initial_eps: float = 1.0,
    max_grad_norm: float = 10.0,
    net_arch=None,
    features_dim: int = 256,
    save_replay_buffer: bool = False,
    checkpoint_steps=None,
    run_context=None,
    mask_td_target: bool = True,
    tensorboard_log_every: int = 100,
):
    """Create or resume an MLP or CNN epsilon-greedy DQN and save the policy."""
    DQN, _ = _require_dqn()
    if mask_td_target:
        from experiment.run.rl.masked_dqn import MaskedTargetDQN
        DQN = MaskedTargetDQN
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = checkpoint_dir.parent / "tensorboard"
    if policy not in {"mlp", "cnn"}:
        raise ValueError(f"Unknown DQN policy {policy!r}; expected 'mlp' or 'cnn'")
    # Both arms take the SAME observation; only the extractor differs, so any
    # difference in results is architectural and not an information change.
    policy_kwargs = {}
    if net_arch is not None:
        policy_kwargs["net_arch"] = list(net_arch)
    if policy == "cnn":
        policy_kwargs.update(features_extractor_class=grid_cnn_extractor_class(),
                             features_extractor_kwargs={"features_dim": features_dim})
    masked_env = DQNValidActionEnv(env)
    if resume_model is not None:
        model = DQN.load(str(resume_model), env=masked_env, device=device)
        replay_buffer = _replay_buffer_path(resume_model)
        if replay_buffer.exists():
            model.load_replay_buffer(str(replay_buffer))
            print(f"[OK] Restored DQN replay buffer: {replay_buffer}")
        else:
            print(
                f"[WARN] No replay buffer found beside {resume_model}; continuing "
                "with an empty replay buffer; this is not an exact training resume."
            )
        model.set_random_seed(seed)
    else:
        model = DQN(
            masked_dqn_policy_class(),
            masked_env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            exploration_initial_eps=exploration_initial_eps,
            gamma=gamma, batch_size=batch_size, train_freq=train_freq,
            gradient_steps=gradient_steps, target_update_interval=target_update_interval,
            tau=tau, max_grad_norm=max_grad_norm,
            seed=seed,
            device=device,
            tensorboard_log=str(tensorboard_dir),
            verbose=1,
        )
        print(f"[OK] DQN policy architecture: {policy}")
        print("[OK] DQN valid-action adapter enabled")

    from experiment.run.rl.provenance import model_config, source_identity, versions, write_json
    provenance = {
        "source": source_identity(), "versions": versions(),
        "resolved_config": model_config(model, env), "run_context": run_context or {},
        "requested_additional_timesteps": total_timesteps,
        "starting_timestep": model.num_timesteps,
        "resume_model": str(resume_model) if resume_model else None,
        "save_replay_buffer": save_replay_buffer,
        "checkpoint_steps": list(checkpoint_steps) if checkpoint_steps is not None else None,
        "checkpoint_freq": checkpoint_freq,
        "tensorboard_log_every": tensorboard_log_every,
    }
    write_json(checkpoint_dir.parent / "resolved_config.json", provenance)
    callback = _dqn_checkpoint_callback(checkpoint_dir, checkpoint_freq,
                                        save_replay_buffer=save_replay_buffer,
                                        checkpoint_steps=checkpoint_steps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[callback, _subject_reward_callback(tensorboard_dir, tensorboard_log_every)],
        reset_num_timesteps=resume_model is None,
        tb_log_name="DQN",
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    write_json(checkpoint_dir.parent / "training_complete.json", {
        "environment_transitions": model.num_timesteps, "optimizer_updates": model._n_updates,
        "training_td_loss_last_logged": model.logger.name_to_value.get("train/loss"),
        "replay_buffer_size": model.replay_buffer.size(), "exploration_rate": model.exploration_rate,
        "checkpoint_phase": "after final rollout optimizer update",
    })
    return model
