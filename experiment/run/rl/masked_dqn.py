"""DQN variant matching the physical mask in greedy selection and TD targets.

Adapted from stable-baselines3 2.9 DQN.train (MIT); other optimizer, replay,
terminal, and target-update semantics are unchanged.
"""
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import DQN


def masked_max(q, observations):
    valid = observations["action_mask"] > 0.5
    if not bool(valid.any(dim=1).all()):
        raise ValueError("Every state must permit an action, including WAIT")
    return q.masked_fill(~valid, -float("inf")).max(dim=1).values


class MaskedTargetDQN(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_td_target = True

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = data.discounts if data.discounts is not None else self.gamma
            with th.no_grad():
                next_q = masked_max(self.q_net_target(data.next_observations), data.next_observations)
                target = data.rewards + (1 - data.dones) * discounts * next_q.reshape(-1, 1)
            current = th.gather(self.q_net(data.observations), 1, data.actions.long())
            loss = F.smooth_l1_loss(current, target)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
