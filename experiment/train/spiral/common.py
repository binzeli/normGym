"""Shared machinery for the SPIRAL-style trainers (v1/v2/v3 under spiral/).

SPIRAL (arXiv:2506.24119) trains with REINFORCE on sparse terminal rewards
and stays stable WITHOUT a KL anchor through three mechanisms; this module
implements the two that transfer to a single-player setting:

- ``EmaBaselines`` — the paper's Role-conditioned Advantage Estimation (RAE),
  Algorithm 1 / Eq. 2: a per-key (here per-norm, their per-game-per-role)
  exponential-moving-average baseline, A = R - b_key. Unlike the RLOO
  group baseline used in vigal/, this baseline is CROSS-BATCH: when every
  sample earns the same reward (the degenerate-group state that made the
  vigal runs' collapse absorbing — 64/64 groups, zero advantage, zero
  gradient, no way back), A = R - b is still nonzero and the gradient
  points away from a below-baseline mode. The collapse trap loses its floor.
- ``DifficultyController`` — the single-player substitute for self-play's
  auto-curriculum. Zero-sum self-play pins the win rate near 50% so reward
  variance never dies; with no opponent, the controller instead moves a
  per-norm difficulty level (e.g. crowd capacity) to hold the policy's
  success rate inside a target band.

The third mechanism — optimizer discipline (lr ~1e-6, grad clip 1.0,
temperature 1.0) — lives in each trainer's defaults.

The loss is the same OpenRLHF-style clipped surrogate as the vigal trainers
(one epoch over on-policy samples ≈ REINFORCE with a safety clip); the
functions are copied here so spiral/ imports nothing from vigal/ (only the
version-neutral layer: config, modeling, policy_client, rollout*).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# RAE: per-key EMA baselines (SPIRAL Eq. 2)
# ---------------------------------------------------------------------------

class EmaBaselines:
    """b_key <- decay * b_key + (1 - decay) * R, then A = R - b_key.

    Follows SPIRAL Algorithm 1 (baseline updated with the current
    trajectory's return BEFORE the advantage is taken) with one deliberate
    deviation: a fresh key is initialized to its FIRST reward (advantage 0)
    instead of 0. The paper's b=0 init is harmless there because self-play
    pins the mean reward near 0; run 295207 showed what it does when the
    reward mean is -0.85 — ~15-20 updates of uniformly negative advantages
    while the EMA crawls down, i.e. "push down everything you sample",
    which exploded entropy (0.77 -> 3.01) and collapsed parse by u17.
    First-touch init removes that transient entirely: advantages are
    centered on the norm's actual level from episode 1.
    """

    def __init__(self, decay: float = 0.95):
        self.decay = decay
        self.values: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def advantage(self, key: str, reward: float) -> float:
        old = self.values.get(key)
        b = reward if old is None else self.decay * old + (1.0 - self.decay) * reward
        self.values[key] = b
        self.counts[key] = self.counts.get(key, 0) + 1
        return reward - b

    def state(self) -> Dict:
        return {'decay': self.decay, 'values': self.values, 'counts': self.counts}

    def load(self, state: Dict) -> None:
        self.decay = state.get('decay', self.decay)
        self.values = dict(state.get('values', {}))
        self.counts = dict(state.get('counts', {}))


# ---------------------------------------------------------------------------
# Auto-curriculum: difficulty controller (self-play's 50%-win-rate analogue)
# ---------------------------------------------------------------------------

class DifficultyController:
    """Per-key ladder of task overrides, stepped to keep success in a band.

    ``ladder`` is a list of dicts merged into the episode task (e.g.
    [{'capacity': 4}, {'capacity': 6}, {'capacity': 8}]). Each key (norm)
    tracks an EMA of its per-episode success signal; after an update, a key
    whose EMA is above ``hi`` steps up a level and one below ``lo`` steps
    down — but only after ``min_samples`` episodes at the current level, so
    one lucky batch cannot bounce the difficulty. The EMA resets on a level
    change (the old level's statistics say nothing about the new one).

    A single-entry ladder disables the controller without special-casing.
    """

    def __init__(self, keys: List[str], ladder: List[Dict], lo: float = 0.3,
                 hi: float = 0.6, decay: float = 0.9, min_samples: int = 6,
                 start_level: int = 0):
        self.ladder = list(ladder)
        self.lo, self.hi, self.decay, self.min_samples = lo, hi, decay, min_samples
        top = len(self.ladder) - 1
        self.level = {k: min(start_level, top) for k in keys}
        self.ema: Dict[str, Optional[float]] = {k: None for k in keys}
        self.samples = {k: 0 for k in keys}

    def overrides(self, key: str) -> Dict:
        return dict(self.ladder[self.level[key]])

    def record(self, key: str, success: float) -> None:
        prev = self.ema.get(key)
        self.ema[key] = (success if prev is None
                         else self.decay * prev + (1.0 - self.decay) * success)
        self.samples[key] = self.samples.get(key, 0) + 1

    def step(self) -> List[str]:
        """Apply the band rule to every key; returns human-readable changes."""
        changes = []
        for key, ema in self.ema.items():
            if ema is None or self.samples[key] < self.min_samples:
                continue
            level = self.level[key]
            if ema > self.hi and level < len(self.ladder) - 1:
                self.level[key] = level + 1
            elif ema < self.lo and level > 0:
                self.level[key] = level - 1
            else:
                continue
            changes.append(f'{key}: level {level} -> {self.level[key]} '
                           f'(ema={ema:.2f}, {self.samples[key]} eps)')
            self.ema[key] = None
            self.samples[key] = 0
        return changes

    def state(self) -> Dict:
        return {'ladder': self.ladder, 'level': self.level, 'ema': self.ema,
                'samples': self.samples}

    def load(self, state: Dict) -> None:
        # The ladder comes from the CLI (it may legitimately change between
        # resumes); only the per-key position/statistics are restored.
        top = len(self.ladder) - 1
        for k, v in state.get('level', {}).items():
            self.level[k] = min(int(v), top)
        for k, v in state.get('ema', {}).items():
            self.ema[k] = v
        for k, v in state.get('samples', {}).items():
            self.samples[k] = int(v)

    def levels(self) -> Dict[str, int]:
        return dict(self.level)


def save_state(path: str, baselines: EmaBaselines,
               controller: Optional[DifficultyController], update: int) -> None:
    state = {'update': update, 'baselines': baselines.state()}
    if controller is not None:
        state['controller'] = controller.state()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def load_state(path: str, baselines: EmaBaselines,
               controller: Optional[DifficultyController]) -> Optional[int]:
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        state = json.load(f)
    baselines.load(state.get('baselines', {}))
    if controller is not None and 'controller' in state:
        controller.load(state['controller'])
    return state.get('update')


# ---------------------------------------------------------------------------
# Loss machinery (identical semantics to the vigal trainers)
# ---------------------------------------------------------------------------

def response_token_logprobs(model, batch, device):
    """One forward -> (per-token log-probs [B, T-1], response mask [B, T-1],
    per-token entropies [B, T-1] (zero off-mask))."""
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    labels = batch.pop('labels')
    outputs = model(**batch)
    logits = outputs.logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets != -100
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    picked = torch.gather(log_probs, 2, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    sel = log_probs[mask]
    token_entropy = -(sel.exp() * sel).sum(dim=-1)
    entropies = torch.zeros_like(picked)
    entropies[mask] = token_entropy
    return picked, mask, entropies


def masked_mean(x, mask):
    return (x * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def policy_loss(new_logp, old_logp, mask, advantages, clip_eps, ratio_clamp):
    """Clipped surrogate, masked mean per sequence -> [B]; also per-sequence
    approx-KL and clip fraction for logging. At one on-policy epoch this is
    REINFORCE with a safety clip (ratio starts at 1)."""
    ratio = (new_logp - old_logp).exp().clamp(max=ratio_clamp)
    adv = advantages.unsqueeze(-1)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
    loss = masked_mean(-torch.min(surr1, surr2), mask)
    with torch.no_grad():
        log_ratio = new_logp - old_logp
        approx_kl = masked_mean(ratio - 1 - log_ratio, mask)
        clip_frac = masked_mean(((ratio - 1).abs() > clip_eps).float(), mask)
    return loss, approx_kl, clip_frac


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_many(backend, messages_list, temperature, max_new_tokens,
                  num_workers):
    """Generate one completion per messages entry (None on failure)."""
    def one(messages):
        if messages is None:
            return None
        try:
            return backend.generate(messages, max_new_tokens=max_new_tokens,
                                    temperature=temperature)
        except Exception as exc:
            print(f'[gen] failed: {exc}', flush=True)
            return None
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        return list(ex.map(one, messages_list))
