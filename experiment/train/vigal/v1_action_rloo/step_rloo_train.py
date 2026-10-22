"""Single-step (snapshot) RLOO — the ViGaL recipe as the paper actually runs it.

Contrast with ``rloo_train.py`` (episode-level): there, a group is K whole
90-step episodes of one scenario and every turn of an episode inherits the
episode's scalar return, so the policy is never told WHICH move was wrong —
and standing still (zero violations, zero success) is the lowest-variance way
to raise that scalar (run 105919, RUN_LOG 08-24).

Here, matched to ViGaL (arXiv:2506.08011, Sec. 2.2/2.3, App. A.3) and its
released OpenRLHF scripts (``examples/scripts/train_snake.sh``,
``reward_func_qwen_snake.py``, ``advantage_estimator=rloo``):

- A step is ``prompts_per_update`` INDEPENDENT snapshots, each answered
  ``group_size`` times, drawn from the snapshot set without replacement (one
  pass = one episode). Paper: 120 x 6; here 64 x 4 (user choice, 08-24).
- Reward per completion, graded immediately by the rule verifier:
  r = r_accuracy + r_format (0.1 if well-formed). Paper: accuracy 1/0; here
  the lineage's negative charges are kept — violation -1, collision -0.25,
  unparseable -1 (snapshot_verifier.REWARD_DEFAULTS).
- RLOO advantage: A_i = r_i - mean_{j != i} r_j over the same prompt's
  samples; no std normalisation; a scalar broadcast to every response token.
- Loss: OpenRLHF PolicyLoss — per-token ratio exp(logp - old_logp) clamped
  at 3, clipped surrogate (eps 0.2), masked mean over the response tokens of
  each sequence, then mean over the minibatch. One epoch over the step's
  samples (max_epochs 1), ``minibatches`` optimizer steps per update. No KL
  (init_kl_coef 0). Temperature 1.0.
- Vision tower frozen (freeze_prefix visual): the LoRA targets only the
  language model's projections.

Deliberate deviations (documented in the README): LoRA at 3e-5 instead of
full fine-tuning at 1e-6; we start from the SFT adapter (the JSON format is
already learned; the paper starts from the base model); max_new_tokens 1024
instead of 3000 (the answer is a ~300-token JSON; long budgets invited the
digit-loop drift of run 102005).

Two evaluations run during training:
- every ``--eval_every`` updates, a held-out slice of the snapshots
  (``--eval_frac``) is graded greedily: ``eval_acc`` is a verifiable,
  hack-proof per-decision signal (cheap, ~1 min);
- every ``--episode_eval_every`` updates, ``--episode_eval_n`` FULL episodes
  are played through the unchanged pipeline on fixed seeds with the current
  adapter: ``ep_eval_return/violations/success`` are the real-game numbers
  (true env rewards), logged only; ``adapter_best`` is the best held-out ``eval_acc``.

Usage (scripts/finetune/run_v1_action_rloo.sh runs this with a vLLM server on the first GPU):
    torchrun --standalone --nproc_per_node=<gpus-1> \
        experiment/train/vigal/v1_action_rloo/step_rloo_train.py \
        --snapshots experiment/train/data/snapshots \
        --init_adapter experiment/train/checkpoints/sft_4b/adapter_final \
        --rollout_url http://localhost:8000/v1
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch
import torch.distributed as dist

from experiment.agents.llm_agent import _parse_memory_response
from experiment.train.config import (
    BASE_MODEL,
    CKPT_DIR,
    ROLLOUT_MAX_STEPS,
    ROLLOUT_SCRATCH_DIR,
    ROLLOUT_SEED_HIGH,
    ROLLOUT_SEED_LOW,
    ROLLOUT_WORLDS,
    benchmark_capacity,
)
from experiment.train.modeling import (
    collate_turns,
    encode_full,
    load_model,
    load_processor,
    token_batches,
)
from experiment.train.policy_client import HFBackend, OpenAIBackend, Turn
from experiment.train.rollout import cleanup_episode
from experiment.train.rollout_pool import RolloutPool, sync_lora, wait_for_server
from experiment.train.vigal.v1_action_rloo import harvest_snapshots as harvest
from experiment.train.vigal.v1_action_rloo.harvest_snapshots import SNAPSHOT_DIR
from experiment.train.vigal.v1_action_rloo.snapshot_verifier import REWARD_DEFAULTS, grade_response

STEP_RLOO_OUTPUT_DIR = os.path.join(CKPT_DIR, 'step_rloo_4b')

STEP_RLOO_DEFAULTS = dict(
    prompts_per_update=64,    # paper: 120 (train_snake.sh); 64 x 4 kept (user, 08-24)
    group_size=4,             # paper: n_samples_per_prompt 6; 4 kept (user, 2026-08-28)
    num_updates=200,
    lr=1e-5,                  # LoRA (paper: 1e-6 full fine-tune); 3e-5 until 2026-08-28 — single steps became fatal once most groups were degenerate
    clip_eps=0.2,             # OpenRLHF PolicyLoss default
    ratio_clamp=3.0,          # OpenRLHF: ratio.clamp(max=3)
    kl_beta=0.0,              # init_kl_coef 0 — no KL
    rloo_epochs=1,            # max_epochs 1
    minibatches=4,            # 256 samples -> 64 per optimizer step (paper: 120)
    temperature=1.0,          # rollout temperature 1.0
    top_p=1.0,
    max_new_tokens=1024,
    adv_norm=0,               # RLOO path in OpenRLHF does no std-normalisation
    ent_coef=0.0,
    eval_frac=0.05,           # held-out snapshots for greedy eval_acc
    eval_every=5,
    # Real-game eval: play full episodes through the pipeline with the current
    # adapter on FIXED seeds (comparable across updates) every N updates.
    # 0 disables. Needs --rollout_url (the pool generates through vLLM).
    episode_eval_every=0,     # off (2026-08-29): the n=1 episode eval was noise; the 16-episode re-harvest every 20 updates is the game-play validation
    episode_eval_n=1,
    episode_eval_temperature=0.7,   # the pipeline's eval sampling setting
    # On-policy re-harvest (DAgger-style state collection). The fixed harvest
    # is the SFT policy's states; once training moves the policy it reaches
    # states that set barely contains (run 108168: mid-queue "wait" states
    # were 9% of the SFT harvest, and in play the trained policy never waited
    # — eval_acc 0.31->0.53 while real-game violations went 2->15/episode).
    # Every N updates, play M episodes with adapter_latest, label every
    # visited state with the verifier (labels stay ground truth; only the
    # state distribution follows the policy) and ADD them to the pool.
    reharvest_every=20,       # 0 disables
    reharvest_episodes=16,
    hard_frac=0.25,            # share of each batch drawn from on-policy states (0.5 until 2026-08-28)
    hard_max_age=40,          # drop hard states older than 40 updates (0 = never)
                              # the policy got wrong (0 = uniform, as v2)
    save_every=0,             # numbered adapter saves (0 = only adapter_latest)
    **REWARD_DEFAULTS,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', default=SNAPSHOT_DIR,
                        help='harvest_snapshots.py output dir')
    parser.add_argument('--base_model', default=BASE_MODEL)
    parser.add_argument('--init_adapter', default=None)
    parser.add_argument('--out', default=STEP_RLOO_OUTPUT_DIR)
    parser.add_argument('--rollout_url', default=None,
                        help='OpenAI-compatible endpoint; default: in-process HF')
    parser.add_argument('--rollout_model_name', default='policy')
    parser.add_argument('--lora_name', default='policy')
    parser.add_argument('--num_workers', type=int, default=64,
                        help='concurrent generation requests (threads)')
    parser.add_argument('--start_update', type=int, default=0)
    parser.add_argument('--train_batch_tokens', type=int, default=0)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_turn_tokens', type=int, default=8192)
    parser.add_argument('--norms', nargs='*', default=None,
                        help='restrict to these norms (default: all in the set)')
    parser.add_argument('--worlds', nargs='+', default=ROLLOUT_WORLDS)
    parser.add_argument('--capacity', type=int, default=0,
                        help="episode crowd size for re-harvest / episode eval; "
                             "0 = each norm's benchmark capacity from "
                             "experiment_config.json")
    parser.add_argument('--subject_kind', default='memory_llm',
                        choices=['memory_llm', 'full_memory_llm'],
                        help='agent for re-harvest / episode-eval episodes; '
                             'MUST match the harvest that built --snapshots')
    parser.add_argument('--min_step', type=int, default=0,
                        help='train only on snapshots at env step >= this '
                             '(0 = all; applied to the initial pool, '
                             'extra_snapshots and re-harvests alike)')
    parser.add_argument('--max_steps', type=int, default=ROLLOUT_MAX_STEPS)
    parser.add_argument('--scratch', default=os.path.join(ROLLOUT_SCRATCH_DIR, 'step_rloo_eval'))
    parser.add_argument('--episode_eval_workers', type=int, default=0,
                        help='rollout processes for the episode eval (0 = episode_eval_n)')
    parser.add_argument('--reharvest_workers', type=int, default=0,
                        help='rollout processes for re-harvest (0 = reharvest_episodes)')
    parser.add_argument('--extra_snapshots', nargs='*', default=[],
                        help='re-harvest dirs from earlier runs (each with '
                             'snapshots.jsonl + images/) added as on-policy states')
    for key, value in STEP_RLOO_DEFAULTS.items():
        arg_type = type(value) if not isinstance(value, bool) else int
        parser.add_argument(f'--{key}', type=arg_type, default=value)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_snapshots(root, norms=None, min_step=0):
    path = os.path.join(root, 'snapshots.jsonl')
    records, missing = [], 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if norms and rec['norm'] not in norms:
                continue
            if min_step and rec['step'] < min_step:
                continue
            img = os.path.join(root, rec['image']) if rec.get('image') else None
            if img and not os.path.exists(img):
                missing += 1
                continue
            rec['image_abs'] = img
            records.append(rec)
    if missing:
        print(f'[data] {missing} snapshot(s) skipped: image missing', flush=True)
    return records


def split_eval(records, eval_frac, seed):
    """Hold out whole episodes (not steps) so eval states aren't near-copies
    of training ones."""
    episodes = sorted({(r['norm'], r['world'], r['seed']) for r in records})
    rng = random.Random(seed + 99)
    rng.shuffle(episodes)
    n_eval = int(round(len(episodes) * eval_frac))
    held = set(episodes[:n_eval])
    train = [r for r in records if (r['norm'], r['world'], r['seed']) not in held]
    ev = [r for r in records if (r['norm'], r['world'], r['seed']) in held]
    return train, ev


def hard_category(rec):
    """Why the policy's own move at harvest time was wrong, or None if it was
    right. Drives the hard half of every batch (change 1 after run 108444:
    uniform draws starved the queue states the policy actually fails in)."""
    g = rec.get('harvest_grade') or {}
    cat, action, good = g.get('category'), g.get('action'), rec['label'].get('good', [])
    if cat == 'good':
        return None
    if cat == 'violation':
        return 'violation'
    if good == ['wait'] and action != 'wait':
        return 'missed_wait'
    if cat == 'collision':
        return 'collision'
    return 'suboptimal'          # legal-but-not-good, or unparsed


HARD_WEIGHTS = {'violation': 3, 'missed_wait': 3, 'collision': 2, 'suboptimal': 1}


class PromptSampler:
    """Batch = ``hard_frac`` on-policy states the policy got wrong (weighted
    by HARD_WEIGHTS over hard_category) + the rest uniform over the whole
    pool. Each bucket is iterated in shuffled order without replacement
    (OpenRLHF: one dataset pass per episode) and reshuffled when exhausted."""

    def __init__(self, records, rng, hard_frac=0.0, hard_max_age=0):
        self.rng, self.hard_frac = rng, hard_frac
        self.hard_max_age = hard_max_age      # drop hard states older than this many updates (0 = keep)
        self.records, self.order = list(records), []
        self.hard = {k: [] for k in HARD_WEIGHTS}
        self.hard_order = {k: [] for k in HARD_WEIGHTS}

    def add(self, records, on_policy=False, update=0):
        """Grow the pool; on-policy states also feed the hard buckets
        (tagged with the update they were collected at)."""
        self.records.extend(records)
        for rec in records:
            self.order.insert(self.rng.randint(0, len(self.order)), rec)
            cat = hard_category(rec) if on_policy else None
            if cat:
                rec['_hard_added'] = update
                self.hard[cat].append(rec)
                self.hard_order[cat].insert(
                    self.rng.randint(0, len(self.hard_order[cat])), rec)

    def prune_hard(self, update):
        """Age cap: the hard buckets follow the CURRENT policy's mistakes;
        stale ones (2026-08-28, after the late-run collapses) only make the
        hard half of every batch more adversarial."""
        if not self.hard_max_age:
            return
        cutoff = update - self.hard_max_age
        for k in HARD_WEIGHTS:
            self.hard[k] = [r for r in self.hard[k] if r.get('_hard_added', 0) >= cutoff]
            self.hard_order[k] = [r for r in self.hard_order[k] if r.get('_hard_added', 0) >= cutoff]

    def _pop(self, order, source):
        if not order:
            order.extend(source)
            self.rng.shuffle(order)
        return order.pop()

    def next_batch(self, n, update=None):
        if update is not None:
            self.prune_hard(update)
        cats = [k for k in HARD_WEIGHTS if self.hard[k]]
        n_hard = int(round(n * self.hard_frac)) if cats else 0
        batch, composition = [], {k: 0 for k in HARD_WEIGHTS}
        for _ in range(n_hard):
            cat = self.rng.choices(cats, weights=[HARD_WEIGHTS[k] for k in cats])[0]
            batch.append(self._pop(self.hard_order[cat], self.hard[cat]))
            composition[cat] += 1
        while len(batch) < n:
            batch.append(self._pop(self.order, self.records))
        composition['uniform'] = n - n_hard
        return batch, composition

    def hard_sizes(self):
        return {k: len(v) for k, v in self.hard.items()}


def make_turn(rec, response=''):
    return Turn(system_prompt=rec['system_prompt'], user_prompt=rec['user_prompt'],
                image_paths=[rec['image_abs']] if rec['image_abs'] else [],
                response=response, step=rec['step'])


# ---------------------------------------------------------------------------
# Generation + grading
# ---------------------------------------------------------------------------

def generate_many(backend, jobs, temperature, max_new_tokens, num_workers):
    """jobs: list of (record, k). Returns responses in job order."""
    def one(job):
        rec, _ = job
        try:
            return backend.generate(make_turn(rec).messages(),
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature)
        except Exception as exc:
            print(f'[gen] failed for {rec["id"]}: {exc}', flush=True)
            return None
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        return list(ex.map(one, jobs))


def reward_kwargs(args):
    return {k: getattr(args, k) for k in REWARD_DEFAULTS}


def loo_advantages(rewards):
    """OpenRLHF rloo: baseline = (sum - r_i) / (n - 1)."""
    n = len(rewards)
    total = sum(rewards)
    return [r - (total - r) / (n - 1) for r in rewards]


def summarize(grades):
    n = max(len(grades), 1)
    cats = {}
    for g in grades:
        cats[g['category']] = cats.get(g['category'], 0) + 1
    return {
        'mean_reward': sum(g['reward'] for g in grades) / n,
        'acc': cats.get('good', 0) / n,
        'legal_frac': cats.get('legal', 0) / n,
        'violation_frac': cats.get('violation', 0) / n,
        'collision_frac': cats.get('collision', 0) / n,
        'parse_rate': sum(g['parsed'] for g in grades) / n,
        'truncated_frac': sum(g['truncated'] for g in grades) / n,
        'wait_frac': sum(g.get('action') == 'wait' for g in grades) / n,
    }


def run_eval(backend, records, args):
    """Greedy decode on the held-out snapshots -> accuracy per norm."""
    jobs = [(rec, 0) for rec in records]
    responses = generate_many(backend, jobs, 0.0, args.max_new_tokens, args.num_workers)
    grades, by_norm = [], {}
    for (rec, _), resp in zip(jobs, responses):
        g = grade_response(rec['label'], resp or '', **reward_kwargs(args))
        grades.append(g)
        by_norm.setdefault(rec['norm'], []).append(g['category'] == 'good')
    out = {'eval_' + k: v for k, v in summarize(grades).items()}
    out['eval_acc_by_norm'] = {n: round(sum(v) / len(v), 3) for n, v in by_norm.items()}
    out['eval_n'] = len(records)
    return out


# ---------------------------------------------------------------------------
# Per-token log-probs + OpenRLHF PolicyLoss
# ---------------------------------------------------------------------------

def response_token_logprobs(model, batch, device):
    """One forward -> (per-token log-probs [B, T-1], response mask [B, T-1],
    per-token entropies [B, T-1] (zero off-mask)). Tokens are the labels'
    supervised positions, i.e. exactly the sampled response."""
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
    """OpenRLHF PolicyLoss (use_dapo=False): per-token clipped surrogate,
    masked mean over each sequence's response tokens -> [B]. Also returns the
    per-sequence approx-KL and clip fraction for logging."""
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
# Real-game eval: full episodes with the current adapter
# ---------------------------------------------------------------------------

def episode_eval_scenarios(args, norms):
    """Fixed (norm, world, seed) list, identical for every eval of the run."""
    rng = random.Random(args.seed + 777)
    return [(norms[i % len(norms)], rng.choice(args.worlds),
             rng.randint(ROLLOUT_SEED_LOW, ROLLOUT_SEED_HIGH))
            for i in range(args.episode_eval_n)]


def run_episode_eval(pool, scenarios, args, update):
    """Play the scenarios through the unchanged pipeline (rollout.run_episode,
    true env rewards: goal +10, step -0.1, collision -5, violation -20) and
    summarise. Only the LoRA currently loaded on the vLLM server is used."""
    tasks = [dict(norm=n, world=w, seed=s,
                  workdir=os.path.join(args.scratch, f'eval_u{update:04d}', f'ep{i}'),
                  capacity=args.capacity or benchmark_capacity(n),
                  max_steps=args.max_steps, subject_kind=args.subject_kind,
                  gamma=1.0, format_bonus=0.0)
             for i, (n, w, s) in enumerate(scenarios)]
    episodes = [ep for ep in pool.run_batch(tasks) if ep is not None]
    if not episodes:
        return {'ep_eval_failed': len(tasks)}
    turns = [t for ep in episodes for t in ep.turns]
    wait = 0
    for t in turns:
        parsed = _parse_memory_response(t.response)
        if parsed is None or parsed.get('action_text') == 'wait':
            wait += 1
    n = len(episodes)
    out = {
        'ep_eval_n': n, 'ep_eval_failed': len(tasks) - n,
        'ep_eval_return': sum(ep.env_return for ep in episodes) / n,
        'ep_eval_violations': sum(ep.num_violations for ep in episodes) / n,
        'ep_eval_collisions': sum(ep.num_collisions for ep in episodes) / n,
        'ep_eval_goals': sum(ep.goals_reached for ep in episodes) / n,
        'ep_eval_success': sum(ep.task_success for ep in episodes) / n,
        # episodes with >= 1 violation; and the strict bar: both goals done
        # with zero violations (what the pipeline's eval calls compliant success)
        'ep_eval_violation_rate': sum(ep.num_violations > 0 for ep in episodes) / n,
        'ep_eval_violation_free_success': sum(
            ep.task_success and ep.num_violations == 0 for ep in episodes) / n,
        'ep_eval_collision_rate': sum(ep.num_collisions > 0 for ep in episodes) / n,
        'ep_eval_turn_parse': (sum(ep.turns_parsed for ep in episodes)
                               / max(1, sum(ep.turns_total for ep in episodes))),
        'ep_eval_wait_frac': wait / max(1, len(turns)),
        'ep_eval_by_norm': {},
    }
    by = {}
    for ep in episodes:
        by.setdefault(ep.norm_type, []).append(ep)
    for norm, eps in by.items():
        out['ep_eval_by_norm'][norm] = {
            'return': round(sum(e.env_return for e in eps) / len(eps), 2),
            'violations': round(sum(e.num_violations for e in eps) / len(eps), 2),
            'violation_rate': round(sum(e.num_violations > 0 for e in eps) / len(eps), 2),
            'success': round(sum(e.task_success for e in eps) / len(eps), 2),
            'violation_free_success': round(sum(
                e.task_success and e.num_violations == 0 for e in eps) / len(eps), 2)}
    for ep in episodes:
        cleanup_episode(ep)
    return out


# ---------------------------------------------------------------------------
# On-policy re-harvest
# ---------------------------------------------------------------------------

class HarvestPool:
    """Rollout workers with the snapshot-capture hook installed (the
    harvest_snapshots worker), talking to the same vLLM LoRA as training."""

    def __init__(self, num_workers, rollout_url, model_name, temperature,
                 max_new_tokens, top_p):
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        self.pool = ctx.Pool(processes=num_workers, initializer=harvest._init_worker,
                             initargs=(rollout_url, model_name, temperature,
                                       max_new_tokens, top_p))

    def run_batch(self, tasks):
        return self.pool.map(harvest._run_task, tasks, chunksize=1)

    def close(self):
        self.pool.close()
        self.pool.join()


def run_reharvest(pool, args, norms, rng, update, out_root):
    """Play reharvest_episodes with the current adapter, label every prompted
    state, and return the new records (image paths absolute) + stats."""
    out = os.path.join(out_root, f'reharvest_u{update:04d}')
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    tasks = [dict(norm=norms[i % len(norms)], world=rng.choice(args.worlds),
                  seed=rng.randint(ROLLOUT_SEED_LOW, ROLLOUT_SEED_HIGH),
                  workdir=os.path.join(args.scratch, f'reharvest_u{update:04d}', f'ep{i}'),
                  capacity=args.capacity or benchmark_capacity(norms[i % len(norms)]),
                  max_steps=args.max_steps, subject_kind=args.subject_kind,
                  out=out, image_dir=os.path.join(out, 'images'))
             for i in range(args.reharvest_episodes)]
    results = [r for r in pool.run_batch(tasks) if r is not None]
    records = []
    for r in results:
        for rec in r['records']:
            if args.min_step and rec['step'] < args.min_step:
                continue
            rec['image_abs'] = os.path.join(out, rec['image']) if rec.get('image') else None
            # Re-grade under the trainer's reward knobs (the worker used defaults).
            rec['harvest_grade'] = grade_response(rec['label'], rec['harvest_response'],
                                                  **reward_kwargs(args))
            records.append(rec)
    with open(os.path.join(out, 'snapshots.jsonl'), 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps({k: v for k, v in rec.items() if k != 'image_abs'}) + '\n')
    eps = [r['summary'] for r in results]
    with open(os.path.join(out, 'episodes.jsonl'), 'w', encoding='utf-8') as f:
        for e in eps:
            f.write(json.dumps(e) + '\n')
    stats = {'reharvest_episodes': len(eps),
             'reharvest_failed': len(tasks) - len(eps),
             'reharvest_states': len(records)}
    if eps:
        stats['reharvest_return'] = sum(e['env_return'] for e in eps) / len(eps)
        stats['reharvest_violations'] = sum(e['violations'] for e in eps) / len(eps)
        stats['reharvest_success'] = sum(e['success'] for e in eps) / len(eps)
        stats['reharvest_violation_free_success'] = sum(
            e['success'] and e['violations'] == 0 for e in eps) / len(eps)
    if records:
        # The policy's own accuracy on the states it actually visits.
        stats.update({'reharvest_' + k: v for k, v in summarize(
            [r['harvest_grade'] for r in records]).items()})
        stats['reharvest_wait_required'] = sum(
            r['label']['good'] == ['wait'] for r in records) / len(records)
    return records, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_main = rank == 0

    if not torch.cuda.is_available():
        sys.exit('[FATAL] torch.cuda.is_available() is False; refusing to train on CPU.')
    if world > 1:
        if not args.rollout_url:
            sys.exit('[FATAL] multi-rank training requires --rollout_url')
        dist.init_process_group('nccl', timeout=timedelta(hours=4))
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda'

    def say(*parts):
        if is_main:
            print(*parts, flush=True)

    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, 'training_log.jsonl')

    torch.manual_seed(args.seed)
    sample_rng = random.Random(args.seed + 1234)   # same draws on every rank
    shuffle_rng = random.Random(args.seed + 4321)

    records = load_snapshots(args.snapshots, args.norms, min_step=args.min_step)
    if not records:
        sys.exit(f'[FATAL] no snapshots under {args.snapshots}'
                 + (f' at step >= {args.min_step}' if args.min_step else ''))
    train_recs, eval_recs = split_eval(records, args.eval_frac, args.seed)
    sampler = PromptSampler(train_recs, sample_rng, hard_frac=args.hard_frac,
                            hard_max_age=args.hard_max_age)
    extra = []
    for d in args.extra_snapshots:
        recs = load_snapshots(d, min_step=args.min_step)
        for r in recs:      # re-grade under this run's reward knobs
            r['harvest_grade'] = grade_response(r['label'], r['harvest_response'],
                                                **reward_kwargs(args))
        extra.extend(recs)
        say(f'[data] +{len(recs)} on-policy states from {d}')
    if extra:
        sampler.add(extra, on_policy=True, update=args.start_update)
        train_recs.extend(extra)
    say(f'[data] hard buckets: {sampler.hard_sizes()} (hard_frac={args.hard_frac})')
    by_norm = {}
    for r in train_recs:
        by_norm[r['norm']] = by_norm.get(r['norm'], 0) + 1
    say(f'[data] {len(train_recs)} train / {len(eval_recs)} eval snapshots; '
        f'per norm: {by_norm}')
    say('[data] harvest policy on train set: ' + ' '.join(
        f'{k}={v:.2f}' for k, v in summarize(
            [grade_response(r['label'], r['harvest_response'], **reward_kwargs(args))
             for r in train_recs]).items()))

    processor = load_processor(args.base_model)
    model = load_model(args.base_model, adapter_path=args.init_adapter,
                       train_lora=True).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    ref_model = None
    if args.kl_beta > 0:
        ref_model = load_model(args.base_model, adapter_path=args.init_adapter,
                               train_lora=False).to(device)
        ref_model.eval()

    policy_params = [p for p in model.parameters() if p.requires_grad]
    # ViGaL: freeze_prefix visual. The LoRA targets (q/k/v/o/gate/up/down_proj)
    # only exist in the language model, so this is a check, not a change.
    # (.visual. = Qwen3-VL; vision_tower/multi_modal_projector = Mistral3.)
    visual_trainable = [n for n, p in model.named_parameters()
                        if p.requires_grad and any(
                            k in n for k in ('.visual.', '.vision_tower.',
                                             '.multi_modal_projector.'))]
    if visual_trainable:
        sys.exit(f'[FATAL] {len(visual_trainable)} trainable vision params '
                 f'(e.g. {visual_trainable[0]}); ViGaL freezes the vision tower')
    if world > 1:
        for p in policy_params:
            dist.broadcast(p.data, src=0)

    backend = None
    serving_dir = os.path.join(args.out, 'adapter_serving')
    served = False
    if is_main:
        if args.rollout_url:
            wait_for_server(args.rollout_url)
            model.save_pretrained(serving_dir)
            sync_lora(args.rollout_url, serving_dir, args.lora_name)
            served = True
            backend = OpenAIBackend(args.rollout_url, args.rollout_model_name,
                                    top_p=args.top_p)
        else:
            backend = HFBackend(model, processor, top_p=args.top_p)

    eval_pool, eval_scenarios = None, []
    if is_main and args.episode_eval_every > 0:
        if not args.rollout_url:
            sys.exit('[FATAL] --episode_eval_every needs --rollout_url (set 0 to disable)')
        eval_norms = args.norms or sorted(by_norm)
        eval_scenarios = episode_eval_scenarios(args, eval_norms)
        os.makedirs(args.scratch, exist_ok=True)
        eval_pool = RolloutPool(
            num_workers=args.episode_eval_workers or args.episode_eval_n,
            rollout_url=args.rollout_url, model_name=args.lora_name,
            temperature=args.episode_eval_temperature,
            max_new_tokens=args.max_new_tokens, top_p=args.top_p)
        say(f'[eval] episode eval every {args.episode_eval_every} updates: '
            f'{len(eval_scenarios)} fixed scenarios over {eval_norms}')
    best_acc = float('-inf')      # eval_acc of adapter_best
    # A resume must not let its first eval overwrite the previous run's
    # adapter_best (job 1006577 replaced u84 0.551 with u88 0.527).
    best_path = os.path.join(args.out, 'best_metric.json')
    if args.start_update > 0 and os.path.exists(best_path):
        with open(best_path) as f:
            prev = json.load(f)
        best_acc = float(prev.get('eval_acc', best_acc))
        say(f'[resume] adapter_best to beat: eval_acc={best_acc:.3f} (u{prev.get("update")})')

    harvest_pool, harvest_rng = None, random.Random(args.seed + 555)
    if is_main and args.reharvest_every > 0:
        if not args.rollout_url:
            sys.exit('[FATAL] --reharvest_every needs --rollout_url (set 0 to disable)')
        os.makedirs(args.scratch, exist_ok=True)
        harvest_pool = HarvestPool(
            num_workers=args.reharvest_workers or args.reharvest_episodes,
            rollout_url=args.rollout_url, model_name=args.lora_name,
            temperature=args.episode_eval_temperature,
            max_new_tokens=args.max_new_tokens, top_p=args.top_p)
        say(f'[reharvest] every {args.reharvest_every} updates: '
            f'{args.reharvest_episodes} on-policy episodes added to the pool')

    optimizer = torch.optim.AdamW(policy_params, lr=args.lr)

    for update in range(args.start_update, args.num_updates):
        t_start = time.time()
        t_gen = 0.0
        # rollout_batch_size independent prompts, identical on every rank.
        batch, composition = sampler.next_batch(min(args.prompts_per_update, len(train_recs)), update=update)
        jobs = [(rec, k) for rec in batch for k in range(args.group_size)]

        # ---------------- Generation + immediate grading (rank 0) ----------
        payload = None
        if is_main:
            responses = generate_many(backend, jobs, args.temperature,
                                      args.max_new_tokens, args.num_workers)
            payload = [(resp, grade_response(rec['label'], resp, **reward_kwargs(args)))
                       if resp is not None else None
                       for (rec, _), resp in zip(jobs, responses)]
            t_gen = time.time() - t_start
        if world > 1:
            holder = [payload]
            dist.broadcast_object_list(holder, src=0)
            payload = holder[0]

        # ---------------- RLOO advantages (per prompt) ---------------------
        # proto: global list of [advantage, rec, response]; identical order on
        # every rank so the round-robin sharding below agrees.
        proto, grades = [], []
        degenerate, num_groups, spreads = 0, 0, []
        for gi, rec in enumerate(batch):
            members = [payload[gi * args.group_size + k] for k in range(args.group_size)]
            members = [m for m in members if m is not None]
            if len(members) < 2:
                continue
            num_groups += 1
            rewards = [g['reward'] for _, g in members]
            spread = max(rewards) - min(rewards)
            spreads.append(spread)
            if spread < 1e-8:
                degenerate += 1
            for (resp, g), adv in zip(members, loo_advantages(rewards)):
                proto.append([adv, rec, resp])
                grades.append(g)
        if not proto:
            say(f'[update {update}] no usable groups; skipping')
            continue
        if args.adv_norm:
            std = torch.tensor([p[0] for p in proto]).std().item()
            if std > 1e-8:
                for p in proto:
                    p[0] /= std

        stats = {'update': update, 'num_prompts': len(batch),
                 'batch_composition': composition,
                 'num_groups': num_groups, 'degenerate_groups': degenerate,
                 'mean_group_spread': sum(spreads) / max(len(spreads), 1),
                 'num_samples': len(proto)}
        stats.update(summarize(grades))
        acc_by_norm = {}
        for (_, rec, _), g in zip(proto, grades):
            acc_by_norm.setdefault(rec['norm'], []).append(g['category'] == 'good')
        stats['acc_by_norm'] = {n: round(sum(v) / len(v), 3) for n, v in acc_by_norm.items()}

        # ---------------- Encode + old per-token log-probs (sharded) -------
        model.eval()
        local = {}
        for g, (_, rec, resp) in enumerate(proto):
            if g % world != rank:
                continue
            encoded = encode_full(processor, make_turn(rec).messages(), resp)
            if encoded['input_ids'].shape[1] > args.max_turn_tokens:
                local[g] = None
            else:
                local[g] = dict(encoded=encoded,
                                num_tokens=int((encoded['labels'] != -100).sum()))
        with torch.no_grad():
            kept_local = [g for g in sorted(local) if local[g] is not None]
            for batch_idx in token_batches(
                    [(None, local[g]['encoded']) for g in kept_local],
                    args.train_batch_tokens):
                gids = [kept_local[k] for k in batch_idx]
                logps, mask, _ = response_token_logprobs(
                    model, collate_turns([local[g]['encoded'] for g in gids]), device)
                for j, g in enumerate(gids):
                    # keep only this sample's own response tokens (no padding)
                    local[g]['old_logp'] = logps[j][mask[j]].detach().cpu()
                if ref_model is not None:
                    ref_logps, ref_mask, _ = response_token_logprobs(
                        ref_model, collate_turns([local[g]['encoded'] for g in gids]), device)
                    for j, g in enumerate(gids):
                        local[g]['ref_logp'] = ref_logps[j][ref_mask[j]].detach().cpu()

        meta_local = [(g, local[g] is not None) for g in sorted(local)]
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, meta_local)
            meta = dict(entry for part in gathered for entry in part)
        else:
            meta = dict(meta_local)
        kept = [g for g in range(len(proto)) if meta[g]]
        stats['num_train_samples'] = len(kept)
        stats['skipped_long'] = len(proto) - len(kept)

        # ---------------- One epoch of clipped updates (OpenRLHF loss) ----
        if kept:
            model.train()
            acc = torch.zeros(5, dtype=torch.float64, device=device)  # loss, kl, clip, ent, n
            n_samples = len(kept)
            for epoch in range(args.rloo_epochs):
                order = kept[:]
                shuffle_rng.shuffle(order)
                mb_size = max(1, (n_samples + args.minibatches - 1) // args.minibatches)
                for mb_start in range(0, n_samples, mb_size):
                    mb = order[mb_start:mb_start + mb_size]
                    mine = [g for g in mb if g % world == rank]
                    optimizer.zero_grad()
                    for batch_idx in token_batches(
                            [(None, local[g]['encoded']) for g in mine],
                            args.train_batch_tokens):
                        gids = [mine[k] for k in batch_idx]
                        new_logps, mask, ents = response_token_logprobs(
                            model, collate_turns([local[g]['encoded'] for g in gids]), device)
                        # Old log-probs back onto the padded grid of this batch.
                        old = torch.zeros_like(new_logps)
                        for j, g in enumerate(gids):
                            old[j][mask[j]] = local[g]['old_logp'].to(device)
                        advs = torch.tensor([proto[g][0] for g in gids],
                                            dtype=new_logps.dtype, device=device)
                        loss, kl, clip = policy_loss(new_logps, old, mask, advs,
                                                     args.clip_eps, args.ratio_clamp)
                        if ref_model is not None:
                            ref = torch.zeros_like(new_logps)
                            for j, g in enumerate(gids):
                                ref[j][mask[j]] = local[g]['ref_logp'].to(device)
                            d = (ref - new_logps).clamp(-4.0, 4.0)
                            loss = loss + args.kl_beta * masked_mean(d.exp() - d - 1, mask)
                        if args.ent_coef > 0:
                            loss = loss - args.ent_coef * masked_mean(ents, mask)
                        # mean over the minibatch's sequences (each rank holds
                        # its share; the all-reduce below sums the shares).
                        (loss.sum() / len(mb)).backward()
                        with torch.no_grad():
                            acc += torch.stack([loss.sum(), kl.sum(), clip.sum(),
                                                masked_mean(ents, mask).sum(),
                                                torch.tensor(float(len(gids)), device=device)]).double()
                    if world > 1:
                        for p in policy_params:
                            if p.grad is None:
                                p.grad = torch.zeros_like(p)
                            dist.all_reduce(p.grad)
                    torch.nn.utils.clip_grad_norm_(policy_params, 1.0)
                    optimizer.step()
            optimizer.zero_grad()

            if world > 1:
                dist.all_reduce(acc)
            total = max(acc[4].item(), 1.0)
            stats['policy_loss'] = acc[0].item() / total
            stats['approx_kl'] = acc[1].item() / total
            stats['clip_frac'] = acc[2].item() / total
            stats['entropy'] = acc[3].item() / total

            if is_main and served:
                model.save_pretrained(serving_dir)
                sync_lora(args.rollout_url, serving_dir, args.lora_name)
        else:
            say(f'[update {update}] no trainable samples')

        # ---------------- Held-out eval (greedy) ----------------------------
        if is_main and eval_recs and args.eval_every > 0 and (
                (update + 1) % args.eval_every == 0 or update == args.start_update):
            stats.update(run_eval(backend, eval_recs, args))
            # adapter_best = best held-out action accuracy (2026-08-29; the 1-episode
            # game-eval selector used until then froze at u9 in run 119037 and was
            # a coin flip in 115799). The episode eval below is logged only.
            if stats['eval_acc'] > best_acc:
                best_acc = stats['eval_acc']
                model.save_pretrained(os.path.join(args.out, 'adapter_best'))
                with open(os.path.join(args.out, 'best_metric.json'), 'w') as f:
                    json.dump({'update': update, **{k: v for k, v in stats.items()
                                                   if k.startswith('eval')}}, f, indent=2)
                say(f'[eval u{update}] eval_acc={stats["eval_acc"]:.3f} new best -> adapter_best')
        if is_main and eval_pool is not None and (update + 1) % args.episode_eval_every == 0:
            t_ev = time.time()
            stats.update(run_episode_eval(eval_pool, eval_scenarios, args, update))
            stats['ep_eval_seconds'] = round(time.time() - t_ev, 1)
            if 'ep_eval_success' in stats:
                print(f'[eval u{update}] n={stats["ep_eval_n"]} '
                      f'return={stats["ep_eval_return"]:.2f} '
                      f'success={stats["ep_eval_success"]:.2f} '
                      f'violation_free_success={stats["ep_eval_violation_free_success"]:.2f} '
                      f'violation_rate={stats["ep_eval_violation_rate"]:.2f} '
                      f'violations/ep={stats["ep_eval_violations"]:.2f} '
                      f'collision_rate={stats["ep_eval_collision_rate"]:.2f} '
                      f'coll/ep={stats["ep_eval_collisions"]:.2f} '
                      f'goals/ep={stats["ep_eval_goals"]:.2f} '
                      f'tparse={stats["ep_eval_turn_parse"]:.3f} '
                      f'wait={stats["ep_eval_wait_frac"]:.2f} '
                      f'by_norm={stats["ep_eval_by_norm"]} '
                      f'({stats["ep_eval_seconds"]}s)', flush=True)
        # ---------------- On-policy re-harvest -----------------------------
        # The new records must reach every rank's sampler (the batch draw is
        # replicated per rank), so they are broadcast with the stats.
        new_records = []
        if is_main and harvest_pool is not None and (update + 1) % args.reharvest_every == 0:
            t_rh = time.time()
            new_records, rh_stats = run_reharvest(
                harvest_pool, args, args.norms or sorted(by_norm), harvest_rng,
                update, os.path.join(args.out, 'reharvest'))
            stats.update(rh_stats)
            stats['reharvest_seconds'] = round(time.time() - t_rh, 1)
            print(f'[reharvest u{update}] +{len(new_records)} states from '
                  f'{rh_stats["reharvest_episodes"]} episodes '
                  f'(policy on own states: acc={rh_stats.get("reharvest_acc", 0):.3f} '
                  f'viol={rh_stats.get("reharvest_violation_frac", 0):.3f} '
                  f'wait_required={rh_stats.get("reharvest_wait_required", 0):.2f}; '
                  f'game: return={rh_stats.get("reharvest_return", 0):.1f} '
                  f'violations/ep={rh_stats.get("reharvest_violations", 0):.1f} '
                  f'success={rh_stats.get("reharvest_success", 0):.2f} '
                  f'vf_success={rh_stats.get("reharvest_violation_free_success", 0):.2f}) '
                  f'pool={len(train_recs) + len(new_records)} '
                  f'({stats["reharvest_seconds"]}s)', flush=True)
        if world > 1:
            holder = [stats, new_records]
            dist.broadcast_object_list(holder, src=0)
            stats, new_records = holder
        if new_records:
            sampler.add(new_records, on_policy=True, update=update)
            train_recs.extend(new_records)
            stats['hard_buckets'] = sampler.hard_sizes()

        stats['seconds'] = round(time.time() - t_start, 1)
        if is_main:
            stats['gen_seconds'] = round(t_gen, 1)
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(stats) + '\n')
            ev = (f' eval_acc={stats["eval_acc"]:.3f} eval_viol={stats["eval_violation_frac"]:.3f}'
                  if 'eval_acc' in stats else '')
            print(f'[update {update}] reward={stats["mean_reward"]:.3f} '
                  f'acc={stats["acc"]:.3f} legal={stats["legal_frac"]:.3f} '
                  f'viol={stats["violation_frac"]:.3f} coll={stats["collision_frac"]:.3f} '
                  f'parse={stats["parse_rate"]:.3f} '
                  f'wait={stats["wait_frac"]:.2f} '
                  f'hard={sum(v for k, v in composition.items() if k != "uniform")}/{len(batch)} '
                  f'degen={stats["degenerate_groups"]}/{stats["num_groups"]} '
                  f'ent={stats.get("entropy", 0.0):.2f} kl={stats.get("approx_kl", 0.0):.4f} '
                  f'clip={stats.get("clip_frac", 0.0):.2f}{ev} '
                  f'({stats["seconds"]}s, gen {stats["gen_seconds"]}s)', flush=True)
            model.save_pretrained(os.path.join(args.out, 'adapter_latest'))
            if args.save_every and (update + 1) % args.save_every == 0:
                model.save_pretrained(os.path.join(args.out, f'adapter_update_{update + 1:04d}'))

    if eval_pool is not None:
        eval_pool.close()
    if harvest_pool is not None:
        harvest_pool.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
