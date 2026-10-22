"""Version 4 — joint action + violation-judge RLOO (v1 ∪ v3 objectives).

One LoRA, one RLOO loop, MIXED batches. Every update draws

    round(prompts_per_update * action_frac)  v1-style action snapshots
    the rest                                 v3-style violation-judge records

and answers each prompt group_size times. Each sample is graded by ITS OWN
verifier (action: the rule verifier on the chosen move; judge: the env's own
violation verdict on the yes/no), the leave-one-out advantage is computed
within the prompt's own group (an action answer is only ever compared with
its action siblings), and the clipped policy loss is applied to all samples
at once. The loss never sees which task a sample came from — the two
objectives meet only in the shared adapter weights. Both rewards are 1/0 +
0.1 format, so nothing is rescaled (``judge_adv_weight`` exists as a lever).

Self-contained like v3 (2026-09-16): its own copies of v1's verifier /
snapshot harvester and v3's judge task / oracle harvester live in this
folder; only the version-neutral layer (config, modeling, policy_client,
rollout, rollout_pool) is imported from outside.

Per task, everything else is what its own version does:

- action side (v1): PromptSampler with weighted hard buckets fed by the
  on-policy RE-HARVEST every ``reharvest_every`` updates (16 episodes played
  by adapter_latest as the full-memory subject; DAgger-style aggregation);
  ``min_step 0``; held-out greedy ``eval_acc`` on 5% of harvest episodes.
- judge side (v3): BalancedSampler (``judge_pos_frac`` violations per batch,
  class-balanced hard pool fed by wrong samples, age/FIFO capped);
  ``judge_min_step 50``; held-out greedy balanced accuracy on <= 400
  class-balanced records of 5% held-out episodes.

Init: the group's SFT adapter (the actor's JSON format must be known to play
the re-harvest episodes). ``--zero_shot N`` runs v3's signal check for the
judge task against ``--zero_shot_model_name`` (default: the served init
adapter) and exits — the "can the SFT adapter answer the observer prompt"
gate. ``adapter_best`` is selected on ``--select_metric`` (``eval_acc`` =
the actor metric, since the final test plays the model as the actor;
``eval_balanced_acc`` or ``mean`` are alternatives). No v3 transfer episode
eval: the re-harvest game stats are the play validation.
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
    build_messages,
    collate_turns,
    encode_full,
    load_model,
    load_processor,
    token_batches,
)
from experiment.train.policy_client import HFBackend, OpenAIBackend
from experiment.train.rollout_pool import sync_lora, wait_for_server
from experiment.train.vigal.v4_joint_action_judge import harvest_snapshots as harvest
from experiment.train.vigal.v4_joint_action_judge.harvest_judge import JUDGE_DATA_DIR
from experiment.train.vigal.v4_joint_action_judge.harvest_snapshots import SNAPSHOT_DIR
from experiment.train.vigal.v4_joint_action_judge.judge_task import build_user_prompt as build_judge_prompt
from experiment.train.vigal.v4_joint_action_judge.judge_task import grade as judge_grade
from experiment.train.vigal.v4_joint_action_judge.snapshot_verifier import REWARD_DEFAULTS, grade_response

V4_OUTPUT_DIR = os.path.join(CKPT_DIR, 'v4_joint_action_judge', 'default')

DEFAULTS = dict(
    # ---- shared RLOO recipe (identical to v1/v3) ----
    prompts_per_update=64,
    group_size=4,
    num_updates=200,
    lr=1e-5,
    clip_eps=0.2,
    ratio_clamp=3.0,
    kl_beta=0.0,
    rloo_epochs=1,
    minibatches=4,
    temperature=1.0,
    top_p=1.0,
    adv_norm=0,
    ent_coef=0.0,
    eval_frac=0.05,           # held-out EPISODES, per task
    eval_every=5,
    save_every=0,
    select_metric='eval_acc',  # eval_acc | eval_balanced_acc | mean
    # ---- mixing ----
    action_frac=0.5,          # share of each update's prompts that are action snapshots (32 of 64)
    judge_adv_weight=1.0,     # multiplies judge-sample advantages (1 = tasks weighted by sample count)
    # ---- action side (v1 knobs) ----
    max_new_tokens=1024,
    min_step=0,               # all steps train (restricting to late steps skews toward wait states, run 108168)
    hard_frac=0.25,
    hard_max_age=40,
    reharvest_every=20,       # 0 disables
    reharvest_episodes=16,
    reharvest_temperature=0.7,
    # ---- judge side (v3 knobs) ----
    judge_max_new_tokens=1024,
    judge_min_step=50,
    judge_pos_frac=0.4,
    judge_show_feedback=1,
    judge_hard_frac=0.25,
    judge_hard_pool_max=4000,
    judge_hard_max_age=40,
    judge_eval_max=400,
    judge_correct_reward=1.0,
    judge_format_reward=0.1,
    zero_shot=0,
    **REWARD_DEFAULTS,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots', default=SNAPSHOT_DIR, help='v1-style action snapshot set')
    p.add_argument('--records', default=os.path.join(JUDGE_DATA_DIR, 'train'),
                   help='v3-style violation-judge record set')
    p.add_argument('--base_model', default=BASE_MODEL)
    p.add_argument('--init_adapter', default=None)
    p.add_argument('--out', default=V4_OUTPUT_DIR)
    p.add_argument('--rollout_url', default=None)
    p.add_argument('--rollout_model_name', default='policy')
    p.add_argument('--zero_shot_model_name', default='policy',
                   help='served model for --zero_shot (policy = the init adapter as loaded by the launcher)')
    p.add_argument('--lora_name', default='policy')
    p.add_argument('--num_workers', type=int, default=64)
    p.add_argument('--start_update', type=int, default=0)
    p.add_argument('--train_batch_tokens', type=int, default=0)
    p.add_argument('--gradient_checkpointing', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_turn_tokens', type=int, default=24000)
    p.add_argument('--norms', nargs='*', default=None)
    p.add_argument('--worlds', nargs='+', default=ROLLOUT_WORLDS)
    p.add_argument('--capacity', type=int, default=0,
                   help="re-harvest crowd size; 0 = each norm's benchmark capacity")
    p.add_argument('--subject_kind', default='full_memory_llm',
                   choices=['memory_llm', 'full_memory_llm'],
                   help='re-harvest subject; MUST match the harvest that built --snapshots')
    p.add_argument('--max_steps', type=int, default=ROLLOUT_MAX_STEPS)
    p.add_argument('--scratch', default=os.path.join(ROLLOUT_SCRATCH_DIR, 'v4_joint'))
    p.add_argument('--reharvest_workers', type=int, default=0)
    p.add_argument('--extra_snapshots', nargs='*', default=[],
                   help='re-harvest dirs of earlier runs added as on-policy action states')
    for k, v in DEFAULTS.items():
        t = type(v) if not isinstance(v, bool) else int
        p.add_argument(f'--{k}', type=t, default=v)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_snapshots(root, norms=None, min_step=0):
    recs, missing = [], 0
    with open(os.path.join(root, 'snapshots.jsonl'), encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if norms and r['norm'] not in norms:
                continue
            if min_step and r['step'] < min_step:
                continue
            img = os.path.join(root, r['image']) if r.get('image') else None
            if img and not os.path.exists(img):
                missing += 1
                continue
            r['image_abs'] = img
            r['task'] = 'action'
            recs.append(r)
    if missing:
        print(f'[data] {missing} snapshot(s) skipped: image missing', flush=True)
    return recs


def load_judge_records(root, norms=None, min_step=0):
    recs, missing = [], 0
    with open(os.path.join(root, 'records.jsonl'), encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if norms and r['norm'] not in norms:
                continue
            if min_step and r['step'] < min_step:
                continue
            img = os.path.join(root, r['image'])
            if not os.path.exists(img):
                missing += 1
                continue
            r['image_abs'] = img
            r['task'] = 'judge'
            recs.append(r)
    if missing:
        print(f'[data] {missing} judge record(s) skipped: image missing', flush=True)
    return recs


def split_eval(records, eval_frac, seed):
    """Hold out whole episodes (not steps)."""
    episodes = sorted({(r['norm'], r['world'], r['seed']) for r in records})
    rng = random.Random(seed + 99)
    rng.shuffle(episodes)
    held = set(episodes[:int(round(len(episodes) * eval_frac))])
    return ([r for r in records if (r['norm'], r['world'], r['seed']) not in held],
            [r for r in records if (r['norm'], r['world'], r['seed']) in held])


def balanced_subset(records, n, rng):
    pos = [r for r in records if r['label']]; neg = [r for r in records if not r['label']]
    rng.shuffle(pos); rng.shuffle(neg)
    k = min(len(pos), n // 2)
    return pos[:k] + neg[:max(0, n - k)]


# ---- action sampler (v1's PromptSampler) -----------------------------------

def hard_category(rec):
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
    return 'suboptimal'


HARD_WEIGHTS = {'violation': 3, 'missed_wait': 3, 'collision': 2, 'suboptimal': 1}


class ActionSampler:
    """hard_frac of the action prompts from on-policy states the policy got
    wrong at (re-)harvest (weighted by HARD_WEIGHTS), the rest uniform; each
    bucket iterated without replacement and reshuffled when exhausted."""

    def __init__(self, records, rng, hard_frac=0.0, hard_max_age=0):
        self.rng, self.hard_frac, self.hard_max_age = rng, hard_frac, hard_max_age
        self.records, self.order = list(records), []
        self.hard = {k: [] for k in HARD_WEIGHTS}
        self.hard_order = {k: [] for k in HARD_WEIGHTS}

    def add(self, records, on_policy=False, update=0):
        self.records.extend(records)
        for rec in records:
            self.order.insert(self.rng.randint(0, len(self.order)), rec)
            cat = hard_category(rec) if on_policy else None
            if cat:
                rec['_hard_added'] = update
                self.hard[cat].append(rec)
                self.hard_order[cat].insert(self.rng.randint(0, len(self.hard_order[cat])), rec)

    def prune_hard(self, update):
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
        batch, comp = [], {k: 0 for k in HARD_WEIGHTS}
        for _ in range(n_hard):
            cat = self.rng.choices(cats, weights=[HARD_WEIGHTS[k] for k in cats])[0]
            batch.append(self._pop(self.hard_order[cat], self.hard[cat]))
            comp[cat] += 1
        while len(batch) < n:
            batch.append(self._pop(self.order, self.records))
        comp['uniform'] = n - n_hard
        return batch, comp

    def hard_sizes(self):
        return {k: len(v) for k, v in self.hard.items()}


# ---- judge sampler (v3's BalancedSampler) ----------------------------------

class JudgeSampler:
    def __init__(self, records, rng, pos_frac, hard_frac=0.0, hard_pool_max=4000, hard_max_age=0):
        self.rng, self.pos_frac, self.hard_frac = rng, pos_frac, hard_frac
        self.pools = {1: [r for r in records if r['label']], 0: [r for r in records if not r['label']]}
        self.order = {1: [], 0: []}
        self.hard = {1: [], 0: []}
        self.hard_order = {1: [], 0: []}
        self.hard_ids = set()
        self.hard_pool_max, self.hard_max_age = hard_pool_max, hard_max_age

    def _pop(self, label):
        if not self.order[label]:
            self.order[label] = list(self.pools[label])
            self.rng.shuffle(self.order[label])
        return self.order[label].pop()

    def _pop_hard(self, label):
        if not self.hard[label]:
            label = 1 - label
            if not self.hard[label]:
                return None
        if not self.hard_order[label]:
            self.hard_order[label] = list(self.hard[label]); self.rng.shuffle(self.hard_order[label])
        return self.hard_order[label].pop()

    def add_hard(self, records, update=0):
        for r in records:
            if r['id'] in self.hard_ids:
                continue
            r['_hard_added'] = update
            c = 1 if r['label'] else 0
            self.hard[c].append(r); self.hard_ids.add(r['id'])
            self.hard_order[c].insert(self.rng.randint(0, len(self.hard_order[c])), r)
        self._trim(update)

    def _trim(self, update):
        cutoff = update - self.hard_max_age if self.hard_max_age else None
        for c in (0, 1):
            keep = [r for r in self.hard[c] if cutoff is None or r.get('_hard_added', 0) >= cutoff]
            while len(keep) > self.hard_pool_max // 2:
                keep.pop(0)
            dropped = {id(r) for r in self.hard[c]} - {id(r) for r in keep}
            if dropped:
                self.hard_ids -= {r['id'] for r in self.hard[c] if id(r) in dropped}
                self.hard[c] = keep
                self.hard_order[c] = [r for r in self.hard_order[c] if id(r) not in dropped]

    def next_batch(self, n, update=None):
        if update is not None:
            self._trim(update)
        n_hard_total = len(self.hard[0]) + len(self.hard[1])
        n_hard = int(round(n * self.hard_frac)) if n_hard_total else 0
        n_hard_pos = int(round(n_hard * self.pos_frac))
        batch = []
        for i in range(n_hard):
            r = self._pop_hard(1 if i < n_hard_pos else 0)
            if r is not None:
                batch.append(r)
        n_hard = len(batch)
        rest = n - n_hard
        n_pos = int(round(rest * self.pos_frac)) if self.pools[1] else 0
        batch += [self._pop(1) for _ in range(n_pos)] + [self._pop(0) for _ in range(rest - n_pos)]
        self.rng.shuffle(batch)
        return batch, {'hard': n_hard, 'hard_pos': min(n_hard_pos, n_hard), 'pos': n_pos,
                       'neg': rest - n_pos, 'hard_pool': n_hard_total, 'hard_pool_pos': len(self.hard[1])}


# ---------------------------------------------------------------------------
# Per-task dispatch: prompt, generation budget, grading, summaries
# ---------------------------------------------------------------------------

def messages_for(rec, args):
    if rec['task'] == 'action':
        return build_messages(rec['system_prompt'], rec['user_prompt'],
                              [rec['image_abs']] if rec['image_abs'] else [])
    return build_messages(rec['system_prompt'], build_judge_prompt(rec, bool(args.judge_show_feedback)),
                          [rec['image_abs']])


def max_tokens_for(rec, args):
    return args.max_new_tokens if rec['task'] == 'action' else args.judge_max_new_tokens


def reward_kwargs(args):
    return {k: getattr(args, k) for k in REWARD_DEFAULTS}


def grade_rec(rec, resp, args):
    """-> grade dict with at least reward / parsed / correct / category."""
    if rec['task'] == 'action':
        g = grade_response(rec['label'], resp or '', **reward_kwargs(args))
        g['correct'] = g['category'] == 'good'
        return g
    return judge_grade(rec['label'], resp or '', correct_reward=args.judge_correct_reward,
                       format_reward=args.judge_format_reward, truncated=False)


def generate_many(backend, recs, args, temperature, num_workers):
    def one(rec):
        try:
            return backend.generate(messages_for(rec, args), max_new_tokens=max_tokens_for(rec, args),
                                    temperature=temperature)
        except Exception as exc:
            print(f'[gen] failed for {rec["id"]}: {exc}', flush=True)
            return None
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        return list(ex.map(one, recs))


def loo_advantages(rewards):
    n, total = len(rewards), sum(rewards)
    return [r - (total - r) / (n - 1) for r in rewards]


def summarize_action(grades):
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


def summarize_judge(grades):
    n = max(len(grades), 1)
    cats = {}
    for g in grades:
        cats[g['category']] = cats.get(g['category'], 0) + 1
    tp, tn, fp, fn = (cats.get(k, 0) for k in ('tp', 'tn', 'fp', 'fn'))
    pos = tp + fn; neg = tn + fp
    return {
        'mean_reward': sum(g['reward'] for g in grades) / n,
        'acc': (tp + tn) / n,
        'balanced_acc': 0.5 * ((tp / pos if pos else 0.0) + (tn / neg if neg else 0.0)),
        'yes_rate': sum(1 for g in grades if g.get('pred') == 1) / n,
        'recall': tp / pos if pos else 0.0,
        'precision': tp / (tp + fp) if (tp + fp) else 0.0,
        'parse_rate': sum(g['parsed'] for g in grades) / n,
        'pos_frac': pos / n,
    }


def by_norm_acc(grades, recs):
    by = {}
    for g, r in zip(grades, recs):
        by.setdefault(r['norm'], []).append(bool(g['correct']))
    return {k: round(sum(v) / len(v), 3) for k, v in by.items()}


def task_stats(grades, recs):
    """Per-task summaries with prefixes + combined top-level numbers."""
    out = {}
    for task, summ in (('action', summarize_action), ('judge', summarize_judge)):
        sel = [(g, r) for g, r in zip(grades, recs) if r['task'] == task]
        if not sel:
            continue
        gs, rs = zip(*sel)
        out.update({f'{task}_{k}': v for k, v in summ(list(gs)).items()})
        out[f'{task}_acc_by_norm'] = by_norm_acc(gs, rs)
        out[f'{task}_num_samples'] = len(gs)
    n = max(len(grades), 1)
    out['mean_reward'] = sum(g['reward'] for g in grades) / n
    out['acc'] = sum(bool(g['correct']) for g in grades) / n
    out['parse_rate'] = sum(g['parsed'] for g in grades) / n
    return out


def run_eval(backend, action_recs, judge_recs, args, judge_out_path=None):
    """Greedy held-out evals for both tasks. Action keys keep v1's names
    (eval_acc, ...); judge keys are eval_judge_* plus the alias
    eval_balanced_acc (v3's selection metric name)."""
    out = {}
    if action_recs:
        responses = generate_many(backend, action_recs, args, 0.0, args.num_workers)
        grades = [grade_rec(r, resp, args) for r, resp in zip(action_recs, responses)]
        out.update({'eval_' + k: v for k, v in summarize_action(grades).items()})
        out['eval_acc_by_norm'] = by_norm_acc(grades, action_recs)
        out['eval_n'] = len(action_recs)
    if judge_recs:
        responses = generate_many(backend, judge_recs, args, 0.0, args.num_workers)
        grades = [grade_rec(r, resp, args) for r, resp in zip(judge_recs, responses)]
        out.update({'eval_judge_' + k: v for k, v in summarize_judge(grades).items()})
        out['eval_judge_acc_by_norm'] = by_norm_acc(grades, judge_recs)
        out['eval_judge_n'] = len(judge_recs)
        out['eval_balanced_acc'] = out['eval_judge_balanced_acc']
        if judge_out_path:
            with open(judge_out_path, 'w', encoding='utf-8') as f:
                for r, resp, g in zip(judge_recs, responses, grades):
                    f.write(json.dumps({'id': r['id'], 'norm': r['norm'], 'label': r['label'],
                                        'pred': g.get('pred'), 'response': resp}) + '\n')
    return out


def selection_key(stats, metric):
    a, j = stats.get('eval_acc'), stats.get('eval_balanced_acc')
    if metric == 'eval_acc':
        return a
    if metric == 'eval_balanced_acc':
        return j
    if metric == 'mean':
        vals = [v for v in (a, j) if v is not None]
        return sum(vals) / len(vals) if vals else None
    raise ValueError(f'unknown select_metric {metric!r}')


def zero_shot_check(backend, records, args):
    """v3's signal check on the judge task: group_size samples per record ->
    parse rate, balanced accuracy, share of groups whose samples disagree."""
    jobs = [r for r in records for _ in range(args.group_size)]
    responses = generate_many(backend, jobs, args, args.temperature, args.num_workers)
    grades = [grade_rec(r, resp, args) for r, resp in zip(jobs, responses)]
    stats = summarize_judge(grades)
    stats['acc_by_norm'] = by_norm_acc(grades, jobs)
    K = args.group_size
    stats['group_disagreement'] = sum(
        1 for i in range(0, len(grades), K) if len({g['reward'] for g in grades[i:i + K]}) > 1
    ) / max(1, len(records))
    stats['n_records'] = len(records)
    return stats


# ---------------------------------------------------------------------------
# Loss (OpenRLHF PolicyLoss, as v1/v3)
# ---------------------------------------------------------------------------

def response_token_logprobs(model, batch, device):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    labels = batch.pop('labels')
    logits = model(**batch).logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets != -100
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    picked = torch.gather(log_probs, 2, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    sel = log_probs[mask]
    ent = torch.zeros_like(picked)
    ent[mask] = -(sel.exp() * sel).sum(dim=-1)
    return picked, mask, ent


def masked_mean(x, mask):
    return (x * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def policy_loss(new_logp, old_logp, mask, advantages, clip_eps, ratio_clamp):
    ratio = (new_logp - old_logp).exp().clamp(max=ratio_clamp)
    adv = advantages.unsqueeze(-1)
    loss = masked_mean(-torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv), mask)
    with torch.no_grad():
        log_ratio = new_logp - old_logp
        approx_kl = masked_mean(ratio - 1 - log_ratio, mask)
        clip_frac = masked_mean(((ratio - 1).abs() > clip_eps).float(), mask)
    return loss, approx_kl, clip_frac


# ---------------------------------------------------------------------------
# On-policy re-harvest (action side; v1's machinery over this folder's harvester)
# ---------------------------------------------------------------------------

class HarvestPool:
    def __init__(self, num_workers, rollout_url, model_name, temperature, max_new_tokens, top_p):
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        self.pool = ctx.Pool(processes=num_workers, initializer=harvest._init_worker,
                             initargs=(rollout_url, model_name, temperature, max_new_tokens, top_p))

    def run_batch(self, tasks):
        return self.pool.map(harvest._run_task, tasks, chunksize=1)

    def close(self):
        self.pool.close()
        self.pool.join()


def run_reharvest(pool, args, norms, rng, update, out_root):
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
            rec['harvest_grade'] = grade_response(rec['label'], rec['harvest_response'], **reward_kwargs(args))
            rec['task'] = 'action'
            records.append(rec)
    with open(os.path.join(out, 'snapshots.jsonl'), 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps({k: v for k, v in rec.items() if k not in ('image_abs', 'task')}) + '\n')
    eps = [r['summary'] for r in results]
    with open(os.path.join(out, 'episodes.jsonl'), 'w', encoding='utf-8') as f:
        for e in eps:
            f.write(json.dumps(e) + '\n')
    stats = {'reharvest_episodes': len(eps), 'reharvest_failed': len(tasks) - len(eps),
             'reharvest_states': len(records)}
    if eps:
        stats['reharvest_return'] = sum(e['env_return'] for e in eps) / len(eps)
        stats['reharvest_violations'] = sum(e['violations'] for e in eps) / len(eps)
        stats['reharvest_success'] = sum(e['success'] for e in eps) / len(eps)
        stats['reharvest_violation_free_success'] = sum(
            e['success'] and e['violations'] == 0 for e in eps) / len(eps)
    if records:
        stats.update({'reharvest_' + k: v for k, v in summarize_action(
            [r['harvest_grade'] for r in records]).items()})
        stats['reharvest_wait_required'] = sum(r['label']['good'] == ['wait'] for r in records) / len(records)
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
    action_rng = random.Random(args.seed + 1234)   # same draws on every rank
    judge_rng = random.Random(args.seed + 2345)
    shuffle_rng = random.Random(args.seed + 4321)
    eval_rng = random.Random(args.seed + 777)

    # ---------------- data: two pools, two samplers ---------------------
    action_all = load_snapshots(args.snapshots, args.norms, min_step=args.min_step)
    if not action_all:
        sys.exit(f'[FATAL] no snapshots under {args.snapshots}')
    judge_all = load_judge_records(args.records, args.norms, min_step=args.judge_min_step)
    if not judge_all:
        sys.exit(f'[FATAL] no judge records under {args.records} at step >= {args.judge_min_step}')
    train_action, eval_action = split_eval(action_all, args.eval_frac, args.seed)
    train_judge, eval_judge_all = split_eval(judge_all, args.eval_frac, args.seed)
    eval_judge = balanced_subset(eval_judge_all, args.judge_eval_max, eval_rng)

    action_sampler = ActionSampler(train_action, action_rng, hard_frac=args.hard_frac,
                                   hard_max_age=args.hard_max_age)
    extra = []
    for d in args.extra_snapshots:
        recs = load_snapshots(d, min_step=args.min_step)
        for r in recs:
            r['harvest_grade'] = grade_response(r['label'], r['harvest_response'], **reward_kwargs(args))
        extra.extend(recs)
        say(f'[data] +{len(recs)} on-policy action states from {d}')
    if extra:
        action_sampler.add(extra, on_policy=True, update=args.start_update)
        train_action.extend(extra)
    judge_sampler = JudgeSampler(train_judge, judge_rng, args.judge_pos_frac,
                                 hard_frac=args.judge_hard_frac, hard_pool_max=args.judge_hard_pool_max,
                                 hard_max_age=args.judge_hard_max_age)
    n_action = int(round(args.prompts_per_update * args.action_frac))
    n_judge = args.prompts_per_update - n_action
    norms_in_pool = sorted({r['norm'] for r in train_action})
    say(f'[data] action: {len(train_action)} train / {len(eval_action)} eval snapshots (step >= {args.min_step}); '
        f'per norm: {({n: sum(1 for r in train_action if r["norm"] == n) for n in norms_in_pool})}')
    say(f'[data] judge: {len(train_judge)} train records at step >= {args.judge_min_step} '
        f'({sum(r["label"] for r in train_judge)} violations), {len(eval_judge_all)} held-out '
        f'({len(eval_judge)} evaluated, {sum(r["label"] for r in eval_judge)} violations)')
    say(f'[data] mix per update: {n_action} action + {n_judge} judge prompts x {args.group_size} samples '
        f'(judge_adv_weight={args.judge_adv_weight}); hard buckets: {action_sampler.hard_sizes()}')
    say('[data] harvest policy on the action train set: ' + ' '.join(
        f'{k}={v:.2f}' for k, v in summarize_action(
            [grade_response(r['label'], r['harvest_response'], **reward_kwargs(args)) for r in train_action]).items()))

    # ---------------- zero-shot judge check (rank 0, served init adapter) --
    if args.zero_shot > 0:
        if not is_main:
            return
        wait_for_server(args.rollout_url)
        backend = OpenAIBackend(args.rollout_url, args.zero_shot_model_name, top_p=args.top_p)
        subset = balanced_subset(eval_judge_all, args.zero_shot, eval_rng)
        stats = zero_shot_check(backend, subset, args)
        say('[zero-shot] judge task with model ' + args.zero_shot_model_name + ': ' + json.dumps(
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()}))
        with open(os.path.join(args.out, 'zero_shot.json'), 'w') as f:
            json.dump({'model': args.zero_shot_model_name, **stats}, f, indent=2)
        ok = stats['parse_rate'] >= 0.9 and stats['balanced_acc'] >= 0.55 and stats['group_disagreement'] >= 0.2
        say(f'[zero-shot] {"OK: the init adapter answers the judge task with RLOO signal" if ok else "WEAK: judge signal from this init is thin"}')
        return

    # ---------------- model -----------------------------------------------
    processor = load_processor(args.base_model)
    model = load_model(args.base_model, adapter_path=args.init_adapter, train_lora=True).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    ref_model = None
    if args.kl_beta > 0:
        ref_model = load_model(args.base_model, adapter_path=args.init_adapter, train_lora=False).to(device)
        ref_model.eval()
    policy_params = [p for p in model.parameters() if p.requires_grad]
    if any(p.requires_grad and '.visual.' in n for n, p in model.named_parameters()):
        sys.exit('[FATAL] trainable vision params; the vision tower must stay frozen')
    if world > 1:
        for p in policy_params:
            dist.broadcast(p.data, src=0)

    backend, served = None, False
    serving_dir = os.path.join(args.out, 'adapter_serving')
    if is_main:
        if args.rollout_url:
            wait_for_server(args.rollout_url)
            model.save_pretrained(serving_dir)
            sync_lora(args.rollout_url, serving_dir, args.lora_name)
            served = True
            backend = OpenAIBackend(args.rollout_url, args.rollout_model_name, top_p=args.top_p)
        else:
            backend = HFBackend(model, processor, top_p=args.top_p)

    best_key = float('-inf')
    if args.start_update > 0 and os.path.exists(os.path.join(args.out, 'best_metric.json')):
        with open(os.path.join(args.out, 'best_metric.json')) as f:
            prev = json.load(f)
        k = selection_key(prev, args.select_metric)
        if k is not None:
            best_key = float(k)
        say(f'[resume] best_metric.json: {args.select_metric}={best_key:.3f} @u{prev.get("update")} '
            f'(adapter_best kept until beaten)')

    harvest_pool, harvest_rng = None, random.Random(args.seed + 555)
    if is_main and args.reharvest_every > 0:
        if not args.rollout_url:
            sys.exit('[FATAL] --reharvest_every needs --rollout_url (set 0 to disable)')
        os.makedirs(args.scratch, exist_ok=True)
        harvest_pool = HarvestPool(num_workers=args.reharvest_workers or args.reharvest_episodes,
                                   rollout_url=args.rollout_url, model_name=args.lora_name,
                                   temperature=args.reharvest_temperature,
                                   max_new_tokens=args.max_new_tokens, top_p=args.top_p)
        say(f'[reharvest] every {args.reharvest_every} updates: {args.reharvest_episodes} on-policy '
            f'episodes ({args.subject_kind}) added to the action pool')

    optimizer = torch.optim.AdamW(policy_params, lr=args.lr)

    for update in range(args.start_update, args.num_updates):
        t_start = time.time()
        # ---------------- mixed batch (identical draw on every rank) -------
        a_batch, a_comp = action_sampler.next_batch(min(n_action, len(train_action)), update=update)
        j_batch, j_comp = judge_sampler.next_batch(min(n_judge, len(train_judge)), update=update)
        batch = a_batch + j_batch
        composition = {'action': a_comp, 'judge': j_comp}
        jobs = [rec for rec in batch for _ in range(args.group_size)]

        # ---------------- generation + grading (rank 0) --------------------
        payload, t_gen = None, 0.0
        if is_main:
            responses = generate_many(backend, jobs, args, args.temperature, args.num_workers)
            payload = [(resp, grade_rec(rec, resp, args)) if resp is not None else None
                       for rec, resp in zip(jobs, responses)]
            t_gen = time.time() - t_start
        if world > 1:
            holder = [payload]
            dist.broadcast_object_list(holder, src=0)
            payload = holder[0]

        # ---------------- RLOO advantages within each prompt's group --------
        proto, grades, recs_of = [], [], []
        degenerate = {'action': 0, 'judge': 0}
        num_groups = {'action': 0, 'judge': 0}
        K = args.group_size
        for gi, rec in enumerate(batch):
            members = [m for m in payload[gi * K:(gi + 1) * K] if m is not None]
            if len(members) < 2:
                continue
            num_groups[rec['task']] += 1
            rewards = [g['reward'] for _, g in members]
            if max(rewards) - min(rewards) < 1e-8:
                degenerate[rec['task']] += 1
            w = args.judge_adv_weight if rec['task'] == 'judge' else 1.0
            for (resp, g), adv in zip(members, loo_advantages(rewards)):
                proto.append([adv * w, rec, resp]); grades.append(g); recs_of.append(rec)
        if not proto:
            say(f'[update {update}] no usable groups; skipping')
            continue
        if args.adv_norm:
            std = torch.tensor([p[0] for p in proto]).std().item()
            if std > 1e-8:
                for p in proto:
                    p[0] /= std
        if args.judge_hard_frac > 0:
            wrong = {id(rec) for g, rec in zip(grades, recs_of) if rec['task'] == 'judge' and not g['correct']}
            judge_sampler.add_hard([rec for rec in j_batch if id(rec) in wrong], update=update)

        stats = {'update': update, 'num_prompts': len(batch), 'batch_composition': composition,
                 'num_groups': sum(num_groups.values()), 'num_groups_by_task': num_groups,
                 'degenerate_groups': sum(degenerate.values()), 'degenerate_by_task': degenerate,
                 'num_samples': len(proto)}
        stats.update(task_stats(grades, recs_of))

        # ---------------- encode + old log-probs (sharded) -----------------
        model.eval()
        local = {}
        for g, (_, rec, resp) in enumerate(proto):
            if g % world != rank:
                continue
            enc = encode_full(processor, messages_for(rec, args), resp)
            local[g] = None if enc['input_ids'].shape[1] > args.max_turn_tokens else dict(encoded=enc)
        with torch.no_grad():
            kept_local = [g for g in sorted(local) if local[g] is not None]
            for batch_idx in token_batches([(None, local[g]['encoded']) for g in kept_local],
                                           args.train_batch_tokens):
                gids = [kept_local[k] for k in batch_idx]
                logps, mask, _ = response_token_logprobs(
                    model, collate_turns([local[g]['encoded'] for g in gids]), device)
                for j, g in enumerate(gids):
                    local[g]['old_logp'] = logps[j][mask[j]].detach().cpu()
                if ref_model is not None:
                    rl, rm, _ = response_token_logprobs(
                        ref_model, collate_turns([local[g]['encoded'] for g in gids]), device)
                    for j, g in enumerate(gids):
                        local[g]['ref_logp'] = rl[j][rm[j]].detach().cpu()
        meta_local = [(g, local[g] is not None) for g in sorted(local)]
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, meta_local)
            meta = dict(e for part in gathered for e in part)
        else:
            meta = dict(meta_local)
        kept = [g for g in range(len(proto)) if meta[g]]
        stats['num_train_samples'] = len(kept)
        stats['skipped_long'] = len(proto) - len(kept)
        stats['skipped_long_by_task'] = {
            t: sum(1 for g in range(len(proto)) if not meta[g] and proto[g][1]['task'] == t)
            for t in ('action', 'judge')}

        # ---------------- one epoch of clipped updates ---------------------
        if kept:
            model.train()
            acc = torch.zeros(5, dtype=torch.float64, device=device)  # loss, kl, clip, ent, n
            n_samples = len(kept)
            for _ in range(args.rloo_epochs):
                order = kept[:]
                shuffle_rng.shuffle(order)
                mb_size = max(1, (n_samples + args.minibatches - 1) // args.minibatches)
                for mb_start in range(0, n_samples, mb_size):
                    mb = order[mb_start:mb_start + mb_size]
                    mine = [g for g in mb if g % world == rank]
                    optimizer.zero_grad()
                    for batch_idx in token_batches([(None, local[g]['encoded']) for g in mine],
                                                   args.train_batch_tokens):
                        gids = [mine[k] for k in batch_idx]
                        new_logps, mask, ents = response_token_logprobs(
                            model, collate_turns([local[g]['encoded'] for g in gids]), device)
                        old = torch.zeros_like(new_logps)
                        for j, g in enumerate(gids):
                            old[j][mask[j]] = local[g]['old_logp'].to(device)
                        advs = torch.tensor([proto[g][0] for g in gids], dtype=new_logps.dtype, device=device)
                        loss, kl, clip = policy_loss(new_logps, old, mask, advs, args.clip_eps, args.ratio_clamp)
                        if ref_model is not None:
                            ref = torch.zeros_like(new_logps)
                            for j, g in enumerate(gids):
                                ref[j][mask[j]] = local[g]['ref_logp'].to(device)
                            d = (ref - new_logps).clamp(-4.0, 4.0)
                            loss = loss + args.kl_beta * masked_mean(d.exp() - d - 1, mask)
                        if args.ent_coef > 0:
                            loss = loss - args.ent_coef * masked_mean(ents, mask)
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

        # ---------------- held-out evals (greedy), both tasks --------------
        if is_main and args.eval_every > 0 and ((update + 1) % args.eval_every == 0 or update == args.start_update):
            stats.update(run_eval(backend, eval_action, eval_judge, args,
                                  os.path.join(args.out, f'eval_judge_u{update:04d}.jsonl')))
            key = selection_key(stats, args.select_metric)
            if key is not None and key > best_key:
                best_key = key
                model.save_pretrained(os.path.join(args.out, 'adapter_best'))
                with open(os.path.join(args.out, 'best_metric.json'), 'w') as f:
                    json.dump({'update': update, 'select_metric': args.select_metric,
                               **{k: v for k, v in stats.items() if k.startswith('eval')}}, f, indent=2)
                say(f'[eval u{update}] {args.select_metric}={key:.3f} new best -> adapter_best')

        # ---------------- on-policy re-harvest (action pool) ---------------
        new_records = []
        if is_main and harvest_pool is not None and (update + 1) % args.reharvest_every == 0:
            t_rh = time.time()
            new_records, rh = run_reharvest(harvest_pool, args, args.norms or norms_in_pool, harvest_rng,
                                            update, os.path.join(args.out, 'reharvest'))
            stats.update(rh)
            stats['reharvest_seconds'] = round(time.time() - t_rh, 1)
            print(f'[reharvest u{update}] +{len(new_records)} states from {rh["reharvest_episodes"]} episodes '
                  f'(policy on own states: acc={rh.get("reharvest_acc", 0):.3f} '
                  f'viol={rh.get("reharvest_violation_frac", 0):.3f} '
                  f'wait_required={rh.get("reharvest_wait_required", 0):.2f}; '
                  f'game: return={rh.get("reharvest_return", 0):.1f} '
                  f'violations/ep={rh.get("reharvest_violations", 0):.1f} '
                  f'success={rh.get("reharvest_success", 0):.2f} '
                  f'vf_success={rh.get("reharvest_violation_free_success", 0):.2f}) '
                  f'pool={len(train_action) + len(new_records)} ({stats["reharvest_seconds"]}s)', flush=True)
        if world > 1:
            holder = [stats, new_records]
            dist.broadcast_object_list(holder, src=0)
            stats, new_records = holder
        if new_records:
            action_sampler.add(new_records, on_policy=True, update=update)
            train_action.extend(new_records)
            stats['hard_buckets'] = action_sampler.hard_sizes()

        stats['seconds'] = round(time.time() - t_start, 1)
        if is_main:
            stats['gen_seconds'] = round(t_gen, 1)
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(stats) + '\n')
            ev = ''
            if 'eval_acc' in stats:
                ev = (f' eval_acc={stats["eval_acc"]:.3f} eval_viol={stats.get("eval_violation_frac", 0):.3f}'
                      f' eval_bacc={stats.get("eval_balanced_acc", 0):.3f}'
                      f' eval_jparse={stats.get("eval_judge_parse_rate", 0):.2f}')
            print(f'[update {update}] reward={stats["mean_reward"]:.3f} '
                  f'act: acc={stats.get("action_acc", 0):.3f} viol={stats.get("action_violation_frac", 0):.3f} '
                  f'wait={stats.get("action_wait_frac", 0):.2f} parse={stats.get("action_parse_rate", 0):.3f} '
                  f'degen={degenerate["action"]}/{num_groups["action"]} | '
                  f'judge: bacc={stats.get("judge_balanced_acc", 0):.3f} yes={stats.get("judge_yes_rate", 0):.2f} '
                  f'parse={stats.get("judge_parse_rate", 0):.3f} degen={degenerate["judge"]}/{num_groups["judge"]} | '
                  f'ent={stats.get("entropy", 0.0):.2f} kl={stats.get("approx_kl", 0.0):.4f} '
                  f'clip={stats.get("clip_frac", 0.0):.2f} long={stats["skipped_long"]}{ev} '
                  f'({stats["seconds"]}s, gen {stats["gen_seconds"]}s)', flush=True)
            model.save_pretrained(os.path.join(args.out, 'adapter_latest'))
            if args.save_every and (update + 1) % args.save_every == 0:
                model.save_pretrained(os.path.join(args.out, f'adapter_update_{update + 1:04d}'))

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
        sys.stderr.flush(); sys.stdout.flush()
        os._exit(1)
