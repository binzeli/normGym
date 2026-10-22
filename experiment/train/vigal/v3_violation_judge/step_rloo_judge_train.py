"""Version 3 — violation judge: RLOO on "does this move violate the norm?".

Self-contained re-implementation of the v1/v2 recipe (nothing imported from
experiment/train/vigal or v2_belief_rloo; only the version-neutral layer:
config, modeling, policy_client, rollout_pool.sync_lora/wait_for_server):

- prompts_per_update independent records (harvest_judge.py), group_size
  samples each at temperature 1.0; batches class-balanced (pos_frac) and
  drawn without replacement per class;
- reward per sample: 1 if the yes/no matches the env's verdict else 0,
  + 0.1 if the JSON parsed (judge_task.grade) — the paper's reward;
- advantage: leave-one-out, A_i = r_i - mean_{j != i} r_j, no std-norm;
- loss: OpenRLHF PolicyLoss (per-token ratio clamp 3, clip 0.2, masked mean
  per sequence, mean over the minibatch), 1 epoch, `minibatches` steps;
- no KL, vision tower frozen (LoRA on the LM projections), lr 3e-5;
- init: base model with a fresh LoRA (default) or --init_adapter.

No episode eval and no on-policy re-harvest: the policy never acts, so the
data distribution does not depend on it. Held-out eval = greedy decode on
records of held-out episodes: accuracy, balanced accuracy, yes-rate, parse
rate, per-norm accuracy; the unrewarded `belief` field is stored for later
judging. --zero_shot N: sample group_size responses on N held-out records
with the current server model and report parse / balanced accuracy /
in-group disagreement, then exit (the "does the base model give RLOO a
signal" check).
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

from experiment.train.config import BASE_MODEL, CKPT_DIR
from experiment.train.modeling import (
    build_messages, collate_turns, encode_full, load_model, load_processor, token_batches,
)
from experiment.train.config import ROLLOUT_MAX_STEPS, ROLLOUT_SCRATCH_DIR, ROLLOUT_SEED_HIGH, ROLLOUT_SEED_LOW, TRAIN_NORMS, benchmark_capacity
from experiment.train.policy_client import HFBackend, OpenAIBackend
from experiment.train.rollout import cleanup_episode
from experiment.train.rollout_pool import RolloutPool, sync_lora, wait_for_server
from experiment.train.vigal.v3_violation_judge.harvest_judge import JUDGE_DATA_DIR
from experiment.train.vigal.v3_violation_judge.judge_task import build_user_prompt, grade, parse_response

V3_OUTPUT_DIR = os.path.join(CKPT_DIR, 'v3_violation_judge', 'default')

DEFAULTS = dict(
    prompts_per_update=64,
    group_size=4,
    num_updates=200,
    lr=1e-5,                  # 3e-5 until 2026-08-28 (collapse after one Adam-amplified step)
    clip_eps=0.2,
    ratio_clamp=3.0,
    rloo_epochs=1,
    minibatches=4,
    kl_beta=0.0,
    ent_coef=0.0,
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=1024,
    adv_norm=0,
    correct_reward=1.0,
    format_reward=0.1,
    pos_frac=0.4,             # violations per batch (they are ~2% of the harvest)
    min_step=50,              # drop records before this env step — too little
                              # evidence (feedback/history) to judge (2026-09-06,
                              # matching v2's identification gate)
    show_feedback=1,          # include the red-cross feedback line in the prompt
    eval_every=5,
    eval_frac=0.05,           # held-out EPISODES
    eval_max=400,             # held-out records evaluated (balanced)
    save_every=0,
    zero_shot=0,
    hard_frac=0.25,           # share of each batch from records the policy got wrong recently (0.5 until 2026-08-28)
    hard_pool_max=4000,
    hard_max_age=40,          # drop hard records older than 40 updates (0 = never)
    # Transfer test (user request 2026-08-27): the judge-trained model ACTS as
    # the pipeline's subject for full episodes on fixed scenarios.
    episode_eval_every=20,    # 0 = off (2026-08-28: 16 episodes every 20 updates, same cadence/sample as v1/v2's re-harvest game stats)
    episode_eval_n=16,
    episode_eval_temperature=0.7,
    episode_eval_max_steps=ROLLOUT_MAX_STEPS,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', default=os.path.join(JUDGE_DATA_DIR, 'train'))
    p.add_argument('--base_model', default=BASE_MODEL)
    p.add_argument('--init_adapter', default=None)
    p.add_argument('--out', default=V3_OUTPUT_DIR)
    p.add_argument('--rollout_url', default=None)
    p.add_argument('--rollout_model_name', default='policy')
    p.add_argument('--zero_shot_model_name', default='base')
    p.add_argument('--lora_name', default='policy')
    p.add_argument('--num_workers', type=int, default=64)
    p.add_argument('--start_update', type=int, default=0)
    p.add_argument('--train_batch_tokens', type=int, default=0)
    p.add_argument('--gradient_checkpointing', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_turn_tokens', type=int, default=8192)
    p.add_argument('--norms', nargs='*', default=None)
    p.add_argument('--worlds', nargs='+', default=['cafe'])
    p.add_argument('--capacity', type=int, default=0,
                   help="episode-eval crowd size; 0 = each norm's benchmark "
                        "capacity from experiment_config.json")
    p.add_argument('--scratch', default=os.path.join(ROLLOUT_SCRATCH_DIR, 'v3_episode_eval'))
    for k, v in DEFAULTS.items():
        p.add_argument(f'--{k}', type=type(v), default=v)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_records(root, norms=None, min_step=0):
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
            recs.append(r)
    if missing:
        print(f'[data] {missing} record(s) skipped: image missing', flush=True)
    return recs


def split_eval(records, eval_frac, seed):
    episodes = sorted({(r['norm'], r['world'], r['seed']) for r in records})
    rng = random.Random(seed + 99)
    rng.shuffle(episodes)
    held = set(episodes[:int(round(len(episodes) * eval_frac))])
    return ([r for r in records if (r['norm'], r['world'], r['seed']) not in held],
            [r for r in records if (r['norm'], r['world'], r['seed']) in held])


class BalancedSampler:
    """hard_frac of each batch from the hard pool (records where a sample was
    wrong in a recent update), the rest class-balanced (pos_frac violations).
    The hard pool is itself class-balanced at the same pos_frac (2026-08-28:
    it had drifted to negatives and pulled run 125557 into answering "no" to
    everything) and age-capped (hard_max_age updates; FIFO-capped at
    hard_pool_max). Each pool is iterated without replacement and
    reshuffled when exhausted."""

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
            label = 1 - label                     # the other class if this one is empty
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
        return batch, {'hard': n_hard, 'hard_pos': min(n_hard_pos, n_hard),
                       'pos': n_pos, 'neg': rest - n_pos,
                       'hard_pool': n_hard_total, 'hard_pool_pos': len(self.hard[1])}


def messages_for(rec, args):
    return build_messages(rec['system_prompt'], build_user_prompt(rec, bool(args.show_feedback)),
                          [rec['image_abs']])


def generate_many(backend, recs, args, temperature, num_workers):
    def one(rec):
        try:
            return backend.generate(messages_for(rec, args), max_new_tokens=args.max_new_tokens,
                                    temperature=temperature)
        except Exception as exc:
            print(f'[gen] failed for {rec["id"]}: {exc}', flush=True)
            return None
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        return list(ex.map(one, recs))


def grade_rec(rec, resp, args):
    return grade(rec['label'], resp, correct_reward=args.correct_reward,
                 format_reward=args.format_reward, truncated=False)


def loo_advantages(rewards):
    n, total = len(rewards), sum(rewards)
    return [r - (total - r) / (n - 1) for r in rewards]


def summarize(grades, recs):
    n = max(len(grades), 1)
    cats = {}
    for g in grades:
        cats[g['category']] = cats.get(g['category'], 0) + 1
    tp, tn, fp, fn = (cats.get(k, 0) for k in ('tp', 'tn', 'fp', 'fn'))
    pos = tp + fn; neg = tn + fp
    out = {
        'mean_reward': sum(g['reward'] for g in grades) / n,
        'acc': (tp + tn) / n,
        'balanced_acc': 0.5 * ((tp / pos if pos else 0.0) + (tn / neg if neg else 0.0)),
        'yes_rate': sum(1 for g in grades if g.get('pred') == 1) / n,
        'recall': tp / pos if pos else 0.0,
        'precision': tp / (tp + fp) if (tp + fp) else 0.0,
        'parse_rate': sum(g['parsed'] for g in grades) / n,
        'pos_frac': pos / n,
    }
    by_norm = {}
    for g, r in zip(grades, recs):
        by_norm.setdefault(r['norm'], []).append(g['correct'])
    out['acc_by_norm'] = {k: round(sum(v) / len(v), 3) for k, v in by_norm.items()}
    return out


def balanced_subset(records, n, rng):
    pos = [r for r in records if r['label']]; neg = [r for r in records if not r['label']]
    rng.shuffle(pos); rng.shuffle(neg)
    k = min(len(pos), n // 2)
    return pos[:k] + neg[:max(0, n - k)]


def run_eval(backend, records, args, out_path=None):
    responses = generate_many(backend, records, args, 0.0, args.num_workers)
    grades = [grade_rec(r, resp or '', args) for r, resp in zip(records, responses)]
    out = {'eval_' + k: v for k, v in summarize(grades, records).items()}
    out['eval_n'] = len(records)
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            for r, resp, g in zip(records, responses, grades):
                f.write(json.dumps({'id': r['id'], 'norm': r['norm'], 'label': r['label'],
                                    'pred': g.get('pred'), 'belief': g.get('belief', ''),
                                    'response': resp}) + '\n')
    return out


def episode_eval_scenarios(args, norms):
    rng = random.Random(args.seed + 777)
    return [(norms[i % len(norms)], rng.choice(args.worlds), rng.randint(ROLLOUT_SEED_LOW, ROLLOUT_SEED_HIGH))
            for i in range(args.episode_eval_n)]


def run_episode_eval(pool, scenarios, args, update):
    """Transfer test: the current LoRA plays the pipeline's memory_llm subject
    (its normal acting prompt) for full episodes; true env rewards."""
    from experiment.agents.llm_agent import _parse_memory_response
    tasks = [dict(norm=n, world=w, seed=s,
                  workdir=os.path.join(args.scratch, f'eval_u{update:04d}', f'ep{i}'),
                  capacity=args.capacity or benchmark_capacity(n),
                  max_steps=args.episode_eval_max_steps,
                  gamma=1.0, format_bonus=0.0)
             for i, (n, w, s) in enumerate(scenarios)]
    episodes = [ep for ep in pool.run_batch(tasks) if ep is not None]
    if not episodes:
        return {'ep_eval_failed': len(tasks)}
    turns = [t for ep in episodes for t in ep.turns]
    wait = sum(1 for t in turns if (_parse_memory_response(t.response) or {}).get('action_text', 'wait') == 'wait')
    n = len(episodes)
    out = {'ep_eval_n': n, 'ep_eval_failed': len(tasks) - n,
           'ep_eval_return': sum(e.env_return for e in episodes) / n,
           'ep_eval_violations': sum(e.num_violations for e in episodes) / n,
           'ep_eval_collisions': sum(e.num_collisions for e in episodes) / n,
           'ep_eval_goals': sum(e.goals_reached for e in episodes) / n,
           'ep_eval_success': sum(e.task_success for e in episodes) / n,
           'ep_eval_violation_free_success': sum(e.task_success and e.num_violations == 0 for e in episodes) / n,
           'ep_eval_turn_parse': sum(e.turns_parsed for e in episodes) / max(1, sum(e.turns_total for e in episodes)),
           'ep_eval_wait_frac': wait / max(1, len(turns)),
           'ep_eval_by_norm': {}}
    for ep in episodes:
        d = out['ep_eval_by_norm'].setdefault(ep.norm_type, {'return': 0.0, 'violations': 0.0, 'success': 0.0, 'n': 0})
        d['return'] += ep.env_return; d['violations'] += ep.num_violations; d['success'] += ep.task_success; d['n'] += 1
    for d in out['ep_eval_by_norm'].values():
        k = d.pop('n'); d.update({x: round(d[x] / k, 2) for x in d})
    for ep in episodes:
        cleanup_episode(ep)
    return out


def zero_shot_check(backend, records, args):
    """group_size samples per record: parse rate, balanced accuracy of the
    samples, and the share of groups whose samples disagree (RLOO signal)."""
    jobs = [r for r in records for _ in range(args.group_size)]
    responses = generate_many(backend, jobs, args, args.temperature, args.num_workers)
    grades = [grade_rec(r, resp or '', args) for r, resp in zip(jobs, responses)]
    stats = summarize(grades, jobs)
    K = args.group_size
    disagree = sum(1 for i in range(0, len(grades), K)
                   if len({g['reward'] for g in grades[i:i + K]}) > 1)
    stats['group_disagreement'] = disagree / max(1, len(records))
    stats['n_records'] = len(records)
    return stats


# ---------------------------------------------------------------------------
# Loss
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
    loss = -torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    loss = masked_mean(loss, mask)
    with torch.no_grad():
        log_ratio = new_logp - old_logp
        approx_kl = masked_mean(ratio - 1 - log_ratio, mask)
        clip_frac = masked_mean(((ratio - 1).abs() > clip_eps).float(), mask)
    return loss, approx_kl, clip_frac


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
        sys.exit('[FATAL] no CUDA')
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
    sample_rng = random.Random(args.seed + 1234)
    shuffle_rng = random.Random(args.seed + 4321)
    eval_rng = random.Random(args.seed + 777)

    records = load_records(args.records, args.norms, min_step=args.min_step)
    if not records:
        sys.exit(f'[FATAL] no records under {args.records}'
                 + (f' at step >= {args.min_step}' if args.min_step else ''))
    train_recs, eval_all = split_eval(records, args.eval_frac, args.seed)
    eval_recs = balanced_subset(eval_all, args.eval_max, eval_rng)
    sampler = BalancedSampler(train_recs, sample_rng, args.pos_frac,
                              hard_frac=args.hard_frac, hard_pool_max=args.hard_pool_max,
                              hard_max_age=args.hard_max_age)
    say(f'[data] {len(train_recs)} train records at step >= {args.min_step} '
        f'({sum(r["label"] for r in train_recs)} violations), '
        f'{len(eval_all)} held-out ({len(eval_recs)} evaluated, '
        f'{sum(r["label"] for r in eval_recs)} violations); per norm: '
        + str({n: sum(1 for r in train_recs if r['norm'] == n) for n in sorted({r['norm'] for r in train_recs})}))

    # ---------------- zero-shot check (rank 0, base model on the server) ----
    if args.zero_shot > 0:
        if not is_main:
            return
        wait_for_server(args.rollout_url)
        backend = OpenAIBackend(args.rollout_url, args.zero_shot_model_name, top_p=args.top_p)
        subset = balanced_subset(eval_all, args.zero_shot, eval_rng)
        stats = zero_shot_check(backend, subset, args)
        say('[zero-shot] ' + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                          for k, v in stats.items()}))
        with open(os.path.join(args.out, 'zero_shot.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        verdict = (stats['parse_rate'] >= 0.9 and stats['balanced_acc'] >= 0.55
                   and stats['group_disagreement'] >= 0.2)
        say(f'[zero-shot] {"OK: RLOO from the base model has signal" if verdict else "WEAK: consider a mini-SFT"}')
        return

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
    optimizer = torch.optim.AdamW(policy_params, lr=args.lr)
    best_key = -1.0
    if args.start_update > 0 and os.path.exists(os.path.join(args.out, 'best_metric.json')):
        # resume: keep adapter_best unless the resumed run actually beats the recorded best
        with open(os.path.join(args.out, 'best_metric.json')) as f:
            prev_best = json.load(f)
        best_key = float(prev_best.get('eval_balanced_acc', -1.0))
        if is_main:
            print(f'[resume] best_metric.json: eval_bacc={best_key:.3f} @u{prev_best.get("update")} '
                  f'(adapter_best kept until beaten)', flush=True)

    eval_pool, eval_scenarios = None, []
    if is_main and args.episode_eval_every > 0:
        if not args.rollout_url:
            sys.exit('[FATAL] --episode_eval_every needs --rollout_url (0 disables)')
        eval_scenarios = episode_eval_scenarios(args, args.norms or TRAIN_NORMS)
        os.makedirs(args.scratch, exist_ok=True)
        eval_pool = RolloutPool(num_workers=args.episode_eval_n, rollout_url=args.rollout_url,
                                model_name=args.lora_name, temperature=args.episode_eval_temperature,
                                max_new_tokens=args.max_new_tokens, top_p=args.top_p)
        say(f'[eval] transfer episode eval every {args.episode_eval_every} updates: '
            f'{len(eval_scenarios)} fixed scenarios {[(n, s) for n, _, s in eval_scenarios]}')

    for update in range(args.start_update, args.num_updates):
        t_start = time.time()
        batch, composition = sampler.next_batch(min(args.prompts_per_update, len(train_recs)), update=update)
        jobs = [rec for rec in batch for _ in range(args.group_size)]

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

        proto, grades, recs_of = [], [], []
        degenerate = num_groups = 0
        K = args.group_size
        for gi, rec in enumerate(batch):
            members = [m for m in payload[gi * K:(gi + 1) * K] if m is not None]
            if len(members) < 2:
                continue
            num_groups += 1
            rewards = [g['reward'] for _, g in members]
            if max(rewards) - min(rewards) < 1e-8:
                degenerate += 1
            for (resp, g), adv in zip(members, loo_advantages(rewards)):
                proto.append([adv, rec, resp]); grades.append(g); recs_of.append(rec)
        if not proto:
            say(f'[update {update}] no usable groups; skipping')
            continue
        if args.adv_norm:
            std = torch.tensor([p[0] for p in proto]).std().item()
            if std > 1e-8:
                for p in proto:
                    p[0] /= std
        if args.hard_frac > 0:
            wrong = {id(rec) for g, rec in zip(grades, recs_of) if not g['correct']}
            sampler.add_hard([rec for rec in batch if id(rec) in wrong], update=update)
        stats = {'update': update, 'num_prompts': len(batch), 'batch_composition': composition,
                 'num_groups': num_groups, 'degenerate_groups': degenerate, 'num_samples': len(proto)}
        stats.update(summarize(grades, recs_of))

        # ---------------- encode + old log-probs (sharded) ------------------
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

        # ---------------- clipped update ------------------------------------
        if kept:
            model.train()
            acc = torch.zeros(5, dtype=torch.float64, device=device)
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
                            acc += torch.stack([loss.sum(), kl.sum(), clip.sum(), masked_mean(ents, mask).sum(),
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

        # ---------------- held-out eval (greedy) ----------------------------
        if is_main and eval_recs and args.eval_every > 0 and (
                (update + 1) % args.eval_every == 0 or update == args.start_update):
            stats.update(run_eval(backend, eval_recs, args,
                                  out_path=os.path.join(args.out, f'eval_u{update:04d}.jsonl')))
            if stats['eval_balanced_acc'] > best_key:
                best_key = stats['eval_balanced_acc']
                model.save_pretrained(os.path.join(args.out, 'adapter_best'))
                with open(os.path.join(args.out, 'best_metric.json'), 'w') as f:
                    json.dump({'update': update, **{k: v for k, v in stats.items() if k.startswith('eval')}}, f, indent=2)
        if is_main and eval_pool is not None and (update + 1) % args.episode_eval_every == 0:
            t_ev = time.time()
            stats.update(run_episode_eval(eval_pool, eval_scenarios, args, update))
            stats['ep_eval_seconds'] = round(time.time() - t_ev, 1)
            if 'ep_eval_success' in stats:
                print(f'[eval u{update}] transfer: n={stats["ep_eval_n"]} return={stats["ep_eval_return"]:.2f} '
                      f'success={stats["ep_eval_success"]:.2f} vf_success={stats["ep_eval_violation_free_success"]:.2f} violations/ep={stats["ep_eval_violations"]:.2f} '
                      f'coll/ep={stats["ep_eval_collisions"]:.2f} goals/ep={stats["ep_eval_goals"]:.2f} '
                      f'tparse={stats["ep_eval_turn_parse"]:.3f} wait={stats["ep_eval_wait_frac"]:.2f} '
                      f'by_norm={stats["ep_eval_by_norm"]} ({stats["ep_eval_seconds"]}s)', flush=True)
        if world > 1:
            holder = [stats]
            dist.broadcast_object_list(holder, src=0)
            stats = holder[0]

        stats['seconds'] = round(time.time() - t_start, 1)
        if is_main:
            stats['gen_seconds'] = round(t_gen, 1)
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(stats) + '\n')
            ev = (f' eval_acc={stats["eval_acc"]:.3f} eval_bacc={stats["eval_balanced_acc"]:.3f} '
                  f'eval_yes={stats["eval_yes_rate"]:.2f} eval_parse={stats["eval_parse_rate"]:.2f}'
                  if 'eval_acc' in stats else '')
            print(f'[update {update}] reward={stats["mean_reward"]:.3f} acc={stats["acc"]:.3f} '
                  f'hard={composition.get("hard", 0)}/{len(batch)} '
                  f'bacc={stats["balanced_acc"]:.3f} yes={stats["yes_rate"]:.2f} '
                  f'recall={stats["recall"]:.2f} prec={stats["precision"]:.2f} '
                  f'parse={stats["parse_rate"]:.3f} '
                  f'degen={stats["degenerate_groups"]}/{stats["num_groups"]} '
                  f'ent={stats.get("entropy", 0.0):.2f} kl={stats.get("approx_kl", 0.0):.4f} '
                  f'clip={stats.get("clip_frac", 0.0):.2f}{ev} '
                  f'({stats["seconds"]}s, gen {stats["gen_seconds"]}s)', flush=True)
            model.save_pretrained(os.path.join(args.out, 'adapter_latest'))
            if args.save_every and (update + 1) % args.save_every == 0:
                model.save_pretrained(os.path.join(args.out, f'adapter_update_{update + 1:04d}'))
    if eval_pool is not None:
        eval_pool.close()
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
