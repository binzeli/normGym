"""SPIRAL v3 — violation judge with RAE baselines instead of RLOO groups.

Same task, same data, same reward as vigal v3 (the task definition and the
harvested records are shared assets — ``judge_task`` and
``data/<group>/violation_judge/`` are imported/reused, only the RL recipe is
new): the policy is an observer answering "does this move violate the
group's norm?", reward = 1/0 on the env's verdict + 0.1 format
(judge_task.grade, repetition guard included).

What changes (the SPIRAL stack, see spiral/common.py):

- ONE sample per record instead of K=4 groups. The advantage is
  A = r - b_(norm,label): a per-norm-per-class EMA baseline (RAE; the class
  plays the paper's "role" — yes-records and no-records have different
  expected rewards, exactly the asymmetry RAE conditions on). The vigal v3
  collapse ("no" to everything, 120 dead updates) was absorbing because
  all-equal groups give RLOO zero gradient; against an EMA baseline a
  collapsed policy scores below its own past and gets a restoring gradient.
- lr 1e-6 constant + grad clip 1.0 (SPIRAL Table 6), temperature 1.0, no KL.
- The hard-example fraction is a difficulty LADDER driven by a controller
  holding train accuracy in a band — the single-player auto-curriculum —
  instead of a fixed hard_frac. The hard pool stays class-balanced at
  pos_frac and age-capped (the vigal v3 lessons).
- Collapse guard: halt after ``halt_patience`` consecutive updates with
  parse < 0.5 or with the answer distribution pinned (yes_frac 0 or 1).

Held-out eval: greedy on class-balanced records of held-out episodes ->
eval_bacc drives ``adapter_best``. The 16-episode transfer eval of vigal v3
is not duplicated here — run it offline from the checkpoints if needed.

Launchers retired 2026-09-11 (not maintained).
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch
import torch.distributed as dist

from experiment.train.config import BASE_MODEL, CKPT_DIR, NORM_GROUP
from experiment.train.modeling import (
    build_messages,
    collate_turns,
    encode_full,
    load_model,
    load_processor,
    token_batches,
)
from experiment.train.policy_client import OpenAIBackend
from experiment.train.rollout_pool import sync_lora, wait_for_server
from experiment.train.spiral.common import (
    DifficultyController,
    EmaBaselines,
    generate_many,
    load_state,
    masked_mean,
    policy_loss,
    response_token_logprobs,
    save_state,
)
from experiment.train.vigal.v3_violation_judge.judge_task import build_user_prompt, grade

V3_OUTPUT_DIR = os.path.join(CKPT_DIR, 'spiral_v3_judge', NORM_GROUP, 'run1')
V3_RECORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', '..', 'data', NORM_GROUP, 'violation_judge', 'train')

DEFAULTS = dict(
    prompts_per_update=256,   # one sample each == the 64x4 generation budget
    num_updates=300,
    lr=1e-6,                  # SPIRAL Table 6 (base-model init, like vigal v3)
    clip_eps=0.2,
    ratio_clamp=3.0,
    epochs=1,
    minibatches=4,
    grad_clip=1.0,
    kl_beta=0.0,
    ent_coef=0.0,
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=1024,
    correct_reward=1.0,
    format_reward=0.1,
    pos_frac=0.4,             # violations per batch (they are ~4% of the harvest)
    show_feedback=1,
    baseline_decay=0.95,
    # Curriculum: the hard-example share steps up when train accuracy leaves
    # the band upward, down when it drops below it.
    hard_ladder='0.0,0.25,0.5',
    ctrl_lo=0.55,
    ctrl_hi=0.8,
    ctrl_decay=0.9,
    ctrl_min_samples=3,       # updates at a level before it may change
    ctrl_start_level=1,       # start at hard_frac 0.25 (the vigal default)
    hard_pool_max=4000,
    hard_max_age=40,
    eval_every=5,
    eval_frac=0.05,           # held-out EPISODES
    eval_max=400,             # held-out records evaluated (class-balanced)
    save_every=25,
    halt_on_collapse=1,
    halt_patience=2,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', default=os.path.normpath(V3_RECORDS_DIR))
    p.add_argument('--base_model', default=BASE_MODEL)
    p.add_argument('--init_adapter', default=None,
                   help='default: base model + fresh LoRA (the vigal v3 choice)')
    p.add_argument('--out', default=V3_OUTPUT_DIR)
    p.add_argument('--rollout_url', required=True)
    p.add_argument('--rollout_model_name', default='policy')
    p.add_argument('--lora_name', default='policy')
    p.add_argument('--num_workers', type=int, default=64)
    p.add_argument('--start_update', type=int, default=0)
    p.add_argument('--resume_state', type=int, default=1)
    p.add_argument('--train_batch_tokens', type=int, default=20000)
    p.add_argument('--gradient_checkpointing', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_turn_tokens', type=int, default=8192)
    p.add_argument('--norms', nargs='*', default=None)
    for k, v in DEFAULTS.items():
        p.add_argument(f'--{k}', type=type(v), default=v)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data (same conventions as vigal v3: records.jsonl + images/)
# ---------------------------------------------------------------------------

def load_records(root, norms=None):
    recs, missing = [], 0
    with open(os.path.join(root, 'records.jsonl'), encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if norms and r['norm'] not in norms:
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
    """Class-balanced batches (pos_frac violations) with a class-balanced,
    age-capped hard pool — the vigal v3 sampler, with the hard share supplied
    per batch by the difficulty controller instead of being fixed."""

    def __init__(self, records, rng, pos_frac, hard_pool_max=4000, hard_max_age=0):
        self.rng, self.pos_frac = rng, pos_frac
        self.pools = {1: [r for r in records if r['label']],
                      0: [r for r in records if not r['label']]}
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
            self.hard_order[label] = list(self.hard[label])
            self.rng.shuffle(self.hard_order[label])
        return self.hard_order[label].pop()

    def add_hard(self, records, update=0):
        for r in records:
            if r['id'] in self.hard_ids:
                continue
            r['_hard_added'] = update
            c = 1 if r['label'] else 0
            self.hard[c].append(r)
            self.hard_ids.add(r['id'])
            self.hard_order[c].insert(self.rng.randint(0, len(self.hard_order[c])), r)
        self._trim(update)

    def _trim(self, update):
        cutoff = update - self.hard_max_age if self.hard_max_age else None
        for c in (0, 1):
            keep = [r for r in self.hard[c]
                    if cutoff is None or r.get('_hard_added', 0) >= cutoff]
            while len(keep) > self.hard_pool_max // 2:
                keep.pop(0)
            dropped = {id(r) for r in self.hard[c]} - {id(r) for r in keep}
            if dropped:
                self.hard_ids -= {r['id'] for r in self.hard[c] if id(r) in dropped}
                self.hard[c] = keep
                self.hard_order[c] = [r for r in self.hard_order[c]
                                      if id(r) not in dropped]

    def next_batch(self, n, hard_frac, update=None):
        if update is not None:
            self._trim(update)
        n_hard_total = len(self.hard[0]) + len(self.hard[1])
        n_hard = int(round(n * hard_frac)) if n_hard_total else 0
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
        return batch, {'hard': n_hard, 'pos': n_pos + min(n_hard_pos, n_hard),
                       'hard_pool': n_hard_total, 'hard_pool_pos': len(self.hard[1])}


def messages_for(rec, args):
    return build_messages(rec['system_prompt'],
                          build_user_prompt(rec, bool(args.show_feedback)),
                          [rec['image_abs']])


def summarize(grades):
    n = max(len(grades), 1)
    yes = sum(g['pred'] == 1 for g in grades if g['pred'] is not None)
    return {
        'mean_reward': sum(g['reward'] for g in grades) / n,
        'acc': sum(g['correct'] for g in grades) / n,
        'parse_rate': sum(g['parsed'] for g in grades) / n,
        'yes_frac': yes / n,
    }


def balanced_accuracy(pairs):
    """pairs: (label, grade). Mean of per-class accuracy over graded records."""
    per_class = {}
    for label, g in pairs:
        per_class.setdefault(int(label), []).append(g['correct'])
    if not per_class:
        return 0.0
    return sum(sum(v) / len(v) for v in per_class.values()) / len(per_class)


def run_eval(backend, records, args, rng):
    """Greedy decode on a class-balanced slice of the held-out records."""
    pos = [r for r in records if r['label']]
    neg = [r for r in records if not r['label']]
    half = args.eval_max // 2
    sample = (rng.sample(pos, min(half, len(pos)))
              + rng.sample(neg, min(args.eval_max - min(half, len(pos)), len(neg))))
    responses = generate_many(backend, [messages_for(r, args) for r in sample],
                              0.0, args.max_new_tokens, args.num_workers)
    pairs = [(r['label'], grade(r['label'], resp or '',
                                correct_reward=args.correct_reward,
                                format_reward=args.format_reward))
             for r, resp in zip(sample, responses)]
    out = {'eval_' + k: v for k, v in summarize([g for _, g in pairs]).items()}
    out['eval_balanced_acc'] = balanced_accuracy(pairs)
    out['eval_n'] = len(sample)
    by_norm = {}
    for r, (_, g) in zip(sample, pairs):
        by_norm.setdefault(r['norm'], []).append(g['correct'])
    out['eval_acc_by_norm'] = {n: round(sum(v) / len(v), 3)
                               for n, v in by_norm.items()}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    hard_ladder = [float(x) for x in str(args.hard_ladder).split(',')]

    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_main = rank == 0

    if not torch.cuda.is_available():
        sys.exit('[FATAL] torch.cuda.is_available() is False; refusing to train on CPU.')
    if world > 1:
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
    state_path = os.path.join(args.out, 'spiral_state.json')

    torch.manual_seed(args.seed)
    sample_rng = random.Random(args.seed + 1234)   # same draws on every rank
    shuffle_rng = random.Random(args.seed + 4321)
    eval_rng = random.Random(args.seed + 99)

    records = load_records(args.records, args.norms)
    if not records:
        sys.exit(f'[FATAL] no records under {args.records} — run '
                 f'vigal/v3_violation_judge/harvest_judge.py first')
    train_recs, eval_recs = split_eval(records, args.eval_frac, args.seed)
    sampler = BalancedSampler(train_recs, sample_rng, args.pos_frac,
                              hard_pool_max=args.hard_pool_max,
                              hard_max_age=args.hard_max_age)
    say(f'[data] {len(train_recs)} train / {len(eval_recs)} eval records '
        f'({sum(1 for r in train_recs if r["label"])} violations in train)')

    baselines = EmaBaselines(decay=args.baseline_decay)
    controller = DifficultyController(
        ['all'], [{'hard_frac': h} for h in hard_ladder],
        lo=args.ctrl_lo, hi=args.ctrl_hi, decay=args.ctrl_decay,
        min_samples=args.ctrl_min_samples, start_level=args.ctrl_start_level)
    if args.start_update > 0 and args.resume_state:
        restored = load_state(state_path, baselines, controller)
        say(f'[state] restored spiral_state.json from update {restored}'
            if restored is not None else '[state] no spiral_state.json to restore')
    say(f'[spiral] hard ladder={hard_ladder} band=[{args.ctrl_lo},{args.ctrl_hi}] '
        f'baseline_decay={args.baseline_decay}')

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
    visual_trainable = [n for n, p in model.named_parameters()
                        if p.requires_grad and '.visual.' in n]
    if visual_trainable:
        sys.exit(f'[FATAL] {len(visual_trainable)} trainable vision params '
                 f'(e.g. {visual_trainable[0]}); the vision tower stays frozen')
    if world > 1:
        for p in policy_params:
            dist.broadcast(p.data, src=0)

    backend = None
    serving_dir = os.path.join(args.out, 'adapter_serving')
    if is_main:
        wait_for_server(args.rollout_url)
        model.save_pretrained(serving_dir)
        sync_lora(args.rollout_url, serving_dir, args.lora_name)
        backend = OpenAIBackend(args.rollout_url, args.rollout_model_name,
                                top_p=args.top_p)

    optimizer = torch.optim.AdamW(policy_params, lr=args.lr)
    best_metric = float('-inf')
    collapse_streak = 0

    for update in range(args.start_update, args.num_updates):
        t_start = time.time()
        hard_frac = controller.overrides('all')['hard_frac']
        batch, composition = sampler.next_batch(
            min(args.prompts_per_update, len(train_recs)), hard_frac, update=update)

        # ---------------- One sample per record + grade (rank 0) -----------
        payload = None
        if is_main:
            t_gen = time.time()
            responses = generate_many(backend, [messages_for(r, args) for r in batch],
                                      args.temperature, args.max_new_tokens,
                                      args.num_workers)
            payload = {'gen_seconds': round(time.time() - t_gen, 1),
                       'grades': [grade(rec['label'], resp or '',
                                        correct_reward=args.correct_reward,
                                        format_reward=args.format_reward)
                                  if resp is not None else None
                                  for rec, resp in zip(batch, responses)],
                       'responses': responses}
        if world > 1:
            holder = [payload]
            dist.broadcast_object_list(holder, src=0)
            payload = holder[0]

        # ---------------- RAE advantages (per norm x class) ----------------
        proto, grades, wrong = [], [], []
        for rec, g, resp in zip(batch, payload['grades'], payload['responses']):
            if g is None:
                continue
            key = f"{rec['norm']}|{int(bool(rec['label']))}"
            adv = baselines.advantage(key, g['reward'])
            proto.append([adv, rec, resp])
            grades.append(g)
            if not g['correct']:
                wrong.append(rec)
        if not proto:
            say(f'[update {update}] nothing graded; skipping')
            continue
        sampler.add_hard(wrong, update=update)

        stats = {'update': update, 'num_prompts': len(batch),
                 'batch_composition': composition, 'hard_frac': hard_frac,
                 'num_samples': len(proto),
                 'mean_advantage': sum(p[0] for p in proto) / len(proto),
                 'gen_seconds': payload['gen_seconds']}
        stats.update(summarize(grades))
        controller.record('all', stats['acc'])
        changes = controller.step()
        for line in changes:
            say(f'[curriculum u{update}] {line}')
        stats['curriculum_changes'] = changes

        # ---------------- Encode + old per-token log-probs (sharded) -------
        model.eval()
        local = {}
        for g, (_, rec, resp) in enumerate(proto):
            if g % world != rank:
                continue
            encoded = encode_full(processor, messages_for(rec, args), resp)
            local[g] = (None if encoded['input_ids'].shape[1] > args.max_turn_tokens
                        else {'encoded': encoded})
        with torch.no_grad():
            kept_local = [g for g in sorted(local) if local[g] is not None]
            for batch_idx in token_batches(
                    [(None, local[g]['encoded']) for g in kept_local],
                    args.train_batch_tokens):
                gids = [kept_local[k] for k in batch_idx]
                logps, mask, _ = response_token_logprobs(
                    model, collate_turns([local[g]['encoded'] for g in gids]), device)
                for j, g in enumerate(gids):
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
        kept = [g for g in range(len(proto)) if meta.get(g)]
        stats['num_train_samples'] = len(kept)
        stats['skipped_long'] = len(proto) - len(kept)

        # ---------------- Clipped-surrogate epoch(s) -----------------------
        if kept:
            model.train()
            acc = torch.zeros(5, dtype=torch.float64, device=device)
            n_samples = len(kept)
            for _epoch in range(args.epochs):
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
                    torch.nn.utils.clip_grad_norm_(policy_params, args.grad_clip)
                    optimizer.step()
            optimizer.zero_grad()
            if world > 1:
                dist.all_reduce(acc)
            total = max(acc[4].item(), 1.0)
            stats['policy_loss'] = acc[0].item() / total
            stats['approx_kl'] = acc[1].item() / total
            stats['clip_frac'] = acc[2].item() / total
            stats['entropy'] = acc[3].item() / total
            if is_main:
                model.save_pretrained(serving_dir)
                sync_lora(args.rollout_url, serving_dir, args.lora_name)
        else:
            say(f'[update {update}] no trainable samples')

        # ---------------- Held-out eval (greedy, balanced) -----------------
        if is_main and eval_recs and args.eval_every > 0 and (
                (update + 1) % args.eval_every == 0 or update == args.start_update):
            stats.update(run_eval(backend, eval_recs, args, eval_rng))
            if stats['eval_balanced_acc'] > best_metric:
                best_metric = stats['eval_balanced_acc']
                model.save_pretrained(os.path.join(args.out, 'adapter_best'))
                with open(os.path.join(args.out, 'best_metric.json'), 'w') as f:
                    json.dump({'update': update,
                               **{k: v for k, v in stats.items()
                                  if k.startswith('eval')}}, f, indent=2)
                say(f'[eval u{update}] eval_bacc={best_metric:.3f} '
                    f'new best -> adapter_best')

        # ---------------- Log / save / guard -------------------------------
        stats['baselines'] = {k: round(v, 3) for k, v in baselines.values.items()}
        stats['seconds'] = round(time.time() - t_start, 1)
        if is_main:
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(stats) + '\n')
            ev = (f' eval_bacc={stats["eval_balanced_acc"]:.3f}'
                  if 'eval_balanced_acc' in stats else '')
            say(f'[update {update}] reward={stats["mean_reward"]:.3f} '
                f'acc={stats["acc"]:.3f} adv={stats["mean_advantage"]:+.3f} '
                f'parse={stats["parse_rate"]:.3f} yes={stats["yes_frac"]:.2f} '
                f'hard={hard_frac:.2f} ent={stats.get("entropy", 0.0):.2f} '
                f'kl={stats.get("approx_kl", 0.0):.4f}{ev} '
                f'({stats["seconds"]}s, gen {stats["gen_seconds"]}s)')
            model.save_pretrained(os.path.join(args.out, 'adapter_latest'))
            save_state(state_path, baselines, controller, update)
            if args.save_every and (update + 1) % args.save_every == 0:
                model.save_pretrained(os.path.join(args.out, f'adapter_update_{update + 1:04d}'))

        # Pinned = the vigal v3 collapse signature ("no" to everything) — but
        # only when parsing is healthy, so a parse slide isn't double-counted
        # and an all-unparsed batch (yes_frac 0 by construction) can't fake it.
        pinned = stats['parse_rate'] < 0.5 or (
            stats['parse_rate'] >= 0.9
            and (stats['yes_frac'] <= 0.01 or stats['yes_frac'] >= 0.99))
        collapse_streak = collapse_streak + 1 if pinned else 0
        if args.halt_on_collapse and collapse_streak >= args.halt_patience:
            say(f'[HALT] parse < 0.5 or answers pinned for {collapse_streak} '
                f'consecutive updates — stopping at u{update} (adapter_best is '
                f'the checkpoint to keep)')
            break

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
