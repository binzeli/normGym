"""SPIRAL-style episode-level trainer engine (shared by spiral v1/v2).

One update:
  1. Roll ``episodes_per_update`` FULL interactive-stage episodes through the
     unchanged pipeline (rollout.run_episode via RolloutPool + vLLM), sampled
     at temperature 1.0. Per-norm difficulty (crowd capacity) comes from the
     DifficultyController — the single-player stand-in for self-play's
     auto-curriculum (see spiral/common.py).
  2. The objective (v1: game outcome; v2: norm identification) turns each
     episode into ONE sparse terminal reward — SPIRAL's r != 0 only at
     termination. No reward shaping, no budgets, no per-step signal.
  3. RAE advantage: A_ep = R_ep - b_norm (per-norm EMA baseline, SPIRAL
     Eq. 2). Every trained turn of the episode carries A_ep (SPIRAL Eq. 3:
     trajectory-level credit — the norm evidence accrues across the episode's
     context, so the single-snapshot identifiability gap of the vigal
     trainers does not exist here).
  4. One epoch of clipped-surrogate updates over the turns (OpenRLHF
     PolicyLoss semantics; at one on-policy epoch this is REINFORCE with a
     safety clip), grad-clipped at 1.0, lr ~3e-6 constant. No KL by default
     (the SPIRAL recipe); ``--kl_beta`` restores the SFT anchor if wanted.

Held-out eval every ``eval_every`` updates: fixed (norm, world, seed)
scenarios at the benchmark capacity, sampled at the pipeline's eval
temperature; ``adapter_best`` tracks the objective's eval metric.

A parse-collapse guard halts the run after ``halt_patience`` consecutive
updates whose rollout turn-parse-rate is below 0.5 — the vigal runs burned
30-120 dead updates after their cliffs; with the EMA baseline the cliff
should not be absorbing in the first place, and if it happens anyway the run
stops instead of squatting on the GPUs.

This engine is not a script; v1/v2 provide the objective + defaults:
    experiment/train/spiral/v1_action/train.py
    experiment/train/spiral/v2_belief/train.py
"""
import json
import os
import random
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import torch.distributed as dist

from experiment.agents.llm_agent import _parse_memory_response
from experiment.train.config import (
    BASE_MODEL,
    ROLLOUT_MAX_STEPS,
    ROLLOUT_SCRATCH_DIR,
    ROLLOUT_SEED_HIGH,
    ROLLOUT_SEED_LOW,
    ROLLOUT_WORLDS,
    TRAIN_NORMS,
    load_experiment_config,
)
from experiment.train.modeling import (
    collate_turns,
    encode_full,
    load_model,
    load_processor,
    token_batches,
)
from experiment.train.policy_client import OpenAIBackend
from experiment.train.rollout import cleanup_episode
from experiment.train.rollout_pool import RolloutPool, sync_lora, wait_for_server
from experiment.train.spiral.common import (
    DifficultyController,
    EmaBaselines,
    load_state,
    masked_mean,
    policy_loss,
    response_token_logprobs,
    save_state,
)

# Shared defaults; each version's train.py overlays its own (reward knobs,
# metric, ladder). SPIRAL's optimizer discipline: small constant lr, grad
# clip 1.0, temperature 1.0, sparse terminal reward, no KL.
ENGINE_DEFAULTS = dict(
    episodes_per_update=16,
    num_updates=150,
    lr=3e-6,                  # SPIRAL: 1e-6 full fine-tune; LoRA runs a bit hotter
    clip_eps=0.2,
    ratio_clamp=3.0,
    kl_beta=0.0,              # SPIRAL: no KL. Set >0 to anchor to the init adapter.
    ent_coef=0.0,
    epochs=1,
    minibatches=4,
    grad_clip=1.0,
    temperature=1.0,          # SPIRAL rollout temperature
    top_p=1.0,
    max_new_tokens=1024,
    # RAE + curriculum
    baseline_decay=0.95,      # SPIRAL Table 6: EMA decay rate 0.95
    ctrl_lo=0.3,              # success band the controller holds each norm in
    ctrl_hi=0.6,
    ctrl_decay=0.9,
    ctrl_min_samples=6,       # episodes at a level before it may change
    ctrl_start_level=0,
    # Cost bound: turns trained per episode (uniform subsample, all carry the
    # same episode advantage, so the gradient estimate stays unbiased).
    # 0 = every turn (SPIRAL-faithful, ~3x the train-half cost at 90 steps).
    max_turns_per_episode=24,
    # Format gate (objective-level): an episode with more than this fraction
    # of unparseable turns scores the objective's loss reward outright —
    # SPIRAL's "invalid move loses the game".
    max_unparsed_frac=0.05,
    # Held-out eval
    eval_every=10,
    eval_episodes=8,
    eval_capacity=0,          # 0 = the norm's benchmark capacity from
                              # experiment_config.json (norm_capacity override,
                              # else the global 'capacity'); >0 forces a value
    eval_temperature=0.7,     # the pipeline's eval sampling setting
    # Collapse guard. ent_halt: entropy above this counts as a bad update
    # (295207's runaway crossed 1.5 at u9, eight updates before the parse
    # guard finally fired at ~3.0; healthy SFT-anchored sampling runs ~0.8).
    # 0 disables.
    halt_on_collapse=1,
    halt_patience=2,
    ent_halt=2.0,
    save_every=25,
)


def add_engine_args(parser, defaults):
    parser.add_argument('--base_model', default=BASE_MODEL)
    parser.add_argument('--init_adapter', default=None)
    parser.add_argument('--out', default=None,
                        help='output dir (each version sets its own default)')
    parser.add_argument('--rollout_url', required=True,
                        help='OpenAI-compatible vLLM endpoint (episode rollouts '
                             'and objective generation both go through it)')
    parser.add_argument('--rollout_model_name', default='policy')
    parser.add_argument('--lora_name', default='policy')
    parser.add_argument('--rollout_workers', type=int, default=0,
                        help='episode worker processes (0 = episodes_per_update)')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='concurrent generation threads for objective calls')
    parser.add_argument('--start_update', type=int, default=0)
    parser.add_argument('--resume_state', type=int, default=1,
                        help='restore baselines/controller from spiral_state.json '
                             'when resuming (0 = fresh)')
    parser.add_argument('--train_batch_tokens', type=int, default=20000)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_turn_tokens', type=int, default=8192)
    parser.add_argument('--norms', nargs='*', default=None,
                        help='default: TRAIN_NORMS of the active NORM_GROUP')
    parser.add_argument('--worlds', nargs='+', default=ROLLOUT_WORLDS)
    parser.add_argument('--max_steps', type=int, default=ROLLOUT_MAX_STEPS)
    parser.add_argument('--capacity_ladder', type=int, nargs='+', default=None,
                        help='difficulty levels (crowd capacity). Default: no '
                             'ladder — every norm trains at ITS benchmark '
                             'capacity from experiment_config.json '
                             '(norm_capacity override, else the global '
                             "'capacity'; user decision 2026-09-03). Pass "
                             'e.g. 4 6 8 to enable the curriculum instead')
    # Per-job scratch: two jobs sharing one rollouts dir rmtree each other's
    # episode frames mid-run — the launchers always pass a job-keyed path.
    parser.add_argument('--scratch',
                        default=os.path.join(ROLLOUT_SCRATCH_DIR, 'spiral'))
    for key, value in defaults.items():
        arg_type = type(value) if not isinstance(value, bool) else int
        parser.add_argument(f'--{key}', type=arg_type, default=value)
    return parser


# ---------------------------------------------------------------------------
# Episode bookkeeping
# ---------------------------------------------------------------------------

_EXP_CONFIG = None


def benchmark_capacity(norm):
    """The norm's crowd size in the benchmark: experiment_config.json's
    per-norm ``norm_capacity`` override, else its global ``capacity``."""
    global _EXP_CONFIG
    if _EXP_CONFIG is None:
        _EXP_CONFIG = load_experiment_config()
    return int(_EXP_CONFIG.get('norm_capacity', {}).get(norm)
               or _EXP_CONFIG['capacity'])


def turn_is_wait(turn):
    """Executed action of a turn: unparseable responses fall back to wait."""
    parsed = _parse_memory_response(turn.response)
    return parsed is None or parsed.get('action_text') == 'wait'


def episode_record(ep, capacity):
    return {
        'norm': ep.norm_type, 'world': ep.world_type, 'seed': ep.seed,
        'capacity': capacity,
        'return': round(ep.env_return, 2),
        'violations': ep.num_violations, 'collisions': ep.num_collisions,
        'goals': ep.goals_reached, 'success': bool(ep.task_success),
        'vf_success': bool(ep.task_success and ep.num_violations == 0),
        'turns': ep.turns_total, 'turns_parsed': ep.turns_parsed,
    }


def aggregate_episodes(records, prefix=''):
    n = max(len(records), 1)
    turns = sum(r['turns'] for r in records)
    out = {
        # 'n_episodes', not 'episodes': the un-prefixed stats dict stores the
        # per-episode records under 'episodes' (295207's log lost them to
        # this key collision).
        prefix + 'n_episodes': len(records),
        prefix + 'return': sum(r['return'] for r in records) / n,
        prefix + 'violations': sum(r['violations'] for r in records) / n,
        prefix + 'collisions': sum(r['collisions'] for r in records) / n,
        prefix + 'goals': sum(r['goals'] for r in records) / n,
        prefix + 'success': sum(r['success'] for r in records) / n,
        prefix + 'vf_success': sum(r['vf_success'] for r in records) / n,
        prefix + 'turn_parse': (sum(r['turns_parsed'] for r in records)
                                / max(1, turns)),
    }
    return out


def build_tasks(norms, controller, args, rng, scratch_tag):
    """episodes_per_update tasks, round-robin over the norms (shuffled so the
    remainder norms rotate), difficulty from the controller."""
    order = list(norms)
    rng.shuffle(order)
    tasks = []
    for i in range(args.episodes_per_update):
        norm = order[i % len(order)]
        overrides = controller.overrides(norm)
        tasks.append(dict(
            norm=norm, world=rng.choice(args.worlds),
            seed=rng.randint(ROLLOUT_SEED_LOW, ROLLOUT_SEED_HIGH),
            workdir=os.path.join(args.scratch, scratch_tag, f'ep{i}'),
            capacity=overrides.get('capacity', benchmark_capacity(norm)),
            max_steps=args.max_steps,
            gamma=1.0, format_bonus=0.0,
        ))
    return tasks


def fixed_eval_scenarios(norms, args):
    """Identical (norm, world, seed) list for every eval of the run."""
    rng = random.Random(args.seed + 777)
    return [(norms[i % len(norms)], rng.choice(args.worlds),
             rng.randint(ROLLOUT_SEED_LOW, ROLLOUT_SEED_HIGH))
            for i in range(args.eval_episodes)]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def run(args, objective):
    """objective protocol:
      objective.metric_key: str      — 'eval_'-prefixed stat for adapter_best
      objective.score(episodes, backend, args, rng, temperature)
          -> list aligned with episodes; each entry None (drop) or
             {'reward': float, 'success': float, 'extra_turn': Turn|None,
              'log': dict}
      objective.summarize(results) -> dict of aggregate stats (may be {})
    """
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

    if not args.out:
        sys.exit('[FATAL] --out is required')
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.scratch, exist_ok=True)
    metrics_path = os.path.join(args.out, 'training_log.jsonl')
    state_path = os.path.join(args.out, 'spiral_state.json')

    torch.manual_seed(args.seed)
    task_rng = random.Random(args.seed + 1234)
    shuffle_rng = random.Random(args.seed + 4321)
    objective_rng = random.Random(args.seed + 555)

    norms = args.norms or list(TRAIN_NORMS)
    # No --capacity_ladder: a single empty rung — the controller is inert and
    # build_tasks falls through to each norm's benchmark capacity.
    ladder = ([{'capacity': c} for c in args.capacity_ladder]
              if args.capacity_ladder else [{}])
    baselines = EmaBaselines(decay=args.baseline_decay)
    controller = DifficultyController(
        norms, ladder, lo=args.ctrl_lo, hi=args.ctrl_hi, decay=args.ctrl_decay,
        min_samples=args.ctrl_min_samples, start_level=args.ctrl_start_level)
    if args.start_update > 0 and args.resume_state:
        restored = load_state(state_path, baselines, controller)
        say(f'[state] restored spiral_state.json from update {restored}'
            if restored is not None else '[state] no spiral_state.json to restore')
    say(f'[spiral] norms={norms}')
    if args.capacity_ladder:
        say(f'[spiral] ladder={ladder} band=[{args.ctrl_lo},{args.ctrl_hi}] '
            f'baseline_decay={args.baseline_decay}')
    else:
        say(f'[spiral] no ladder — benchmark capacities: '
            f'{ {n: benchmark_capacity(n) for n in norms} } '
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

    backend, train_pool, eval_pool = None, None, None
    serving_dir = os.path.join(args.out, 'adapter_serving')
    if is_main:
        wait_for_server(args.rollout_url)
        model.save_pretrained(serving_dir)
        sync_lora(args.rollout_url, serving_dir, args.lora_name)
        backend = OpenAIBackend(args.rollout_url, args.rollout_model_name,
                                top_p=args.top_p)
        train_pool = RolloutPool(
            num_workers=args.rollout_workers or args.episodes_per_update,
            rollout_url=args.rollout_url, model_name=args.lora_name,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            top_p=args.top_p)
        if args.eval_every > 0:
            eval_pool = RolloutPool(
                num_workers=min(args.eval_episodes, 16),
                rollout_url=args.rollout_url, model_name=args.lora_name,
                temperature=args.eval_temperature,
                max_new_tokens=args.max_new_tokens, top_p=args.top_p)
    eval_scen = fixed_eval_scenarios(norms, args)

    optimizer = torch.optim.AdamW(policy_params, lr=args.lr)
    best_metric = float('-inf')
    collapse_streak = 0

    for update in range(args.start_update, args.num_updates):
        t_start = time.time()
        # Same task draw on every rank (task_rng in lockstep); rollouts +
        # scoring run on rank 0 only and are broadcast.
        tasks = build_tasks(norms, controller, args, task_rng, f'u{update:04d}')

        payloads, stats, episodes = None, None, []
        if is_main:
            t_gen = time.time()
            episodes = train_pool.run_batch(tasks)
            gen_seconds = round(time.time() - t_gen, 1)
            live = [(t, ep) for t, ep in zip(tasks, episodes) if ep is not None]
            results = objective.score([ep for _, ep in live], backend, args,
                                      objective_rng, args.temperature)
            payloads, ep_records, wait_turns, total_turns = [], [], 0, 0
            for (task, ep), res in zip(live, results):
                rec = episode_record(ep, task['capacity'])
                total_turns += len(ep.turns)
                wait_turns += sum(turn_is_wait(t) for t in ep.turns)
                if res is None:
                    rec['dropped'] = True
                    ep_records.append(rec)
                    continue
                controller.record(ep.norm_type, float(res['success']))
                advantage = baselines.advantage(ep.norm_type, float(res['reward']))
                rec.update({'reward': res['reward'],
                            'advantage': round(advantage, 4), **res.get('log', {})})
                ep_records.append(rec)
                turns = list(ep.turns)
                if args.max_turns_per_episode and len(turns) > args.max_turns_per_episode:
                    idx = sorted(objective_rng.sample(range(len(turns)),
                                                      args.max_turns_per_episode))
                    turns = [turns[i] for i in idx]
                if res.get('extra_turn') is not None:
                    turns.append(res['extra_turn'])
                payloads.extend((advantage, t) for t in turns)
            changes = controller.step()
            for line in changes:
                say(f'[curriculum u{update}] {line}')
            stats = {'update': update, 'gen_seconds': gen_seconds,
                     'episodes_failed': len(tasks) - len(live),
                     'dropped': sum(1 for r in results if r is None),
                     'mean_reward': (sum(r['reward'] for r in results if r)
                                     / max(1, sum(1 for r in results if r))),
                     'mean_advantage': (sum(p[0] for p in payloads)
                                        / max(1, len(payloads))),
                     'wait_frac': wait_turns / max(1, total_turns),
                     'baselines': {k: round(v, 3) for k, v in baselines.values.items()},
                     'levels': controller.levels(),
                     'curriculum_changes': changes,
                     'episodes': ep_records}
            stats.update(aggregate_episodes(ep_records))
            stats.update(objective.summarize([r for r in results if r]))
        if world > 1:
            holder = [payloads, stats]
            dist.broadcast_object_list(holder, src=0)
            payloads, stats = holder

        # ---------------- Encode + old per-token log-probs (sharded) -------
        proto = payloads or []
        model.eval()
        local = {}
        for g, (_, turn) in enumerate(proto):
            if g % world != rank:
                continue
            encoded = encode_full(processor, turn.messages(), turn.response)
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
        stats['num_train_turns'] = len(kept)
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
            say(f'[update {update}] no trainable turns')

        if world > 1:
            dist.barrier()
        if is_main:
            for ep in episodes:
                if ep is not None:
                    cleanup_episode(ep)

        # ---------------- Held-out eval ------------------------------------
        if is_main and eval_pool is not None and (
                (update + 1) % args.eval_every == 0 or update == args.start_update):
            t_ev = time.time()
            ev_tasks = [dict(norm=n, world=w, seed=s,
                             workdir=os.path.join(args.scratch, f'eval_u{update:04d}', f'ep{i}'),
                             capacity=args.eval_capacity or benchmark_capacity(n),
                             max_steps=args.max_steps,
                             gamma=1.0, format_bonus=0.0)
                        for i, (n, w, s) in enumerate(eval_scen)]
            ev_eps = [ep for ep in eval_pool.run_batch(ev_tasks) if ep is not None]
            ev_results = objective.score(ev_eps, backend, args, objective_rng,
                                         args.eval_temperature)
            ev_records = [episode_record(ep, args.eval_capacity
                                         or benchmark_capacity(ep.norm_type))
                          for ep in ev_eps]
            stats.update(aggregate_episodes(ev_records, prefix='eval_'))
            scored = [r for r in ev_results if r]
            if scored:
                stats['eval_reward'] = sum(r['reward'] for r in scored) / len(scored)
            stats.update({'eval_' + k: v for k, v in
                          objective.summarize(scored).items()})
            stats['eval_seconds'] = round(time.time() - t_ev, 1)
            for ep in ev_eps:
                cleanup_episode(ep)
            metric = stats.get(objective.metric_key)
            if metric is not None and metric > best_metric:
                best_metric = metric
                model.save_pretrained(os.path.join(args.out, 'adapter_best'))
                with open(os.path.join(args.out, 'best_metric.json'), 'w') as f:
                    json.dump({'update': update, 'metric': objective.metric_key,
                               **{k: v for k, v in stats.items()
                                  if k.startswith('eval_')}}, f, indent=2)
                say(f'[eval u{update}] {objective.metric_key}={metric:.3f} '
                    f'new best -> adapter_best')

        # ---------------- Log / save / guard -------------------------------
        stats['seconds'] = round(time.time() - t_start, 1)
        if is_main:
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(stats) + '\n')
            say(f'[update {update}] reward={stats["mean_reward"]:.3f} '
                f'adv={stats["mean_advantage"]:+.3f} '
                f'vf={stats.get("vf_success", 0.0):.2f} '
                f'succ={stats.get("success", 0.0):.2f} '
                f'viol={stats.get("violations", 0.0):.2f} '
                f'tparse={stats.get("turn_parse", 0.0):.3f} '
                f'wait={stats.get("wait_frac", 0.0):.2f} '
                f'levels={stats.get("levels")} '
                f'ent={stats.get("entropy", 0.0):.2f} '
                f'kl={stats.get("approx_kl", 0.0):.4f} '
                f'({stats["seconds"]}s, gen {stats.get("gen_seconds", 0)}s)')
            model.save_pretrained(os.path.join(args.out, 'adapter_latest'))
            save_state(state_path, baselines, controller, update)
            if args.save_every and (update + 1) % args.save_every == 0:
                model.save_pretrained(os.path.join(args.out, f'adapter_update_{update + 1:04d}'))

        # Collapse guard: parse slide (the vigal signature) or entropy
        # runaway (295207's signature — sustained negative advantages
        # spreading probability into noise).
        parse = stats.get('turn_parse')
        ent = stats.get('entropy')
        bad = ((parse is not None and parse < 0.5)
               or (args.ent_halt > 0 and ent is not None and ent > args.ent_halt))
        collapse_streak = collapse_streak + 1 if bad else 0
        if args.halt_on_collapse and collapse_streak >= args.halt_patience:
            say(f'[HALT] turn_parse < 0.5 or entropy > {args.ent_halt} for '
                f'{collapse_streak} consecutive updates — stopping at '
                f'u{update} (adapter_best is the checkpoint to keep)')
            break

    if is_main:
        if train_pool is not None:
            train_pool.close()
        if eval_pool is not None:
            eval_pool.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
