"""Harvest single-step training snapshots for ViGaL-style RLOO.

ViGaL synthesises tens of thousands of standalone game states and trains on
them one decision at a time. normGym's states can't be sampled procedurally —
the subject's prompt carries its own memory (past reasoning, belief, the
penalties it saw) — so snapshots are harvested from real episodes: a policy
plays the interactive stage exactly as in ``rollout.run_episode``, and every
time the subject is prompted we record

    (system_prompt, user_prompt, step image)      -> the snapshot
    label_state(subject, world, positions)         -> the verifier's labels
    (the harvest policy's own response + grade)    -> for reference only

The harvest policy only decides WHICH states get visited; the labels come
from the rule-based verifier, not from the policy. Use the SFT adapter
(the state distribution the RL learner starts from) and re-harvest with a
later adapter if the learner's states drift.

Auto-wait steps (the pipeline never prompts the model while it sits on its
goal) and collision-forced steps (the executed move is not the model's) are
skipped — there is no decision to grade.

Output layout (``--out``):
    snapshots.jsonl     one record per prompted step
    images/<ep>_<step>.png
    episodes.jsonl      per-episode summary (harvest policy's env return etc.)

Usage (with a vLLM server up, as in run_step_rloo_vllm.sh):
    python experiment/train/vigal/v2_belief_rloo/harvest_snapshots.py \
        --rollout_url http://localhost:PORT/v1 --model_name policy \
        --num_episodes 120 --num_workers 32 --out experiment/train/data/snapshots
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from experiment.train.config import (
    DATA_DIR,
    ROLLOUT_MAX_STEPS,
    ROLLOUT_SCRATCH_DIR,
    ROLLOUT_SEED_HIGH,
    ROLLOUT_SEED_LOW,
    ROLLOUT_WORLDS,
    TRAIN_NORMS,
    benchmark_capacity,
)

SNAPSHOT_DIR = os.path.join(DATA_DIR, 'snapshots')

# Worker-process state (set by _init_worker; never touched in the parent).
_CLIENT = None
_CAPTURED = []


def _install_hook():
    """Wrap LLMAgent.get_action so every real prompt of the subject is
    labelled by the verifier at the moment the model is asked."""
    from experiment.agents.llm_agent import LLMAgent
    from experiment.train.vigal.v2_belief_rloo.snapshot_verifier import label_state

    original = LLMAgent.get_action

    def capturing_get_action(self, world, all_current_positions, output_dir,
                             *, actual_action=None):
        label = None
        if actual_action is None and not self.done:
            label = label_state(self, world, all_current_positions)
        turns_before = len(self.client.turns)
        action = original(self, world, all_current_positions, output_dir,
                          actual_action=actual_action)
        # Only a real prompt appends a turn; auto-wait doesn't call the model.
        if label is not None and len(self.client.turns) > turns_before:
            turn = self.client.turns[-1]
            _CAPTURED.append({
                'agent_id': self.agent_id,
                'agent_class': self.agent_class,
                'step': self.current_step,
                'system_prompt': turn.system_prompt,
                'user_prompt': turn.user_prompt,
                'image_path': turn.image_paths[0] if turn.image_paths else None,
                'response': turn.response,
                'label': label,
            })
        return action

    LLMAgent.get_action = capturing_get_action


def _init_worker(rollout_url, model_name, temperature, max_new_tokens, top_p):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception:
        pass
    from experiment.train.policy_client import OpenAIBackend, RecordingClient
    global _CLIENT
    _CLIENT = RecordingClient(OpenAIBackend(rollout_url, model_name, top_p=top_p),
                              temperature=temperature,
                              max_new_tokens=max_new_tokens)
    _install_hook()


def _run_task(task):
    from experiment.train.rollout import cleanup_episode, run_episode
    from experiment.train.vigal.v2_belief_rloo.snapshot_verifier import grade_response
    _CAPTURED.clear()
    try:
        ep = run_episode(_CLIENT, norm_type=task['norm'], world_type=task['world'],
                         seed=task['seed'], workdir=task['workdir'],
                         capacity=task['capacity'], max_steps=task['max_steps'],
                         subject_kind=task.get('subject_kind', 'memory_llm'))
    except Exception:
        sys.stderr.write(f"[harvest-worker] episode failed "
                         f"({task['norm']} seed={task['seed']}):\n"
                         f"{traceback.format_exc()}\n")
        return None

    ep_key = f"{task['norm']}_{task['world']}_{task['seed']}"
    os.makedirs(task['image_dir'], exist_ok=True)
    records = []
    for snap in _CAPTURED:
        image_rel = None
        if snap['image_path'] and os.path.exists(snap['image_path']):
            image_rel = os.path.join('images', f"{ep_key}_step{snap['step']:03d}.png")
            shutil.copyfile(snap['image_path'],
                            os.path.join(task['out'], image_rel))
        grade = grade_response(snap['label'], snap['response'])
        records.append({
            'id': f"{ep_key}_step{snap['step']:03d}",
            'norm': task['norm'], 'world': task['world'], 'seed': task['seed'],
            'step': snap['step'], 'agent_id': snap['agent_id'],
            'agent_class': snap['agent_class'],
            'system_prompt': snap['system_prompt'],
            'user_prompt': snap['user_prompt'],
            'image': image_rel,
            'label': snap['label'],
            'harvest_response': snap['response'],
            'harvest_grade': grade,
        })
    summary = {
        'norm': ep.norm_type, 'world': ep.world_type, 'seed': ep.seed,
        'env_return': round(ep.env_return, 2), 'violations': ep.num_violations,
        'collisions': ep.num_collisions, 'goals': ep.goals_reached,
        'success': ep.task_success, 'turns': ep.turns_total,
        'snapshots': len(records),
    }
    cleanup_episode(ep)
    return {'records': records, 'summary': summary}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout_url', required=True)
    parser.add_argument('--model_name', default='policy',
                        help='served model / LoRA name on the vLLM server')
    parser.add_argument('--out', default=SNAPSHOT_DIR)
    parser.add_argument('--scratch', default=os.path.join(ROLLOUT_SCRATCH_DIR, 'harvest'))
    parser.add_argument('--norms', nargs='+', default=TRAIN_NORMS)
    parser.add_argument('--worlds', nargs='+', default=ROLLOUT_WORLDS)
    parser.add_argument('--num_episodes', type=int, default=120)
    parser.add_argument('--capacity', type=int, default=0,
                        help="crowd size; 0 = each norm's benchmark capacity "
                             "from experiment_config.json")
    parser.add_argument('--subject_kind', default='memory_llm',
                        choices=['memory_llm', 'full_memory_llm'],
                        help='agent whose prompts become the snapshots; '
                             'full_memory_llm = full episode history, no '
                             'belief line (RL-training prompts)')
    parser.add_argument('--max_steps', type=int, default=ROLLOUT_MAX_STEPS)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--append', action='store_true',
                        help='add to an existing snapshot set instead of refusing')
    return parser.parse_args()


def main():
    args = parse_args()
    import multiprocessing as mp

    snap_path = os.path.join(args.out, 'snapshots.jsonl')
    if os.path.exists(snap_path) and not args.append:
        sys.exit(f'[FATAL] {snap_path} exists; pass --append or choose another --out')
    os.makedirs(os.path.join(args.out, 'images'), exist_ok=True)
    os.makedirs(args.scratch, exist_ok=True)

    from experiment.train.rollout_pool import wait_for_server
    wait_for_server(args.rollout_url)

    rng = random.Random(args.seed)
    tasks = []
    for i in range(args.num_episodes):
        norm = args.norms[i % len(args.norms)]       # balanced over norms
        tasks.append(dict(
            norm=norm, world=rng.choice(args.worlds),
            seed=rng.randint(ROLLOUT_SEED_LOW, ROLLOUT_SEED_HIGH),
            workdir=os.path.join(args.scratch, f'ep{i:04d}'),
            capacity=args.capacity or benchmark_capacity(norm),
            max_steps=args.max_steps, subject_kind=args.subject_kind,
            out=args.out, image_dir=os.path.join(args.out, 'images')))

    t0 = time.time()
    ctx = mp.get_context('spawn')
    total, failed = 0, 0
    cats = {}
    with ctx.Pool(processes=args.num_workers, initializer=_init_worker,
                  initargs=(args.rollout_url, args.model_name, args.temperature,
                            args.max_new_tokens, args.top_p)) as pool, \
            open(snap_path, 'a', encoding='utf-8') as snap_f, \
            open(os.path.join(args.out, 'episodes.jsonl'), 'a', encoding='utf-8') as ep_f:
        for result in pool.imap_unordered(_run_task, tasks, chunksize=1):
            if result is None:
                failed += 1
                continue
            for rec in result['records']:
                snap_f.write(json.dumps(rec) + '\n')
                cats[rec['harvest_grade']['category']] = \
                    cats.get(rec['harvest_grade']['category'], 0) + 1
            ep_f.write(json.dumps(result['summary']) + '\n')
            snap_f.flush(); ep_f.flush()
            total += len(result['records'])
            s = result['summary']
            print(f"[harvest] {s['norm']} seed={s['seed']} snapshots={s['snapshots']} "
                  f"return={s['env_return']} viol={s['violations']} "
                  f"success={s['success']} (total {total}, {time.time() - t0:.0f}s)",
                  flush=True)

    n = max(total, 1)
    print(f'[harvest] done: {total} snapshots from {args.num_episodes - failed} '
          f'episodes ({failed} failed) in {time.time() - t0:.0f}s -> {snap_path}')
    print('[harvest] harvest policy vs verifier: ' +
          ' '.join(f'{k}={v / n:.2f}' for k, v in sorted(cats.items())))
    with open(os.path.join(args.out, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'snapshots': total, 'failed_episodes': failed,
                   'harvest_categories': cats}, f, indent=2)


if __name__ == '__main__':
    main()
