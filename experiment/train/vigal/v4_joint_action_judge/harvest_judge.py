"""v3 harvest: oracle-only episodes -> (pre-move frame, move, env verdict) records.

Plays the pipeline's interactive stage with NO model in the loop: the subject
is an ``oracle_without_norm`` planner and one more background agent is
norm-blind (``--num_no_norm_agents 1`` -> 2 norm-blind agents among
``--capacity 10``), everyone else follows the norm. Every step, every agent's
move is logged by the env together with its verdict, so one episode yields
~100 labelled questions for free.

Timing (verified on a rendered frame, 2026-08-27): ``step_{s}.png`` is
rendered after ``env.step()`` at step s but BEFORE step s's moves are applied
(moves execute at the start of the next step), so it shows every agent at the
``current_position`` of step s's actions and marks the violations injected
into step s's data (= moves made at step s-1). A move made at step s is
charged in ``episode_log[s+1]['norm_violations']`` as (agent_id, from, to).
So the record for the move at step s uses frame ``step_{s}.png`` (positions
match the prompt text; only earlier moves' feedback visible) and the label
from ``episode_log[s+1]``.

Questions are asked about every move of the norm-blind agents and about a
``--compliant_frac`` share of the norm-following agents' moves (so that
"which agents are blind" is not itself the cue). Auto-wait steps where the
agent is done/inactive are skipped.

Output: ``<out>/records.jsonl`` + ``<out>/images/``. Independent of v1/v2.
"""
import argparse
import contextlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from experiment.train.config import (  # version-neutral layer only
    DATA_DIR, GROUP_DATA_DIR, NORM_CATEGORY, ROLLOUT_SEED_HIGH,
    ROLLOUT_SEED_LOW, TRAIN_NORMS, benchmark_capacity, default_env_kwargs,
)
from experiment.train.vigal.v4_joint_action_judge.judge_task import CATEGORY_HINTS, action_text, build_system_prompt

JUDGE_DATA_DIR = os.path.join(GROUP_DATA_DIR, 'violation_judge')   # data/<group>/violation_judge/


def _pos(p):
    return [int(p[0]), int(p[1])] if p is not None else None


def run_oracle_episode(norm, world_type, seed, workdir, capacity=10, max_steps=120,
                       num_no_norm_agents=1, quiet=True):
    """run_experiment.run_interactive_stage without the subject-frame filter
    and raw-run writing; returns (episode dict, figures_dir, names_by_step,
    system_prompt_fields)."""
    from experiment.run.common import (
        _random_subject_positions, build_initial_agents, build_scripted_pool,
        is_subject_agent, make_env, pop_scripted_replacement, respawn_agent,
        subject_class_schedule, summarize_episode,
    )
    from experiment.run.run_experiment import RunRecorder  # noqa: F401  (import check)

    os.makedirs(workdir, exist_ok=True)
    figures_dir = os.path.join(workdir, 'frames')
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
    os.makedirs(figures_dir)
    random.seed(seed)
    subject_positions = _random_subject_positions(capacity, norm)
    num_subjects = len(subject_positions)
    background_slots = capacity - num_subjects
    pool_size = capacity * 20
    subject_pool_size = pool_size * num_subjects // capacity
    customer_pool_size = pool_size - subject_pool_size

    log = open(os.path.join(workdir, 'sim_log.txt'), 'w', encoding='utf-8')
    with (contextlib.redirect_stdout(log) if quiet else contextlib.nullcontext()):
        env = make_env(norm, figures_dir, capacity, world_type, max_steps, **default_env_kwargs())
        subject_classes = subject_class_schedule(seed, 1 + subject_pool_size)
        initial_agents = build_initial_agents(
            capacity, subject_positions, env.world, norm, None, 'without',
            subject_kind='oracle_without_norm', prompt_stage='interactive',
            max_steps=max_steps, memory_filename=None,
            num_no_norm_agents=num_no_norm_agents, subject_class=subject_classes[0])
        from experiment.run.common import make_agent
        agent_id_counter = capacity + 1
        waiting_subject = deque()
        for pool_index in range(subject_pool_size):
            waiting_subject.append(make_agent(
                f'C{agent_id_counter}', env.world, norm, None,
                subject_kind='oracle_without_norm', ground_truth='without',
                prompt_stage='interactive', max_steps=max_steps, memory_filename=None,
                subject_class=subject_classes[pool_index + 1]))
            agent_id_counter += 1
        waiting_norm, waiting_no_norm, agent_id_counter = build_scripted_pool(
            customer_pool_size, background_slots, num_no_norm_agents,
            env.world, norm, None, 'without', first_agent_number=agent_id_counter)

        # --- the interactive loop (run_interactive_simulation) + per-step name/norm snapshots
        obs, info = env.reset(options={'agents': initial_agents})
        env.output_dir = figures_dir
        env.render()
        names_by_step = {0: {a.agent_id: a.name for a in env.agents if a.active and not a.done}}
        blind_ids = {a.agent_id for a in env.agents if a.norm is None}
        done_ids = set()
        terminated = truncated = False
        next_replacement_step, last_spawn_entrance = 1, None
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step()
            env.render()
            names_by_step[env.current_step] = {a.agent_id: a.name for a in env.agents
                                               if a.active and not a.done}
            blind_ids |= {a.agent_id for a in env.agents if a.norm is None}
            newly_done = [a for a in env.agents if a.done and a.agent_id not in done_ids]
            for agent in newly_done:
                done_ids.add(agent.agent_id)
                if is_subject_agent(agent):
                    new_agent = waiting_subject.popleft() if waiting_subject else None
                else:
                    new_agent = pop_scripted_replacement(agent, waiting_norm, waiting_no_norm)
                if new_agent is None:
                    continue
                next_replacement_step, last_spawn_entrance = respawn_agent(
                    env, new_agent, next_replacement_step, last_spawn_entrance)
        episode = summarize_episode(env, done_ids, env.current_step, figures_dir)
        world = env.world
        fields = dict(
            world_description=f"You are in a {getattr(world, 'WORLD_NAME', 'a cafe').replace(' LAYOUT', '').lower()}.",
            object_positions=world.build_object_positions_text(),
            counter=str(tuple(_pos(getattr(world, 'counter_position', None) or
                                   next((a.target_positions[0] for a in env.agents
                                         if getattr(a, 'target_positions', None)), (2, 6))))))
    log.close()
    episode['blind_ids'] = sorted(blind_ids)
    return episode, figures_dir, names_by_step, fields


def extract_records(episode, figures_dir, names_by_step, fields, norm, world_type, seed,
                    compliant_frac, history_len, rng):
    log = episode['episode_log']
    by_step = {sd['step']: sd for sd in log}
    blind = set(episode['blind_ids'])
    system_prompt = build_system_prompt(**fields)
    feedback = []            # (step, agent_id) of violations shown so far (cross at frame `step`)
    history = {}             # agent_id -> deque of (step, action_text, from, to)
    maxlen = history_len if history_len > 0 else None   # 0 = full history
    memory_lines = []        # per lived step: EVERY agent's position — the observer
                             # analog of v1's full-memory section (user request 2026-09-06)
    records = []
    for s in sorted(by_step):
        sd = by_step[s]
        # crosses drawn on frame s = violations injected into step s's data
        for v in sd.get('norm_violations', []):
            feedback.append((s, v[0]))
        nxt = by_step.get(s + 1)
        frame = os.path.join(figures_dir, f'step_{s:03d}.png')
        names = names_by_step.get(s, {})
        positions = {aid: _pos(a['current_position']) for aid, a in sd.get('actions', {}).items()
                     if a.get('current_position') is not None and aid in names}
        pos_text = ', '.join(f'{names[i]} at ({p[0]}, {p[1]})'
                             for i, p in sorted(positions.items(), key=lambda kv: names[kv[0]]))
        if nxt is None or not os.path.exists(frame) or s < 1:
            for aid, a in sd.get('actions', {}).items():
                history.setdefault(aid, deque(maxlen=maxlen)).append(
                    (s, action_text(a['action']), _pos(a['current_position']), _pos(a['next_position'])))
            if pos_text:
                memory_lines.append(f'- Step {s}: {pos_text}.')
            continue
        charged = {(v[0], tuple(_pos(v[1])) if v[1] is not None else None,
                    tuple(_pos(v[2])) if v[2] is not None else None)
                   for v in nxt.get('norm_violations', [])}
        charged_ids = {c[0] for c in charged}
        mem = memory_lines if history_len <= 0 else memory_lines[-history_len:]
        memory_text = '\n'.join(mem)
        for aid, a in sd.get('actions', {}).items():
            state = sd.get('agent_states', {}).get(aid, {})
            if not state.get('active', True) or state.get('done') or aid not in names:
                continue
            fr, to = _pos(a['current_position']), _pos(a['next_position'])
            if fr is None or to is None:
                continue
            is_blind = aid in blind
            if not is_blind and rng.random() > compliant_frac:
                pass_record = False
            else:
                pass_record = True
            label = int(aid in charged_ids)
            if pass_record:
                hist = list(history.get(aid, []))
                if history_len > 0:
                    hist = hist[-history_len:]
                hist_text = '; '.join(f'step {hs}: {ht}' + (f' {hf}->{ht2}' if ht != 'wait' else '')
                                      for hs, ht, hf, ht2 in hist) or ''
                fb = feedback if history_len <= 0 else feedback[-8:]
                fb_text = ', '.join(f'{names_by_step.get(fs, {}).get(fid, fid)} at step {fs}'
                                    for fs, fid in fb)
                records.append({
                    'id': f'{norm}_{world_type}_{seed}_step{s:03d}_{aid}',
                    'norm': norm, 'world': world_type, 'seed': seed, 'step': s,
                    'agent_id': aid, 'agent_name': names[aid], 'agent_is_blind': is_blind,
                    'agent_class': state.get('agent_class'),
                    'from_pos': fr, 'to_pos': to, 'action_text': action_text(a['action']),
                    'label': label,
                    'positions_text': pos_text, 'history_text': hist_text, 'feedback_text': fb_text,
                    'memory_text': memory_text,
                    'system_prompt': system_prompt,
                    'image_src': frame,
                })
            history.setdefault(aid, deque(maxlen=maxlen)).append(
                (s, action_text(a['action']), fr, to))
        if pos_text:
            memory_lines.append(f'- Step {s}: {pos_text}.')
    return records


def _run_task(task):
    try:
        rng = random.Random(task['seed'] + 17)
        ep, figs, names, fields = run_oracle_episode(
            task['norm'], task['world'], task['seed'], task['workdir'],
            capacity=task['capacity'], max_steps=task['max_steps'],
            num_no_norm_agents=task['num_no_norm_agents'])
        fields['category'] = task.get('category', 'queue')
        recs = extract_records(ep, figs, names, fields, task['norm'], task['world'], task['seed'],
                               task['compliant_frac'], task['history_len'], rng)
        os.makedirs(task['image_dir'], exist_ok=True)
        for r in recs:
            rel = os.path.join('images', os.path.basename(r['id']) + '.png')
            dst = os.path.join(task['out'], rel)
            if not os.path.exists(dst):
                shutil.copyfile(r['image_src'], dst)
            r['image'] = rel
            del r['image_src']
        summary = dict(norm=task['norm'], seed=task['seed'], steps=ep['steps'],
                       records=len(recs), positives=sum(r['label'] for r in recs),
                       blind_records=sum(r['agent_is_blind'] for r in recs),
                       blind_ids=ep['blind_ids'])
        if not task.get('keep_frames'):
            shutil.rmtree(task['workdir'], ignore_errors=True)
        return {'records': recs, 'summary': summary}
    except Exception:
        sys.stderr.write(f"[harvest-judge] episode failed ({task['norm']} seed={task['seed']}):\n"
                         f"{traceback.format_exc()}\n")
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=os.path.join(JUDGE_DATA_DIR, 'train'))
    p.add_argument('--scratch', default=os.path.join(DATA_DIR, '..', 'rollouts', 'judge_harvest'))
    p.add_argument('--norms', nargs='+', default=TRAIN_NORMS)
    p.add_argument('--worlds', nargs='+', default=['cafe'])
    p.add_argument('--num_episodes', type=int, default=176)
    p.add_argument('--capacity', type=int, default=0,
                   help="crowd size; 0 = each norm's benchmark capacity "
                        "from experiment_config.json")
    p.add_argument('--num_no_norm_agents', type=int, default=1,
                   help='norm-blind background agents (+ the norm-blind subject = 2)')
    p.add_argument('--max_steps', type=int, default=120)
    p.add_argument('--compliant_frac', type=float, default=0.25,
                   help='share of norm-following agents\' moves also asked about')
    p.add_argument('--history_len', type=int, default=0,
                   help='moves of the queried agent shown in the prompt; '
                        '0 = its FULL move history + ALL red-cross feedback '
                        '(the full-memory default, 2026-09-05; was 5/8)')
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--keep_frames', action='store_true')
    p.add_argument('--category', default=NORM_CATEGORY, choices=sorted(CATEGORY_HINTS),
                   help='norm category: picks the category-specific wording of the '
                        'observer system prompt; defaults to the active NORM_GROUP\'s '
                        'tag from config.py:NORM_GROUPS (the agents\' goal is NOT '
                        'always the counter — non-queue groups get their own hints)')
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.join(args.out, 'images'), exist_ok=True)
    tasks = []
    for i in range(args.num_episodes):
        norm = args.norms[i % len(args.norms)]
        seed = rng.randint(ROLLOUT_SEED_LOW, ROLLOUT_SEED_HIGH)
        tasks.append(dict(norm=norm, world=rng.choice(args.worlds), seed=seed,
                          workdir=os.path.join(args.scratch, f'{norm}_{seed}'),
                          capacity=args.capacity or benchmark_capacity(norm),
                          max_steps=args.max_steps,
                          num_no_norm_agents=args.num_no_norm_agents,
                          compliant_frac=args.compliant_frac, history_len=args.history_len,
                          category=args.category,
                          out=args.out, image_dir=os.path.join(args.out, 'images'),
                          keep_frames=args.keep_frames))
    t0 = time.time()
    total = pos = failed = 0
    per_norm = {}
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp.get_context('spawn')) as ex, \
            open(os.path.join(args.out, 'records.jsonl'), 'w', encoding='utf-8') as f:
        for res in ex.map(_run_task, tasks, chunksize=1):
            if res is None:
                failed += 1
                continue
            for r in res['records']:
                f.write(json.dumps(r) + '\n')
            s = res['summary']
            total += s['records']; pos += s['positives']
            pn = per_norm.setdefault(s['norm'], [0, 0]); pn[0] += s['records']; pn[1] += s['positives']
            print(f"[harvest] {s['norm']} seed={s['seed']} steps={s['steps']} records={s['records']} "
                  f"positives={s['positives']} blind={s['blind_records']} (total {total}, {int(time.time()-t0)}s)",
                  flush=True)
    print(f'[harvest] done: {total} records ({pos} violations = {pos / max(total, 1):.2f}) from '
          f'{len(tasks) - failed} episodes ({failed} failed) in {int(time.time() - t0)}s -> {args.out}')
    print('[harvest] per norm (records, positives): ' + ', '.join(f'{k}: {v[0]}/{v[1]}' for k, v in sorted(per_norm.items())))


if __name__ == '__main__':
    main()
