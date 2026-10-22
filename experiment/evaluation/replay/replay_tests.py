"""Replay the recorded test episodes and keep the world running to the end.

eval.py calls ``replay_one_run`` for every test run to compute path length,
community path length and community welfare; no separate step is needed for
the metrics. The command line below writes the replayed runs to disk for the
figures (plot_absolute_costs.py, plot_cost_externality.py), including the
best human participant's runs and the oracle re-run in each model's setting.
Both test splits are replayed: same-layout (``test_<k>``) and new-layout
(``new_map_test_<k>_<world>``).

**Why.** A test episode ends the moment the subject finishes its goals, so a
fast subject cuts the bystanders' episode short (the norm oracle finishes in
~39 steps, a stalled open-weight model in ~90). Any community metric that
compares bystander outcomes across subjects therefore compares different
horizons. This tool re-simulates every recorded test run *exactly* and then,
instead of stopping when the subject finishes, keeps stepping the scripted
background agents until the episode's own end (``max_steps``, 100), so the
community is observed over the same full horizon under every subject.

**How.** Every test run is a deterministic function of its seed: the pipeline
reseeds the global RNG with ``base_seed + run_index`` and draws the setting
(subject slot, arrival times, colours, doors, scripted customers) in a fixed
order, and every background agent is scripted. The only non-scripted part is
the subject's action at each step, and the run's memory log
(``simulation_log/test_<k>.json``) records the action the model *chose* at
every step it was on the map (auto-waits and parse fallbacks included). So
the replay rebuilds the run through the pipeline's own builders, swaps the
subject's ``get_action`` for a lookup into that log, and never calls an LLM.
The oracle baseline needs no log: its subject is scripted too.

**Verification.** Every replay is checked against the original raw trace
over the steps the original covered (position, action, rewards of every
agent). A run whose replay diverges before the subject finished is written
with ``verify.mismatches > 0`` and reported; the metrics built on the data
should exclude it. The subject leaves the map when done, as it did in the
original run, so what follows is the community without it.

**Output.** ``experiment/results_replay/<spec>/cafe/<norm>/raw_runs/<run_id>.json``,
mirroring the results layout (new-layout runs sit next to the same-layout
ones, as in the results tree): the subject's trace (same schema as
the pipeline's raw runs), every agent's final reward, the step each agent
finished at (``completions``), and every agent's cumulative reward at every
step (``rewards_by_step``), which is what the full-horizon community metrics
read.

Usage:
    python experiment/evaluation/replay/replay_tests.py --specs memory_llm/anthropic_claude-opus-5 oracle_with_norm
        [--norms ...] [--indices 1 2 3] [--splits test new_map_test]
        [--out experiment/results_replay] [--workers 8] [--stop-at-subject]
"""
import argparse
import contextlib
import io
import json
import os
import random
import shutil
import sys
import types
import concurrent.futures
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from experiment.run import common as pc  # noqa: E402
from src.env.actions import Action  # noqa: E402

# NORMGYM_RESULTS_ROOT points the replay at another results tree, e.g. when
# the tool runs from a worktree of an older simulator against today's logs.
RESULTS_ROOT = os.environ.get('NORMGYM_RESULTS_ROOT') or os.path.join(
    PROJECT_ROOT, 'experiment', 'results')
DEFAULT_OUT = os.path.join(PROJECT_ROOT, 'experiment', 'results_replay')
ORACLE_SPEC = 'oracle_with_norm'
# The scripted-subject baselines: a spec naming one of these is a results
# tree of its own (no AGENT/MODEL split) and needs no action log to replay.
ORACLE_SPECS = ('oracle_with_norm', 'oracle_without_norm')
WORLD = 'cafe'


# ----------------------------------------------------------------- inputs

def spec_base(spec, world=WORLD):
    """Results directory holding the norms of one subject tree."""
    if spec in ORACLE_SPECS:
        return os.path.join(RESULTS_ROOT, spec, world)
    agent, model = spec.split('/', 1)
    return os.path.join(RESULTS_ROOT, agent, model, world)


def spec_subject_kind(spec):
    return spec if spec in ORACLE_SPECS else spec.split('/', 1)[0]


def run_config(base, norm):
    """Seed and settings the original run used, from its results record."""
    log_dir = os.path.join(base, norm, 'simulation_log')
    for name in (f'experiment_results_{norm}.json', f'oracle_results_{norm}.json'):
        path = os.path.join(log_dir, name)
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                rec = json.load(f)
            stages = rec.get('stages', [])
            tests = [s for s in stages if s.get('stage') == 'test']
            new_map = [s for s in stages if s.get('stage') == 'new_map_test']
            return {
                'random_seed': int(rec['random_seed']),
                'capacity': int(rec['capacity']),
                'test_max_steps': int(rec['test_max_steps']),
                'world_type': rec.get('world_type', WORLD),
                'num_tests': len(tests) or 3,
                'tests': tests,
                'new_map_worlds': rec.get('new_map_worlds')
                                  or [s['world_type'] for s in new_map],
            }
    raise FileNotFoundError(f'No results record under {log_dir}')


def chosen_actions(base, norm, run_id):
    """{env step: Action} the model chose, from the run's memory log."""
    path = os.path.join(base, norm, 'simulation_log', f'{run_id}.json')
    with open(path, encoding='utf-8') as f:
        steps = json.load(f)['steps']
    out = {}
    for s in steps:
        name = str(s.get('action', 'wait')).strip().upper().replace(' ', '_')
        out[int(s['step'])] = Action[name] if name in Action.__members__ else Action.WAIT
    return out


def original_trace(base, norm, run_id):
    path = os.path.join(base, norm, 'raw_runs', f'{run_id}.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)['trace']


HUMAN_SPEC = 'human'
HUMAN_DATA = os.path.join(PROJECT_ROOT, 'experiment', 'human', 'data')
PIPELINE_CONFIG = os.path.join(PROJECT_ROOT, 'experiment', 'run',
                               'experiment_config.json')
# The study app logs the participant's key, not the pipeline's action name.
HUMAN_ACTIONS = {'left': Action.MOVE_LEFT, 'right': Action.MOVE_RIGHT,
                 'up': Action.MOVE_UP, 'down': Action.MOVE_DOWN,
                 'wait': Action.WAIT}


def pipeline_config():
    with open(PIPELINE_CONFIG, encoding='utf-8') as f:
        return json.load(f)


def best_participant_id(norm):
    """The participant with the highest violation-free success rate on this
    norm's test episodes (the rule every human figure uses)."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiment', 'plot'))
    from plot_ability_radar import best_participant, human_runs_by_participant
    return best_participant(human_runs_by_participant(norm))[0]['participant_id']


def human_norms():
    """Norms the study covers (one metrics summary per norm)."""
    root = os.path.join(PROJECT_ROOT, 'experiment', 'human', 'eval', 'results')
    return sorted(os.path.basename(p)[len('metrics_summary_'):-len('.json')]
                  for p in glob.glob(os.path.join(root, 'metrics_summary_*.json')))


def human_test_run(norm, world=WORLD):
    """({meta}, {step: Action}, [logged step events]) for the best
    participant's same-layout test episode on one norm.

    The study app (experiment/human/app.py) replays the pipeline's draw
    sequence with the pipeline's own per-norm seeds, so a human round can be
    rebuilt exactly like a model run; the log carries the seed, the entry
    slot and the run index it drew.
    """
    pid = best_participant_id(norm)
    paths = [p for p in sorted(glob.glob(os.path.join(
        HUMAN_DATA, pid, f'{pid}_b*_{norm}_test_ep*.jsonl')))
        if 'new_map' not in os.path.basename(p)]
    if not paths:
        raise FileNotFoundError(f'no same-layout test episode for {pid} on {norm}')
    actions, steps, meta = {}, [], {'participant_id': pid, 'path': paths[-1]}
    with open(paths[-1], encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            event = record.get('event')
            if event == 'round_start':
                meta.update(seed=int(record['random_seed']),
                            positions=set(record['human_positions']),
                            run_id=record['pipeline_run_id'])
            elif event == 'session_start':
                meta['config'] = record.get('config') or {}
            elif event == 'action':
                name = str(record['action']).strip().lower()
                actions[int(record['step'])] = HUMAN_ACTIONS.get(name, Action.WAIT)
            elif event == 'step':
                steps.append(record)
    meta['index'] = int(meta['run_id'].split('_')[1])
    config = meta.get('config') or {}
    norm_capacity = config.get('norm_capacity') or pipeline_config().get('norm_capacity', {})
    meta['capacity'] = (config.get('capacity')
                        or norm_capacity.get(norm)
                        or pipeline_config().get('capacity', 8))
    meta['max_steps'] = config.get('test_max_steps') or pipeline_config().get('test_max_steps', 100)
    meta['num_tests'] = config.get('num_tests') or pipeline_config().get('num_tests', 3)
    meta['base_seed'] = pipeline_config().get('norm_seeds', {}).get(norm, meta['seed'] - meta['index'])
    meta['world_type'] = world
    return meta, actions, steps


def compare_human_trace(logged, replayed):
    """{'compared_steps', 'mismatches', 'first_mismatch'} over the logged step
    events: the participant's position and running reward each step.

    The study app logs every env step, including the ones before the
    participant's character arrives on the map, which the run recorder does
    not cover; those are skipped rather than counted as mismatches.
    """
    by_step = {r['step']: r for r in replayed}
    if not by_step:
        return {'compared_steps': 0, 'mismatches': len(logged),
                'first_mismatch': {'what': 'replay recorded no subject step'}}
    arrival = min(by_step)
    logged = [r for r in logged if int(r['step']) >= arrival]
    mismatches, first = 0, None
    for record in logged:
        rep = by_step.get(int(record['step']))
        bad = None
        if rep is None:
            bad = 'missing step'
        elif list(record['human_position']) != list(rep['position'] or []):
            bad = (f"position: logged {record['human_position']} vs replay "
                   f"{rep['position']}")
        elif round(float(record['cumulative_reward']), 2) != round(float(rep['cumulative_reward']), 2):
            bad = (f"cumulative_reward: logged {record['cumulative_reward']} vs "
                   f"replay {rep['cumulative_reward']}")
        if bad:
            mismatches += 1
            if first is None:
                first = {'step': record['step'], 'what': bad}
    return {'compared_steps': len(logged), 'mismatches': mismatches,
            'first_mismatch': first}


class _NoClient:
    """Stands in for the LLM client: the replay never prompts."""

    def response(self, *args, **kwargs):
        raise RuntimeError('replay agent must not call the LLM')


# ----------------------------------------------------------------- replay

def _replay_get_action(actions):
    def get_action(self, world, all_current_positions, output_dir, *, actual_action=None):
        return actions.get(int(self.current_step), Action.WAIT)
    return get_action


_PLACEHOLDER_PNG = None


def _stub_drawing(env):
    """Swap the env's frame drawing for a 1x1 placeholder write and drop the
    video assembly, so render() keeps its side effects at no drawing cost."""
    global _PLACEHOLDER_PNG
    if _PLACEHOLDER_PNG is None:
        import numpy as np
        import matplotlib.image as mpimg
        buf = io.BytesIO()
        mpimg.imsave(buf, np.zeros((1, 1, 3)), format='png')
        _PLACEHOLDER_PNG = buf.getvalue()

    def visualize(save_path=None, *args, **kwargs):
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(_PLACEHOLDER_PNG)
    env._visualize_step = visualize
    env._save_video = lambda *a, **k: None


def replay_one_run(base, norm, subject_kind, split, index, frames_dir,
                   replay_actions=True, check=True, stop_at_subject=False,
                   draw=False):
    """Rebuild one recorded test run under ``base`` (a results directory
    holding <norm>/raw_runs and <norm>/simulation_log) and replay it to the
    end of the episode. Returns (replayed raw run, home world).

    ``split`` is ``test`` (same layout) or ``new_map_test`` (the index-th
    new-layout world), drawn exactly as run_test_series draws it. With
    ``replay_actions`` the subject repeats the actions in the run's memory
    log; otherwise it plays itself (a scripted oracle). With ``check`` the
    replay is compared against the original trace (``verify`` in the result).
    """
    cfg = run_config(base, norm)
    capacity, max_steps = cfg['capacity'], cfg['test_max_steps']
    home_world = cfg['world_type']
    if split == 'test':
        world_type, count, seed_offset = home_world, cfg['num_tests'], 0
        run_id = f'test_{index:03d}'
    elif split == 'new_map_test':
        worlds = cfg['new_map_worlds']
        if not 1 <= index <= len(worlds):
            raise ValueError(f'{norm}: no new-layout test {index} (worlds: {worlds})')
        world_type, count, seed_offset = worlds[index - 1], len(worlds), cfg['num_tests']
        run_id = f'new_map_test_{index:03d}_{world_type}'
    else:
        raise ValueError(f'Unknown split: {split}')

    # Same draws, same order as run_test_series -> run_test_stage.
    subject_classes = pc.subject_class_schedule(cfg['random_seed'], count)
    seed = cfg['random_seed'] + seed_offset + index
    random.seed(seed)
    subject_positions = pc._random_subject_positions(capacity, norm)
    background_slots = capacity - len(subject_positions)
    env = pc.make_env(norm, None, capacity, world_type, max_steps)
    client = None if subject_kind in ORACLE_SPECS else _NoClient()
    initial_agents = pc.build_initial_agents(
        capacity, subject_positions, env.world, norm, client,
        ground_truth='without', subject_kind=subject_kind, prompt_stage='test',
        reference_memory_path=None, memory_filename=None, num_no_norm_agents=0,
        subject_class=subject_classes[index - 1], test_index=index)
    waiting_norm, waiting_no_norm, _ = pc.build_scripted_pool(
        capacity * 20, background_slots, 0, env.world, norm, client,
        ground_truth='without', first_agent_number=capacity + 1)

    subject_ids = {a.agent_id for a in initial_agents if pc.is_subject_agent(a)}
    if replay_actions and subject_kind not in ORACLE_SPECS:
        actions = chosen_actions(base, norm, run_id)
        for a in initial_agents:
            if a.agent_id in subject_ids:
                a.get_action = types.MethodType(_replay_get_action(actions), a)

    recorder, completions, rewards_by_step, subject_finish_step = simulate(
        env, initial_agents, waiting_norm, waiting_no_norm, norm, subject_ids,
        frames_dir, stop_at_subject, draw)
    shutil.rmtree(env.output_dir, ignore_errors=True)

    verify = (compare_traces(original_trace(base, norm, run_id), recorder.trace)
              if check else None)
    raw = recorder.raw_run(run_id=run_id, stage=f'{split}_replay',
                           subject_kind=subject_kind, norm_type=norm,
                           world_type=world_type, capacity=capacity,
                           max_steps=max_steps, random_seed=seed)
    raw.update({
        'subject_finish_step': subject_finish_step,
        'completions': completions,
        'rewards_by_step': rewards_by_step,
        'verify': verify,
    })
    return raw, home_world


def replay_run(spec, norm, index, out_root, stop_at_subject=False, draw=False,
               oracle_like=None, split='test'):
    """Replay test run ``index`` of ``spec``/``norm`` (see replay_one_run) and
    write it under ``out_root``; returns its verification.

    With ``oracle_like`` set to a model tree spec, the run is instead the
    norm oracle playing the subject in *that tree's* setting (its seed,
    capacity and run count): the matched baseline for a model run whose
    settings differ from the recorded oracle tree's. Nothing to verify
    against, so ``verify`` is None; the output goes under
    ``oracle_like/<tree>``."""
    if spec == HUMAN_SPEC:
        return replay_human_run(norm, out_root, stop_at_subject, draw)
    out_spec = os.path.join('oracle_like', oracle_like) if oracle_like else spec
    subject_kind = ORACLE_SPEC if oracle_like else spec_subject_kind(spec)
    frames_dir = os.path.join(out_root, '_frames', out_spec.replace('/', '_'),
                              norm, f'{split}_{index:03d}')
    raw, home_world = replay_one_run(
        spec_base(oracle_like or spec), norm, subject_kind, split, index,
        frames_dir, replay_actions=not oracle_like, check=not oracle_like,
        stop_at_subject=stop_at_subject, draw=draw)
    raw['spec'] = out_spec
    out_dir = os.path.join(out_root, out_spec, home_world, norm, 'raw_runs')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{raw['run_id']}.json"), 'w', encoding='utf-8') as f:
        json.dump(raw, f)
    return raw['verify']


def simulate(env, initial_agents, waiting_norm, waiting_no_norm, norm,
             subject_ids, frames_dir, stop_at_subject=False, draw=False):
    """Run one rebuilt episode to its end (or to the subject's finish).

    Returns (recorder, {agent: finish step}, [{step, rewards}], subject finish
    step). The pipeline's render() carries state side effects beyond drawing,
    so it is still called every step; unless ``draw``, the frame drawing is
    replaced by a placeholder write.
    """
    env.reset(options={'agents': initial_agents})
    env.output_dir = frames_dir
    os.makedirs(env.output_dir, exist_ok=True)
    if not draw:
        _stub_drawing(env)
    recorder = pc.RunRecorder(env)
    done_ids, completions, rewards_by_step = set(), {}, []
    subject_finish_step = None
    next_replacement_step, last_spawn_entrance = 1, None
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step()
        step = info['step']
        recorder.record_step(info)
        env.render()
        pc.release_region_holders(env, subject_ids, norm)
        newly_done = [a for a in env.agents if a.done and a.agent_id not in done_ids]
        for agent in newly_done:
            done_ids.add(agent.agent_id)
            completions[agent.agent_id] = step
            if agent.agent_id in subject_ids:
                continue
            new_agent = pc.pop_scripted_replacement(agent, waiting_norm, waiting_no_norm)
            if new_agent is None:
                continue
            next_replacement_step, last_spawn_entrance = pc.respawn_agent(
                env, new_agent, next_replacement_step, last_spawn_entrance,
                keep_out_of_region=pc.region_staged(norm))
        rewards_by_step.append({
            'step': step,
            'rewards': {a.agent_id: round(float(a.total_reward), 3)
                        for a in env.agents if pc._entered_map(a)}})
        if subject_finish_step is None and subject_ids <= done_ids:
            subject_finish_step = step
            if stop_at_subject:
                break
    return recorder, completions, rewards_by_step, subject_finish_step


def replay_human_run(norm, out_root, stop_at_subject=False, draw=False):
    """Rebuild and extend the best participant's same-layout test episode.

    The study app draws its setting in the pipeline's order from the same
    per-norm seed, so the rebuild differs from a model run only in the
    subject: a HumanAgent whose actions come from the log's ``action``
    events. Steps the interface never prompted (the participant was settled
    at a target) have no logged action and wait, as the app's agent does.
    """
    meta, actions, logged = human_test_run(norm)
    capacity, max_steps = meta['capacity'], meta['max_steps']
    world_type, run_id = meta['world_type'], meta['run_id']

    random.seed(meta['seed'])
    subject_positions = pc._random_subject_positions(capacity, norm)
    if subject_positions != meta['positions']:
        print(f"[WARN] {norm}: entry slot {sorted(subject_positions)} rebuilt vs "
              f"{sorted(meta['positions'])} logged; using the logged slot")
        subject_positions = meta['positions']
    background_slots = capacity - len(subject_positions)
    env = pc.make_env(norm, None, capacity, world_type, max_steps)
    subject_class = pc.subject_class_schedule(
        meta['base_seed'], meta['num_tests'])[meta['index'] - 1]
    initial_agents = pc.build_initial_agents(
        capacity, subject_positions, env.world, norm, None,
        ground_truth='without', subject_kind='human', prompt_stage='test',
        num_no_norm_agents=0, subject_class=subject_class)
    waiting_norm, waiting_no_norm, _ = pc.build_scripted_pool(
        capacity * 20, background_slots, 0, env.world, norm, None,
        ground_truth='without', first_agent_number=capacity + 1)

    subject_ids = {a.agent_id for a in initial_agents if pc.is_subject_agent(a)}
    for agent in initial_agents:
        if agent.agent_id in subject_ids:
            agent.get_action = types.MethodType(_replay_get_action(actions), agent)

    recorder, completions, rewards_by_step, subject_finish_step = simulate(
        env, initial_agents, waiting_norm, waiting_no_norm, norm, subject_ids,
        os.path.join(out_root, '_frames', HUMAN_SPEC, norm, run_id),
        stop_at_subject, draw)

    verify = compare_human_trace(logged, recorder.trace)
    raw = recorder.raw_run(run_id=run_id, stage='test_replay', spec=HUMAN_SPEC,
                           subject_kind='human', norm_type=norm,
                           world_type=world_type, capacity=capacity,
                           max_steps=max_steps, random_seed=meta['seed'])
    raw.update({
        'participant_id': meta['participant_id'],
        'subject_finish_step': subject_finish_step,
        'completions': completions,
        'rewards_by_step': rewards_by_step,
        'verify': verify,
    })
    shutil.rmtree(os.path.join(out_root, '_frames', HUMAN_SPEC, norm, run_id),
                  ignore_errors=True)
    out_dir = os.path.join(out_root, HUMAN_SPEC, world_type, norm, 'raw_runs')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{run_id}.json'), 'w', encoding='utf-8') as f:
        json.dump(raw, f)
    return verify


_COMPARE = ('position', 'action', 'step_reward', 'cumulative_reward',
            'other_agent_cumulative_reward')


def compare_traces(original, replayed):
    """{'compared_steps', 'mismatches', 'first_mismatch'} over the original's
    steps; a replay that stops short of the original counts the missing
    steps as mismatches."""
    by_step = {(r['step'], r['agent_id']): r for r in replayed}
    mismatches, first = 0, None
    for r in original:
        rep = by_step.get((r['step'], r['agent_id']))
        bad = None
        if rep is None:
            bad = 'missing step'
        else:
            for k in _COMPARE:
                if rep.get(k) != r.get(k):
                    bad = f'{k}: original {r.get(k)!r} vs replay {rep.get(k)!r}'
                    break
        if bad:
            mismatches += 1
            if first is None:
                first = {'step': r['step'], 'agent_id': r['agent_id'], 'what': bad}
    return {'compared_steps': len(original), 'mismatches': mismatches,
            'first_mismatch': first}


# -------------------------------------------------------------------- CLI

def _job(args):
    spec, norm, index, out_root, stop, draw, oracle_like, split = args
    label = f'{split}_{index:03d}'
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            verify = replay_run(spec, norm, index, out_root, stop, draw,
                                oracle_like, split)
        except Exception as exc:  # report, keep the batch going
            return spec, norm, label, {'error': f'{type(exc).__name__}: {exc}'}
    return spec, norm, label, verify or {'compared_steps': 0, 'mismatches': 0,
                                         'first_mismatch': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--specs', nargs='+', required=True,
                        help='AGENT/MODEL trees, oracle_with_norm, '
                             'oracle_without_norm, or human (the best '
                             "participant's test episode per norm)")
    parser.add_argument('--norms', nargs='+', default=None,
                        help='Norms to replay (default: every norm of the tree)')
    parser.add_argument('--indices', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--splits', nargs='+', default=['test', 'new_map_test'],
                        choices=['test', 'new_map_test'],
                        help='Test splits to replay (default: both); human '
                             'covers the same-layout split only')
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--stop-at-subject', action='store_true',
                        help='Stop when the subject finishes (verification '
                             'only; no extension)')
    parser.add_argument('--draw', action='store_true',
                        help='Really draw every frame as the pipeline did '
                             '(slow; for diagnosing a divergence)')
    parser.add_argument('--oracle-like', action='store_true',
                        help='For each AGENT/MODEL spec, generate the norm '
                             'oracle playing the subject in that tree\'s own '
                             'setting (written under oracle_like/<spec>)')
    args = parser.parse_args()

    jobs = []
    for spec in args.specs:
        if spec == HUMAN_SPEC:
            # One same-layout test episode per norm: the best participant's.
            for n in (args.norms or human_norms()):
                jobs.append((spec, n, 1, args.out, args.stop_at_subject,
                             args.draw, None, 'test'))
            continue
        base = spec_base(spec)
        norms = args.norms or sorted(
            n for n in os.listdir(base)
            if os.path.isfile(os.path.join(base, n, 'raw_runs', 'test_001.json')))
        like = spec if args.oracle_like else None
        jobs += [(spec, n, i, args.out, args.stop_at_subject, args.draw, like,
                  split)
                 for split in args.splits for n in norms for i in args.indices]
    print(f'{len(jobs)} runs -> {args.out}', flush=True)

    ok = bad = err = 0
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for spec, norm, label, verify in pool.map(_job, jobs):
            if 'error' in verify:
                err += 1
                print(f'[ERROR] {spec} {norm} {label}: {verify["error"]}', flush=True)
            elif verify['mismatches']:
                bad += 1
                print(f'[DIVERGED] {spec} {norm} {label}: '
                      f'{verify["mismatches"]}/{verify["compared_steps"]} steps, '
                      f'first {verify["first_mismatch"]}', flush=True)
            else:
                ok += 1
    print(f'done: {ok} verified, {bad} diverged, {err} errors', flush=True)


if __name__ == '__main__':
    main()
