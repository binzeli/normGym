"""Metrics for the human study (``experiment/human/data/``).

The human counterpart of ``experiment/evaluation/eval.py``: same metric
definitions, same aggregate labels, so a human summary can be put side by side
with an LLM summary. The only input is the web-app log of a run — one
``*.jsonl`` file of JSON events written by ``experiment/human/app.py`` — so
metrics can be recomputed without re-running the study.

Per run (see ``run_metrics``):
    violation_free_success  success * (1 - norm_violation): success that also
                            kept the norm (0 if the run violated it at all)
    success               mean over the participant's characters of that
                          character's task success (1.0 left through the door,
                          0.5 reached the second goal, 0.0 neither)
    norm_violation        1 if the participant violated the norm at least once
    num_norm_violations   count of participant violations
    collision             1 if the participant collided with anyone
    num_collisions        count of participant collisions
    episode_length        steps the participant spent on the map
    reward                the participant's total reward, summed over their
                          characters (the interactive stage hands the
                          participant a fresh character after each one leaves,
                          which resets the logged ``cumulative_reward``)

Per norm (see ``aggregate_runs``): the mean of each of the above over that
norm's test runs — ViolationFreeSuccessRate, SuccessRate, ViolationRate,
ViolationsPerEpisode, CollisionRate, CollisionsPerEpisode, AverageLength,
AverageReward. The interactive-stage run is reported separately, as in the
pipeline.

Two pipeline metrics have no human counterpart and are omitted: AverageAllReward
and AverageRewardGap need every agent's reward, and the web app only logs the
participant's own.

With ``--judge``, the participant's final inferred-rule guess (the
``*_guess_*.json`` written at the end of each block) is graded against the
ground-truth norm description by the same LLM judge the pipeline uses for an
LLM's norm belief (see ``experiment/utils/norm_judge.py``) on a 7-point Likert
scale — NormBeliefLikert in 1..7 (7 = the guess states the same rule as the
ground truth) and NormBeliefScore = (likert − 1) / 6 in [0, 1].
``--judge-mid`` additionally grades the mid-interactive guesses, giving a
belief trajectory over steps.

Usage:
    python experiment/human/eval/eval.py                     # all participants
    python experiment/human/eval/eval.py --participants p1 p2
    python experiment/human/eval/eval.py --judge             # + NormBeliefScore
    python experiment/human/eval/eval.py --out-dir some/dir
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_HERE), 'data')
RESULTS_DIR = os.path.join(_HERE, 'results')

_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# (per-run key, aggregate label) for every metric aggregate_runs averages.
# Mirrors _MEAN_KEYS in experiment/evaluation/eval.py, minus the multi-agent
# metrics the human logs do not carry (all_reward / reward_gap).
_MEAN_KEYS = [
    ('violation_free_success', 'ViolationFreeSuccessRate'),
    ('success', 'SuccessRate'),
    ('norm_violation', 'ViolationRate'),
    ('num_norm_violations', 'ViolationsPerEpisode'),
    ('collision', 'CollisionRate'),
    ('num_collisions', 'CollisionsPerEpisode'),
    ('episode_length', 'AverageLength'),
    ('reward', 'AverageReward'),
]

_AGGREGATE_LABELS = [label for _, label in _MEAN_KEYS]

# The scored test stages, as in the pipeline: each norm block ends with one
# same-world test and one new-map test. 'interactive' (the learning stage) is
# reported on its own.
TEST_STAGES = ('test', 'new_map_test')


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def read_events(jsonl_path: str) -> List[Dict[str, Any]]:
    """The JSON events of one run log, in order."""
    events = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _character_traces(step_events: Sequence[Dict[str, Any]]
                      ) -> List[List[Dict[str, Any]]]:
    """Split the step trace by participant character.

    The interactive stage replaces the participant's character after it leaves
    through the door, so one run can hold several of them — the analogue of the
    pipeline's several subject instances. A character's last step is the one
    where it has no target left (it reached the exit); whatever follows belongs
    to its replacement.
    """
    traces: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for step in step_events:
        current.append(step)
        if not step.get('human_target'):
            traces.append(current)
            current = []
    if current:
        traces.append(current)
    return traces


def _character_success(trace: Sequence[Dict[str, Any]]) -> float:
    """1.0 the character left through the door, 0.5 it reached the second goal,
    0.0 neither.

    Goal 1 (order at the counter) -> Goal 2 (leave through the door) -> done.
    A finished character's last step carries no target; a character still on the
    map when the run ended is scored by the goal it had reached.
    """
    if not trace:
        return 0.0
    if not trace[-1].get('human_target'):
        return 1.0
    reached_goal2 = any(
        (step.get('human_goal') or '').startswith('Goal 2') for step in trace
    )
    return 0.5 if reached_goal2 else 0.0


def run_metrics(jsonl_path: str) -> Dict[str, Any]:
    """Derive the per-run metrics from one human run log."""
    events = read_events(jsonl_path)
    by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_event[event.get('event')].append(event)

    session = (by_event.get('session_start') or [{}])[0]
    game_over = (by_event.get('game_over') or [{}])[-1]
    step_events = by_event.get('step', [])
    traces = _character_traces(step_events)

    # The web app logs the participant's own violation / collision totals
    # (background characters are counted separately), so they can be read off
    # the final game_over event.
    num_violations = int(game_over.get('norm_violations',
                                       sum(1 for s in step_events
                                           if s.get('human_violated'))))
    num_collisions = int(game_over.get('total_collisions', 0))
    episode_length = int(game_over.get('total_steps', len(step_events)))

    # cumulative_reward restarts with every new character, so the participant's
    # total over the run is the sum of their step rewards.
    subject_rewards = [
        round(sum(float(s.get('step_reward', 0.0)) for s in trace), 3)
        for trace in traces
    ]
    success = _mean([_character_success(trace) for trace in traces])
    norm_violation = 1 if num_violations > 0 else 0

    return {
        'run_id': os.path.basename(jsonl_path),
        'participant_id': session.get('participant_id'),
        # Rounds logged before the new-map test existed carry only 'phase'
        # (always 'test' for a test round) and the study-wide world in config.
        'stage': session.get('stage') or session.get('phase'),
        'norm_type': session.get('norm_type'),
        'world_type': (session.get('world_type')
                       or (session.get('config') or {}).get('world_type')),
        'block': session.get('block'),
        'episode': session.get('episode'),
        # Which of the pipeline's runs of the stage this participant drew, named
        # as the pipeline names it — the join key to the llm/rl run of the same
        # setting. Absent in logs written before the draw existed.
        'stage_run': session.get('stage_run'),
        'pipeline_run_id': session.get('pipeline_run_id'),
        # Success that also kept the norm: zeroed if the run violated it at all.
        'violation_free_success': round(success * (1 - norm_violation), 3),
        'success': success,
        'norm_violation': norm_violation,
        'num_norm_violations': num_violations,
        'collision': 1 if num_collisions > 0 else 0,
        'num_collisions': num_collisions,
        'episode_length': episode_length,
        'reward': round(sum(subject_rewards), 3),
        'subject_rewards': subject_rewards,
    }


# ---------------------------------------------------------------------------
# Aggregation over the runs of one norm
# ---------------------------------------------------------------------------

def aggregate_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Average the per-run metrics of a set of runs."""
    summary: Dict[str, Any] = {'num_runs': len(runs)}
    for key, label in _MEAN_KEYS:
        summary[label] = _mean([float(r[key]) for r in runs if key in r])
    return summary


def build_metrics_summary(norm_type: str,
                          runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Metrics summary for one norm, over every participant who saw it."""
    interactive = [r for r in runs if r['stage'] == 'interactive']
    # The norm's own world is the one it was learned in; the new-map test runs
    # somewhere else, so it must not decide the summary's world.
    first = (interactive or runs or [{}])[0]
    new_map_worlds = sorted({r.get('world_type') for r in runs
                             if r['stage'] == 'new_map_test' and r.get('world_type')})

    by_participant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_participant[run['participant_id']].append(run)

    summary: Dict[str, Any] = {
        'norm_type': norm_type,
        'agent_type': 'human',
        'world_type': first.get('world_type'),
        'new_map_worlds': new_map_worlds,
        'participants': sorted(by_participant),
        'interactive': {
            'runs': interactive,
            'summary': aggregate_runs(interactive),
        },
    }
    for stage in TEST_STAGES:
        stage_runs = [r for r in runs if r['stage'] == stage]
        summary[stage] = {
            'runs': stage_runs,
            'summary': aggregate_runs(stage_runs),
        }

    # Same breakdown per participant, so single-subject variation stays visible.
    summary['per_participant'] = {
        pid: {
            stage: aggregate_runs([r for r in p_runs if r['stage'] == stage])
            for stage in ('interactive',) + TEST_STAGES
        }
        for pid, p_runs in sorted(by_participant.items())
    }
    return summary


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_participants(data_dir: str = DATA_DIR) -> List[str]:
    """Every participant folder in the data directory."""
    return sorted(
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith('_')
    )


def load_runs(participants: Sequence[str],
              data_dir: str = DATA_DIR) -> List[Dict[str, Any]]:
    """Per-run metrics for every interactive / test run of the participants.

    Practice and demo runs are not scored: they use the tutorial norm.
    """
    runs = []
    for participant in participants:
        pattern = os.path.join(data_dir, participant, '*.jsonl')
        for path in sorted(glob.glob(pattern)):
            metrics = run_metrics(path)
            if metrics['stage'] in ('interactive',) + TEST_STAGES and metrics['norm_type']:
                runs.append(metrics)
    if not runs:
        raise ValueError(f"No interactive/test run logs found in: {data_dir}")
    return runs


# ---------------------------------------------------------------------------
# Norm-belief judge (the human counterpart of the LLM's norm belief)
# ---------------------------------------------------------------------------

def load_guesses(participant: str, norm_type: str, kind: str,
                 data_dir: str = DATA_DIR) -> List[Dict[str, Any]]:
    """The participant's inferred-rule guesses for one norm.

    ``kind`` is 'final' (the guess at the end of the block) or 'mid' (the
    guesses prompted during the interactive stage), sorted by step.
    """
    guesses = []
    pattern = os.path.join(data_dir, participant, '*_guess_*.json')
    if kind == 'mid':
        pattern = os.path.join(data_dir, participant, '*_midguess_*.json')
    for path in sorted(glob.glob(pattern)):
        with open(path, 'r', encoding='utf-8') as f:
            guess = json.load(f)
        if guess.get('norm_type') == norm_type and guess.get('kind') == kind:
            guesses.append(guess)
    return sorted(guesses, key=lambda g: (g.get('step') or 0, g.get('timestamp', '')))


def judge_norm_guesses(summary: Dict[str, Any], judge_mid: bool = False,
                       data_dir: str = DATA_DIR) -> None:
    """Grade each participant's inferred-rule guess for this norm with the LLM
    judge and record it under ``summary['norm_belief_judge']``.

    The final guess is the human counterpart of the LLM's interactive-stage norm
    belief, so NormBeliefScore is comparable across subject kinds.
    """
    try:
        from experiment.utils.norm_judge import (
            ground_truth_norm_description, judge_norm_belief)
        ground_truth = ground_truth_norm_description(summary['norm_type'])
    except Exception as exc:
        print(f"[WARN] Norm-belief judge unavailable; skipping: {exc}")
        return

    per_participant: Dict[str, Any] = {}
    for participant in summary['participants']:
        finals = load_guesses(participant, summary['norm_type'], 'final', data_dir)
        if not finals:
            continue
        try:
            verdict = judge_norm_belief(finals[-1].get('norm_guess'), ground_truth)
        except Exception as exc:
            print(f"[WARN] Judge failed for {participant}: {exc}")
            continue
        entry: Dict[str, Any] = {
            'NormBeliefLikert': verdict['likert'],
            'NormBeliefScore': verdict['score'],
            'belief': verdict['belief'],
            'reasoning': verdict['reasoning'],
            'judge_model': verdict['judge_model'],
            'confidence': finals[-1].get('confidence'),
        }
        if judge_mid:
            trajectory = []
            for guess in load_guesses(participant, summary['norm_type'], 'mid', data_dir):
                try:
                    mid = judge_norm_belief(guess.get('norm_guess'), ground_truth)
                except Exception as exc:
                    print(f"[WARN] Judge failed for {participant} mid-guess: {exc}")
                    continue
                trajectory.append({
                    'step': guess.get('step'),
                    'NormBeliefLikert': mid['likert'],
                    'NormBeliefScore': mid['score'],
                    'belief': mid['belief'],
                    'confidence': guess.get('confidence'),
                })
            entry['trajectory'] = trajectory
        per_participant[participant] = entry

    if not per_participant:
        return
    summary['norm_belief_judge'] = {
        'NormBeliefLikert': _mean([e['NormBeliefLikert']
                                   for e in per_participant.values()]),
        'NormBeliefScore': _mean([e['NormBeliefScore']
                                  for e in per_participant.values()]),
        'ground_truth': ground_truth,
        'per_participant': per_participant,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(f"METRICS SUMMARY — norm={summary.get('norm_type')} "
          f"agent=human world={summary.get('world_type')}")
    print(f"participants: {', '.join(summary.get('participants', [])) or '-'}")
    print("=" * 78)

    for stage, title in (('interactive', 'Interactive stage'),
                         ('test', 'Same-world tests'),
                         ('new_map_test',
                          'New-map tests' + (f" ({', '.join(summary['new_map_worlds'])})"
                                             if summary.get('new_map_worlds') else ''))):
        stats = summary.get(stage, {}).get('summary', {})
        print(f"\n{title} (n={stats.get('num_runs', 0)}):")
        for label in _AGGREGATE_LABELS:
            print(f"  {label + ':':<26}{stats.get(label)}")

    per_participant = summary.get('per_participant', {})
    if per_participant:
        for stage in TEST_STAGES:
            print(f"\nPer participant ({stage} runs):")
            print(f"  {'participant':<26}{'n':>3}  " +
                  " ".join(f"{label[:11]:>11}" for label in _AGGREGATE_LABELS))
            for pid, stages in per_participant.items():
                stats = stages.get(stage, {})
                row = " ".join(f"{stats.get(l)!s:>11}" for l in _AGGREGATE_LABELS)
                print(f"  {pid:<26}{stats.get('num_runs'):>3}  {row}")

    judge = summary.get('norm_belief_judge')
    if judge:
        print(f"\nNorm-belief judge:")
        print(f"  NormBeliefLikert (1-7):   {judge.get('NormBeliefLikert')}")
        print(f"  NormBeliefScore:          {judge.get('NormBeliefScore')}")
        for pid, entry in judge.get('per_participant', {}).items():
            print(f"    {pid:<24}{entry['NormBeliefScore']}  "
                  f"(confidence {entry.get('confidence')})")
    print("\n" + "=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description='Compute the pipeline eval metrics for the human study runs, '
                    'one summary per norm.')
    parser.add_argument('--participants', nargs='*', default=None,
                        help='Participant folder names (default: every folder '
                             'in experiment/human/data/)')
    parser.add_argument('--data-dir', default=DATA_DIR,
                        help='Where the participant folders live '
                             '(default: experiment/human/data)')
    parser.add_argument('--out-dir', default=RESULTS_DIR,
                        help='Where to write metrics_summary_<norm>.json '
                             '(default: experiment/human/eval/results)')
    parser.add_argument('--judge', action='store_true',
                        help="Grade each participant's final rule guess with the "
                             'LLM judge (NormBeliefScore); needs API access')
    parser.add_argument('--judge-mid', action='store_true',
                        help='With --judge, also grade the mid-interactive '
                             'guesses to get a belief trajectory')
    args = parser.parse_args()

    participants = args.participants or find_participants(args.data_dir)
    runs = load_runs(participants, args.data_dir)

    by_norm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_norm[run['norm_type']].append(run)

    os.makedirs(args.out_dir, exist_ok=True)
    for norm_type in sorted(by_norm):
        summary = build_metrics_summary(norm_type, by_norm[norm_type])
        if args.judge:
            judge_norm_guesses(summary, args.judge_mid, args.data_dir)
        print_metrics_summary(summary)

        out_path = os.path.join(args.out_dir, f'metrics_summary_{norm_type}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Metrics summary saved to: {out_path}")


if __name__ == '__main__':
    main()
