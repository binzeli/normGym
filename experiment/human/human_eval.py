"""Metrics for the first human pilot (``experiment/human/old_results/``).

Mirrors the metric definitions in ``experiment/evaluation/eval.py`` but reads the
human web-app logs instead of the pipeline's raw run logs. Each human *test*
run is one ``*_test_ep*.jsonl`` file (one JSON event per line) plus its
``*_summary.json``. Only test runs are scored here (practice / interactive /
guess / quiz / survey stages are ignored).

Per run (see ``run_metrics``):
    success               1.0 both goals reached, 0.5 one of two, 0.0 none
    norm_violation        1 if the participant violated the norm at least once
    num_norm_violations   count of participant violations
    collision             1 if the participant collided with any scripted agent
    num_collisions        count of participant-vs-scripted collisions
    episode_length        steps the participant spent on the map
    reward                participant's total reward

Per norm (see ``aggregate_runs``): the mean of each of the above over all test
runs of that norm — SuccessRate, ViolationRate, ViolationsPerEpisode,
CollisionRate, CollisionsPerEpisode, AverageLength, AverageReward.

(The pipeline's ``all_reward`` / ``reward_gap`` are multi-subject metrics; the
human is the only subject, so they are omitted here.)

HUMAN-ONLY counting — important
    These result files (2026-07-07) were written *before* commit c5ef99b
    ("Only track human-specific collisions and norm violations in the human
    study"). In those logs the cumulative ``norm_violations`` and
    ``total_collisions`` fields count *every* agent in the environment, not the
    participant, so they must NOT be used as the participant's counts.

    We instead recover the participant's own counts, which the logs do carry
    per step:
      * violations — the per-step ``human_violated`` boolean (attributed to the
        participant when it was logged).
      * collisions — the participant's own ``step_reward``. Rewards are
        per-agent, so the participant is only ever docked for their own
        collisions (COLLISION_PENALTY=5) and violations (NORM_VIOLATION=20) on
        top of the STEP_COST=0.1. A step that carries a −5 chunk with no
        violation is one participant collision.

Usage:
    python experiment/human/human_eval.py                 # default participants
    python experiment/human/human_eval.py Austen Chris     # explicit set
    python experiment/human/human_eval.py --out summary.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'old_results')

DEFAULT_PARTICIPANTS = [
    'Austen', 'Chris', 'Gillian', 'Maria', 'MatthewRenze', 'Navya', 'rebekah',
]

NORMS = ['occupation_seat_no_bottom_left', 'movement_clockwise_move', 'spacing_no_adjacent']

# Per-agent reward constants from src/env/rewards.py (used to recover the
# participant's own collision count from their own step_reward).
STEP_COST = 0.1
COLLISION_PENALTY = 5.0
NORM_VIOLATION_PENALTY = 20.0

_MEAN_KEYS = [
    ('success', 'SuccessRate'),
    ('norm_violation', 'ViolationRate'),
    ('num_norm_violations', 'ViolationsPerEpisode'),
    ('collision', 'CollisionRate'),
    ('num_collisions', 'CollisionsPerEpisode'),
    ('episode_length', 'AverageLength'),
    ('reward', 'AverageReward'),
]
# Extra aggregates derived from the means above (not a direct per-run mean).
# ComplianceRate is the fraction of runs with no violation = 1 - ViolationRate.
_DERIVED_LABELS = ['ComplianceRate']
_AGGREGATE_LABELS = (
    ['SuccessRate', 'ViolationRate', 'ComplianceRate']
    + [label for _, label in _MEAN_KEYS
       if label not in ('SuccessRate', 'ViolationRate')]
)


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def _read_events(jsonl_path: str) -> List[Dict[str, Any]]:
    events = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _success(step_events: List[Dict[str, Any]]) -> float:
    """1.0 both goals reached, 0.5 reached the second goal phase, 0.0 none.

    Goal 1 -> Goal 2 (leave through the door) -> finished. When the participant
    reaches the exit the final step carries ``human_target == null``; reaching
    the second goal phase shows up as a ``Goal 2`` goal string.
    """
    if not step_events:
        return 0.0
    reached_exit = step_events[-1].get('human_target') is None
    if reached_exit:
        return 1.0
    reached_goal2 = any(
        (e.get('human_goal') or '').startswith('Goal 2') for e in step_events
    )
    return 0.5 if reached_goal2 else 0.0


def _step_collisions(step: Dict[str, Any]) -> int:
    """Participant collisions on this step, recovered from their own step_reward.

    reward = GOAL*g - STEP_COST*acted - COLLISION_PENALTY*c - NORM_VIOLATION*v.
    A violation step is flagged by ``human_violated``; strip it, and any −5 chunk
    left in the penalty (beyond the step cost) is a participant collision. Goal
    steps have a positive reward and never overlap a collision in this data.
    """
    r = round(float(step.get('step_reward', 0.0)), 3)
    if step.get('human_violated') or r >= 0:
        return 0
    penalty = -r - STEP_COST  # penalty beyond the per-step action cost
    if penalty <= 0:
        return 0
    c = round(penalty / COLLISION_PENALTY)
    if abs(penalty - c * COLLISION_PENALTY) > 1e-6:
        print(f"[WARN] unrecognized step_reward {r} — collision count may be off")
    return c


def run_metrics(jsonl_path: str) -> Dict[str, Any]:
    """Derive the per-run metrics from one human test-run jsonl log."""
    events = _read_events(jsonl_path)
    by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_event[e.get('event')].append(e)

    session = (by_event.get('session_start') or [{}])[0]
    game_over = (by_event.get('game_over') or [{}])[-1]
    step_events = by_event.get('step', [])

    # Participant-only counts (see the module docstring): the logged cumulative
    # norm_violations / total_collisions are env-wide in these pre-c5ef99b files.
    num_violations = sum(1 for e in step_events if e.get('human_violated'))
    num_collisions = sum(_step_collisions(e) for e in step_events)
    episode_length = int(game_over.get('total_steps', len(step_events)))
    reward = float(game_over.get('final_reward', 0.0))

    return {
        'run_id': os.path.basename(jsonl_path),
        'participant_id': session.get('participant_id'),
        'norm_type': session.get('norm_type'),
        'block': session.get('block'),
        'episode': session.get('episode'),
        'success': _success(step_events),
        'norm_violation': 1 if num_violations > 0 else 0,
        'num_norm_violations': num_violations,
        'collision': 1 if num_collisions > 0 else 0,
        'num_collisions': num_collisions,
        'episode_length': episode_length,
        'reward': round(reward, 3),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Average the per-run metrics over a set of test runs."""
    summary: Dict[str, Any] = {'num_runs': len(runs)}
    for key, label in _MEAN_KEYS:
        summary[label] = _mean([float(r[key]) for r in runs if key in r])
    summary['ComplianceRate'] = round(1.0 - summary['ViolationRate'], 3)
    return summary


def find_test_runs(participants: Sequence[str]) -> List[str]:
    """All ``*_test_ep*.jsonl`` files (not the ``_summary.json``) for the given
    participants, sorted."""
    files: List[str] = []
    for p in participants:
        pattern = os.path.join(RESULTS_DIR, p, '*_test_ep*.jsonl')
        files.extend(sorted(glob.glob(pattern)))
    return files


def _excluded(run: Dict[str, Any], excludes: Sequence[tuple]) -> bool:
    """True if this run's (participant, norm) matches an exclusion pair. The norm
    side matches by substring, so ``movement`` covers ``movement_clockwise_move``."""
    for pid, norm_frag in excludes:
        if run['participant_id'] == pid and norm_frag in (run['norm_type'] or ''):
            return True
    return False


def build_summary(participants: Sequence[str],
                  excludes: Sequence[tuple] = ()) -> Dict[str, Any]:
    runs = [run_metrics(f) for f in find_test_runs(participants)]
    runs = [r for r in runs if not _excluded(r, excludes)]

    by_norm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_norm_participant: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by_norm[r['norm_type']].append(r)
        by_norm_participant[r['norm_type']][r['participant_id']].append(r)

    per_norm: Dict[str, Any] = {}
    for norm in NORMS:
        norm_runs = by_norm.get(norm, [])
        per_participant = {
            pid: {'runs': p_runs, 'summary': aggregate_runs(p_runs)}
            for pid, p_runs in sorted(by_norm_participant.get(norm, {}).items())
        }
        per_norm[norm] = {
            'summary': aggregate_runs(norm_runs),
            'per_participant': per_participant,
            'runs': norm_runs,
        }

    return {'participants': list(participants), 'per_norm': per_norm}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(summary: Dict[str, Any]) -> None:
    for norm in NORMS:
        block = summary['per_norm'].get(norm, {})
        stats = block.get('summary', {})
        print("\n" + "=" * 78)
        print(f"NORM: {norm}   (n={stats.get('num_runs', 0)} test runs)")
        print("=" * 78)
        for label in _AGGREGATE_LABELS:
            print(f"  {label + ':':<24}{stats.get(label)}")

        per_participant = block.get('per_participant', {})
        if per_participant:
            print(f"\n  {'participant':<16}{'n':>3}  " +
                  "  ".join(f"{l:>12}" for l in _AGGREGATE_LABELS))
            for pid, pdata in per_participant.items():
                s = pdata['summary']
                row = "  ".join(f"{s.get(l):>12}" for l in _AGGREGATE_LABELS)
                print(f"  {pid:<16}{s.get('num_runs'):>3}  {row}")
    print("\n" + "=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description='Compute eval.py-style metrics for the human test runs.')
    parser.add_argument('participants', nargs='*', default=DEFAULT_PARTICIPANTS,
                        help='Participant folder names (default: the study set)')
    parser.add_argument('--out', type=str, default=None,
                        help='Where to write the summary JSON '
                             '(default: human/old_results/human_metrics_summary.json)')
    parser.add_argument('--exclude', nargs='*', default=[], metavar='PID:NORM',
                        help='Drop specific (participant, norm) runs, e.g. '
                             'Chris:movement Maria:spacing. Norm matches by substring.')
    args = parser.parse_args()

    excludes = []
    for spec in args.exclude:
        pid, _, norm_frag = spec.partition(':')
        excludes.append((pid, norm_frag))

    participants = args.participants or DEFAULT_PARTICIPANTS
    summary = build_summary(participants, excludes)
    print_summary(summary)

    out_path = args.out or os.path.join(RESULTS_DIR, 'human_metrics_summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Metrics summary saved to: {out_path}")


if __name__ == '__main__':
    main()
