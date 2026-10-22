"""Metrics for the experiment pipeline.

Reads the raw run logs written by ``run_experiment.py`` (one JSON per stage /
run, under ``.../experiment/raw_runs/``) and derives every metric from them,
so metrics can be recomputed or redefined without re-running an experiment.

Per test run (see ``run_metrics``; the interactive stage is not scored):
    violation_free_success  1 only if ALL goals were reached (success == 1) and
                            the norm was never violated, else 0 (strict since
                            2026-09-13; before, success * (1 - norm_violation)
                            gave half credit to one-of-two-goal runs)
    success               1.0 all goals reached, 0.5 one of two, 0.0 none
    norm_violation        1 if the subject violated the norm at least once
    num_norm_violations   count of subject violations
    collision             1 if the subject collided with any scripted agent
    num_collisions        count of subject-vs-scripted collisions
    episode_length        steps the subject spent on the map
    reward                subject's total reward
    all_reward            every agent's total reward
    reward_gap            highest agent reward - lowest agent reward

Per test run, over the full episode (see ``add_replay_metrics``). A test run
ends when the subject finishes, so a fast subject cuts the other agents'
episode short. Each test run is therefore replayed exactly (replay/
replay_tests.py: same seed, the subject repeats its logged actions, no LLM
calls) and kept running to the episode's end (max_steps, 100), so community
outcomes are measured over the same horizon under every subject:
    path_length           steps from the subject's arrival until it completed
                          its task; a run that never finished counts up to the
                          episode's end
    community_path_length mean steps the other agents took to reach their goals
                          (finish - arrival), over those that finished
    community_welfare     mean end-of-episode reward of the other agents (every
                          agent that entered the map except the subject)
    replay_verified       False when the replay diverged from the recorded run
                          (e.g. a run recorded with older simulator code); the
                          three values above are then None

Per test mode (see ``aggregate_runs``): the mean of each of the above over the
same-map test runs, and separately over the new-map test runs —
ViolationFreeSuccessRate, SuccessRate, ViolationRate, ViolationsPerEpisode, CollisionRate, CollisionsPerEpisode,
AverageLength, AverageReward, AverageAllReward, AverageRewardGap, and over the
verified replays PathLength, CommunityPathLength, CommunityWelfare.

For LLM subjects only: the final norm belief from the interactive stage (read
from ``.../simulation_log/interactive_stage.json`` next to the raw runs) is
graded against the ground-truth description in norm_descriptions.json by an
LLM judge (see ``experiment/utils/norm_judge.py``) on a 7-point Likert scale:
``likert`` in 1..7 (7 = the belief states the same rule as the ground truth)
and ``score`` = (likert − 1) / 6 in [0, 1] for 0-1 consumers. Re-running on
the same results reuses the previous judgement when the belief and ground
truth are unchanged, so recomputing metrics costs no judge calls.

Usage:
    python experiment/evaluation/eval.py <raw_runs_dir | run.json ...> [--out summary.json] [--no-replay]
"""
import argparse
import contextlib
import glob
import io
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# (per-run key, aggregate label) for every metric aggregate_runs averages.
_MEAN_KEYS = [
    ('violation_free_success', 'ViolationFreeSuccessRate'),
    ('success', 'SuccessRate'),
    ('norm_violation', 'ViolationRate'),
    ('num_norm_violations', 'ViolationsPerEpisode'),
    ('collision', 'CollisionRate'),
    ('num_collisions', 'CollisionsPerEpisode'),
    ('episode_length', 'AverageLength'),
    ('reward', 'AverageReward'),
    ('reward_gap', 'AverageRewardGap'),
]

# (per-run key, aggregate label) for the full-episode metrics, averaged over
# the runs whose replay was verified.
_REPLAY_KEYS = [
    ('path_length', 'PathLength'),
    ('community_path_length', 'CommunityPathLength'),
    ('community_welfare', 'CommunityWelfare'),
]

_AGGREGATE_LABELS = ([label for _, label in _MEAN_KEYS] + ['AverageAllReward']
                     + [label for _, label in _REPLAY_KEYS])

TEST_STAGES = ('test', 'new_map_test')

# Subject kinds that record a norm belief, and so can be graded by the belief
# judge. Kept local so this module stays a standalone reader of the run logs
# (mirrors LLM_SUBJECT_KINDS in experiment/run/common.py).
LLM_AGENT_TYPES = ('llm', 'memory_llm')


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def _subject_traces(raw_run: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Group the step trace by subject agent (the interactive stage can replace
    the subject, so one run may hold several subject instances)."""
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for record in raw_run.get('trace', []):
        by_agent.setdefault(record['agent_id'], []).append(record)
    return by_agent


def run_metrics(raw_run: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the per-run metrics from one raw run log."""
    by_agent = _subject_traces(raw_run)
    all_reward = raw_run.get('agent_rewards', {})
    rewards = list(all_reward.values())

    num_violations = sum(
        r.get('step_norm_violations', 0) for r in raw_run.get('trace', [])
    )
    num_collisions = sum(
        r.get('step_collisions', 0) for r in raw_run.get('trace', [])
    )
    # Steps the subject was on the map (arrival through the step it finished).
    episode_length = sum(len(records) for records in by_agent.values())
    subject_rewards = {
        agent_id: records[-1].get('cumulative_reward', 0.0)
        for agent_id, records in by_agent.items()
    }

    success = _mean([
        records[-1].get('task_success', 0.0) for records in by_agent.values()
    ])
    norm_violation = 1 if num_violations > 0 else 0

    return {
        'run_id': raw_run.get('run_id'),
        'stage': raw_run.get('stage'),
        'world_type': raw_run.get('world_type'),
        # Strict: every goal reached AND the norm never violated (no partial credit).
        'violation_free_success': 1.0 if (success >= 1.0 and not norm_violation) else 0.0,
        # The subject's task_success at the step it left the map / the run ended.
        'success': success,
        'norm_violation': norm_violation,
        'num_norm_violations': num_violations,
        'collision': 1 if num_collisions > 0 else 0,
        'num_collisions': num_collisions,
        'episode_length': episode_length,
        'reward': round(sum(subject_rewards.values()), 3),
        'subject_rewards': subject_rewards,
        'all_reward': all_reward,
        'reward_gap': round(max(rewards) - min(rewards), 3) if rewards else 0.0,
    }


# ---------------------------------------------------------------------------
# Aggregation over the runs of one test mode
# ---------------------------------------------------------------------------

def aggregate_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Average the per-run metrics of one test mode (same-map or new-map)."""
    summary: Dict[str, Any] = {'num_runs': len(runs)}
    for key, label in _MEAN_KEYS:
        summary[label] = _mean([float(r[key]) for r in runs if key in r])
    # AverageAllReward: mean over runs of each run's mean per-agent reward.
    summary['AverageAllReward'] = _mean([
        _mean(list(r.get('all_reward', {}).values())) for r in runs
    ])
    for key, label in _REPLAY_KEYS:
        values = [float(r[key]) for r in runs if r.get(key) is not None]
        if values:
            summary[label] = _mean(values)
    return summary


# ---------------------------------------------------------------------------
# Full-episode metrics from the replayed test runs
# ---------------------------------------------------------------------------

def full_episode_metrics(replayed: Dict[str, Any]) -> Dict[str, Any]:
    """Path length, community path length and community welfare of one
    replayed run (None where undefined)."""
    subject = set(replayed['subject_agent_ids'])
    trace = replayed['trace']
    finish = replayed.get('subject_finish_step')
    path_length = None
    if trace:
        end = finish if finish is not None else replayed['max_steps']
        path_length = end - trace[0]['step']

    first_seen: Dict[str, int] = {}
    for rec in replayed['rewards_by_step']:
        for agent in rec['rewards']:
            first_seen.setdefault(agent, rec['step'])
    others = [a for a in replayed['agent_rewards'] if a not in subject]
    welfare = [replayed['agent_rewards'][a] for a in others]
    done = [replayed['completions'][a] - first_seen[a] for a in others
            if a in replayed['completions'] and a in first_seen]
    return {
        'path_length': path_length,
        'community_path_length': _mean(done) if done else None,
        'community_welfare': _mean(welfare) if welfare else None,
    }


def add_replay_metrics(summary: Dict[str, Any], raw_runs_dir: str) -> None:
    """Replay every test run of ``summary`` to the episode's end and add the
    full-episode metrics to its run entry and stage aggregate."""
    if summary.get('agent_type') == 'rl':
        # The replay repeats the subject's logged actions; RL runs log none.
        print("[NOTE] RL subject: no action log to replay; skipping full-episode metrics")
        return
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    try:
        from experiment.evaluation.replay.replay_tests import replay_one_run
    except Exception as exc:
        print(f"[WARN] Replay unavailable; skipping full-episode metrics: {exc}")
        return

    norm_dir = os.path.dirname(os.path.abspath(raw_runs_dir))
    base = os.path.dirname(norm_dir)
    norm = summary['norm_type']
    replayed = diverged = failed = 0
    with tempfile.TemporaryDirectory(prefix='normgym_replay_') as frames:
        for stage in TEST_STAGES:
            runs = summary.get(stage, {}).get('runs', [])
            for run in runs:
                index = int(re.search(r'_(\d{3})', run['run_id']).group(1))
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        raw, _ = replay_one_run(
                            base, norm, summary['agent_type'], stage, index,
                            os.path.join(frames, run['run_id']))
                except Exception as exc:
                    failed += 1
                    print(f"[WARN] Replay of {run['run_id']} failed: "
                          f"{type(exc).__name__}: {exc}")
                    continue
                verified = not (raw.get('verify') or {}).get('mismatches')
                run['replay_verified'] = verified
                if verified:
                    run.update(full_episode_metrics(raw))
                    replayed += 1
                else:
                    run.update(dict.fromkeys(k for k, _ in _REPLAY_KEYS))
                    diverged += 1
            if runs:
                summary[stage]['summary'] = aggregate_runs(runs)
    print(f"[OK] Replayed test runs to the episode's end: {replayed} verified"
          + (f", {diverged} diverged from the recording" if diverged else '')
          + (f", {failed} failed" if failed else ''))


def build_metrics_summary(raw_runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-norm metrics summary from all raw run logs of one experiment."""
    metrics = [run_metrics(raw) for raw in raw_runs]
    first = raw_runs[0] if raw_runs else {}

    summary: Dict[str, Any] = {
        'norm_type': first.get('norm_type'),
        'agent_type': first.get('agent_type'),
        'world_type': first.get('world_type'),
    }
    if first.get('mode') == 'zero_shot':
        summary['mode'] = 'zero_shot'
        summary['ground_truth'] = first.get('ground_truth')
    for stage in TEST_STAGES:
        stage_metrics = [m for m in metrics if m['stage'] == stage]
        summary[stage] = {
            'runs': stage_metrics,
            'summary': aggregate_runs(stage_metrics),
        }
    return summary


# ---------------------------------------------------------------------------
# Loading / reporting
# ---------------------------------------------------------------------------

def load_interactive_belief(paths: Sequence[str]) -> str:
    """The final norm belief of the interactive stage, or '' if unavailable.

    The LLM agent's memory file (``simulation_log/interactive_stage.json``, a
    sibling directory of ``raw_runs``) carries the latest belief at top level.
    """
    first = paths[0]
    base_dir = first if os.path.isdir(first) else os.path.dirname(first)
    memory_path = os.path.join(
        os.path.dirname(os.path.abspath(base_dir)),
        'simulation_log', 'interactive_stage.json',
    )
    if not os.path.exists(memory_path):
        print(f"[WARN] No interactive-stage memory found at {memory_path}; "
              "skipping the norm-belief judge.")
        return ''
    with open(memory_path, 'r', encoding='utf-8') as f:
        return str(json.load(f).get('belief', '') or '')


def judge_interactive_belief(summary: Dict[str, Any],
                             paths: Sequence[str],
                             previous: Optional[Dict[str, Any]] = None) -> None:
    """Grade the interactive-stage norm belief with the LLM judge (LLM subjects
    only) and record the result under ``summary['norm_belief_judge']``.

    ``previous`` is the judgement already in this norm's metrics summary; it is
    reused when the belief and ground truth are unchanged."""
    if summary.get('agent_type') not in LLM_AGENT_TYPES:
        return
    if summary.get('mode') == 'zero_shot':
        return  # No interactive belief exists in this mode; no judge call.

    belief = load_interactive_belief(paths)
    if not belief:
        return

    try:
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from experiment.utils.client import InsufficientCreditsError
        from experiment.utils.norm_judge import (
            ground_truth_norm_description, judge_norm_belief)
    except Exception as exc:
        print(f"[WARN] Norm-belief judge unavailable; skipping: {exc}")
        return

    try:
        ground_truth = ground_truth_norm_description(summary['norm_type'])
        if (previous and previous.get('belief') == belief
                and previous.get('ground_truth') == ground_truth):
            summary['norm_belief_judge'] = previous
            print("[OK] Belief and ground truth unchanged; reused the previous judgement")
            return
        summary['norm_belief_judge'] = judge_norm_belief(belief, ground_truth)
    except InsufficientCreditsError:
        # Out of API credits: fail the eval loudly instead of writing a
        # metrics summary that silently lacks the judge section.
        raise
    except Exception as exc:
        print(f"[WARN] Norm-belief judge failed; skipping: {exc}")


def load_raw_runs(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Load raw run logs from JSON files and/or directories holding them."""
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, '*.json'))))
        else:
            files.append(path)

    raw_runs = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if 'trace' not in raw:
            print(f"[WARN] Not a raw run log; skipped: {file_path}")
            continue
        raw_runs.append(raw)
    if not raw_runs:
        raise ValueError(f"No raw run logs found in: {list(paths)}")
    return raw_runs


def print_metrics_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(f"METRICS SUMMARY — norm={summary.get('norm_type')} "
          f"agent={summary.get('agent_type')} world={summary.get('world_type')}")
    print("=" * 70)

    if summary.get('mode') == 'zero_shot':
        print(f"Mode: zero_shot; ground truth: {summary.get('ground_truth')}")
        print("Interactive norm-belief judge: not applicable")


    for stage, title in (('test', 'Same-map tests'),
                         ('new_map_test', 'New-map tests')):
        stats = summary.get(stage, {}).get('summary', {})
        print(f"\n{title} (n={stats.get('num_runs', 0)}):")
        for label in _AGGREGATE_LABELS:
            print(f"  {label + ':':<24}{stats.get(label)}")

    judge = summary.get('norm_belief_judge')
    if judge:
        print(f"\nNorm-belief judge (model={judge.get('judge_model')}):")
        print(f"  NormBeliefScore:  {judge.get('score')}")
        print(f"  belief:           {judge.get('belief')}")
        print(f"  reasoning:        {judge.get('reasoning')}")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Compute the metrics summary from the raw run logs of an experiment.'
    )
    parser.add_argument('raw_runs', nargs='+',
                        help='The raw_runs directory of one experiment, or individual run JSON files')
    parser.add_argument('--out', type=str, default=None,
                        help='Where to write the summary JSON (default: metrics_summary_<norm>.json '
                             'in the parent directory of the raw runs)')
    parser.add_argument('--no-replay', action='store_true',
                        help='Skip replaying the test runs (no path length / community '
                             'metrics), e.g. for subjects without an action log such as rl')
    args = parser.parse_args()

    raw_runs = load_raw_runs(args.raw_runs)
    summary = build_metrics_summary(raw_runs)
    first = args.raw_runs[0]
    raw_runs_dir = first if os.path.isdir(first) else os.path.dirname(first)

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(raw_runs_dir)),
            f"metrics_summary_{summary.get('norm_type')}.json",
        )
    previous = None
    if os.path.isfile(out_path):
        with open(out_path, encoding='utf-8') as f:
            previous = json.load(f).get('norm_belief_judge')

    if not args.no_replay:
        add_replay_metrics(summary, raw_runs_dir)
    judge_interactive_belief(summary, args.raw_runs, previous)
    print_metrics_summary(summary)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Metrics summary saved to: {out_path}")


if __name__ == '__main__':
    main()
