"""Evaluate a (merged) checkpoint through the unchanged experiment pipeline.

For each norm this runs the full pipeline (interactive stage, then the 3
same-map test games and the 3 new-map test games of experiment_config.json;
every test game ends when the subject reaches its goal or test_max_steps
runs out) with the checkpoint served locally via the pipeline's own
``--provider hf`` client, then computes metrics with the existing
``experiment/evaluation/eval.py``. Nothing about evaluation is redefined.

Usage:
    # Held-out norms of $NORM_GROUP (the default):
    python experiment/train/eval_checkpoint.py \
        --model_path experiment/train/checkpoints/v1_action_rloo/<group>/<run>/merged_adapter_best

    # Training norms (the benchmark seeds are disjoint from the RL rollout seeds):
    python experiment/train/eval_checkpoint.py --model_path ... --norms train

Merge the LoRA adapter first with experiment/train/merge_adapter.py.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.train.config import (PROJECT_ROOT, TEST_NORMS, TRAIN_NORMS,
                                     load_experiment_config)

RAW_RUNS_RE = re.compile(r'^\[OK\] Raw run logs saved to: (.+)$')
AGENT_TYPES = ('llm', 'memory_llm', 'full_memory_llm')


def resolve_output_dir(output_dir):
    """Same rule as run_experiment.py: relative paths live under experiment/."""
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(PROJECT_ROOT, 'experiment', output_dir)


def existing_summary(output_dir, agent_type, norm):
    """metrics_summary_<norm>.json of a finished run of this norm, if any.

    run_experiment.py writes <output_dir>/<agent_type>/<model_dir>/<world>/<norm>/
    and eval.py puts the summary next to that dir's raw_runs; the model dir
    name is the flattened model path, so it is globbed rather than rebuilt.
    """
    world = load_experiment_config().get('world_type', 'cafe')
    pattern = os.path.join(output_dir, agent_type, '*', world, norm,
                           f'metrics_summary_{norm}.json')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def run_one_norm(model_path, norm, agent_type, output_dir, extra_args, log_dir):
    command = [
        sys.executable, '-u',
        os.path.join(PROJECT_ROOT, 'experiment', 'run', 'run_experiment.py'),
        '--norm_type', norm,
        '--agent_type', agent_type,
        '--provider', 'hf',
        '--model', model_path,
        '--output_dir', output_dir,
    ] + extra_args
    print(f'[RUN] {" ".join(command)}', flush=True)
    # The HF client prints every prompt and response, and a full-memory
    # prompt is tens of KB, so the run's output goes to a per-norm log file
    # instead of memory; only its tail is echoed here.
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{norm}.log')
    with open(log_path, 'w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                text=True, cwd=PROJECT_ROOT)

    raw_runs_dir = None
    tail = deque(maxlen=40)
    with open(log_path, 'r', encoding='utf-8', errors='replace') as log:
        for line in log:
            tail.append(line)
            match = RAW_RUNS_RE.match(line.rstrip('\n'))
            if match:
                raw_runs_dir = match.group(1).strip()
    sys.stdout.write(''.join(tail))
    print(f'[LOG] full run log: {log_path}', flush=True)
    if result.returncode != 0:
        print(f'[FAIL] run_experiment failed for {norm} (exit {result.returncode})',
              flush=True)
        return None
    if not raw_runs_dir or not os.path.isdir(raw_runs_dir):
        print(f'[FAIL] could not locate raw_runs for {norm}', flush=True)
        return None

    eval_cmd = [sys.executable, '-u',
                os.path.join(PROJECT_ROOT, 'experiment', 'evaluation', 'eval.py'),
                raw_runs_dir]
    print(f'[EVAL] {" ".join(eval_cmd)}', flush=True)
    eval_result = subprocess.run(eval_cmd, cwd=PROJECT_ROOT)
    if eval_result.returncode != 0:
        print(f'[FAIL] eval.py failed for {norm}', flush=True)
        return None

    # eval.py writes metrics_summary_<norm>.json next to raw_runs.
    summaries = glob.glob(
        os.path.join(os.path.dirname(raw_runs_dir), 'metrics_summary_*.json')
    )
    if not summaries:
        return None
    with open(summaries[0], 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True,
                        help='Merged checkpoint directory (loadable by '
                             'AutoModelForImageTextToText.from_pretrained) '
                             'or a Hub id (e.g. the base model as a control)')
    parser.add_argument('--norms', nargs='+', default=['test'],
                        help="Norm names, or the shorthands 'test' (held-out "
                             "norms of $NORM_GROUP) / 'train' (its training norms)")
    parser.add_argument('--agent_type', default='memory_llm', choices=AGENT_TYPES,
                        help="Subject kind the checkpoint plays as. The RLOO "
                             "adapters were trained on full_memory_llm prompts "
                             "(whole episode in the prompt, no belief line); "
                             "memory_llm is what the API-model benchmark uses.")
    parser.add_argument('--output_dir', default='results_finetuned',
                        help='Passed to run_experiment.py --output_dir '
                             '(relative to experiment/, or absolute)')
    parser.add_argument('--summary_out', default=None,
                        help='Where to write the combined metrics JSON '
                             '(default: <output_dir>/eval_summary.json)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Reuse a norm whose metrics_summary_<norm>.json '
                             'already exists under --output_dir instead of '
                             'replaying it (resubmit after a timeout)')
    parser.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                        help='Extra args forwarded to run_experiment.py '
                             '(e.g. --extra --max_steps 100)')
    args = parser.parse_args()

    norms = []
    for entry in args.norms:
        if entry == 'test':
            norms.extend(TEST_NORMS)
        elif entry == 'train':
            norms.extend(TRAIN_NORMS)
        else:
            norms.append(entry)

    # a local dir -> absolute; anything else (a Hub id such as the base model) as given
    model_path = os.path.abspath(args.model_path) if os.path.isdir(args.model_path) else args.model_path
    output_dir = resolve_output_dir(args.output_dir)
    log_dir = os.path.join(output_dir, 'logs')
    summary = {'model_path': model_path, 'agent_type': args.agent_type,
               'norms': {}}
    failures = []
    for norm in norms:
        if args.skip_existing:
            done = existing_summary(output_dir, args.agent_type, norm)
            if done:
                print(f'[SKIP] {norm}: reusing {done}', flush=True)
                with open(done, 'r', encoding='utf-8') as f:
                    summary['norms'][norm] = json.load(f)
                continue
        metrics = run_one_norm(model_path, norm, args.agent_type,
                               output_dir, args.extra, log_dir)
        if metrics is None:
            failures.append(norm)
        else:
            summary['norms'][norm] = metrics

    summary['failures'] = failures
    out_path = args.summary_out or os.path.join(output_dir, 'eval_summary.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'[OK] combined summary: {out_path}')
    if failures:
        print(f'[WARN] failed norms: {failures}')
        sys.exit(1)


if __name__ == '__main__':
    main()
