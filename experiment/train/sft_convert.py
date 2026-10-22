"""Build the SFT cold-start dataset from existing API-model experiment runs.

The pipeline's stdout logs (``simulation_log/experiment_results_<norm>.txt``)
contain, verbatim, every system prompt, user prompt and raw model response of
the run — so no prompt reconstruction is needed: this script parses those logs
and pairs each turn with the step image the model saw
(``simulation_figures/<stage>/step_NNN.png``).

Quality filter (an SFT example teaches procedure, so bad episodes are dropped
aggressively):
- interactive stage: a subject incarnation contributes its turns only if it
  finished all its goals with zero norm violations;
- test stages: a run contributes only if the subject completed the run with
  zero norm violations;
- any turn whose response does not parse under the pipeline's own parser is
  dropped.

Usage:
    python experiment/train/sft_convert.py \
        [--norms zigzag_right2_line ...] [--models anthropic_claude-opus-5 ...] \
        [--val_fraction 0.05] [--no_test_stage]
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.agents.llm_agent import _parse_memory_response, _parse_test_response
from experiment.train.config import (
    DATA_DIR,
    RESULTS_ROOT,
    SFT_DATASET,
    SFT_VAL_DATASET,
    TRACE_AGENT_KIND,
    TRACE_MODEL_DIRS,
    TRAIN_NORMS,
)

_FIGURES_RE = re.compile(r'^Figures will be saved to: (.+?)/?\s*$')
_SYSTEM_RE = re.compile(r'^LLMAgent (C\d+) system prompt:\s*$')
_USER_RE = re.compile(r'^LLMAgent (C\d+) user prompt:\s*$')
_RESPONSE_RE = re.compile(r'^LLMAgent (C\d+) received response: ?(.*)$')
_AGENT_LINE_RE = re.compile(r'^LLMAgent C\d+ ')
_STEP_RE = re.compile(r'Current step:\s*(\d+)\.')


def parse_turns_from_log(log_path: str) -> List[Dict]:
    """Extract (figures_dir, agent_id, system, user, response) turns in order."""
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    turns = []
    figures_dir = None
    i = 0
    while i < len(lines):
        fig_match = _FIGURES_RE.match(lines[i])
        if fig_match:
            figures_dir = fig_match.group(1)
            i += 1
            continue

        sys_match = _SYSTEM_RE.match(lines[i])
        if not sys_match:
            i += 1
            continue

        agent_id = sys_match.group(1)
        i += 1
        system_lines = []
        while i < len(lines) and not _USER_RE.match(lines[i]):
            system_lines.append(lines[i])
            i += 1
        if i >= len(lines):
            break

        i += 1  # past the user-prompt marker
        user_lines = []
        while i < len(lines) and not _RESPONSE_RE.match(lines[i]):
            user_lines.append(lines[i])
            i += 1
        if i >= len(lines):
            break

        resp_match = _RESPONSE_RE.match(lines[i])
        response_lines = [resp_match.group(2)]
        i += 1
        # A response may span multiple lines (pretty-printed JSON); it ends at
        # the next agent-status line ("parsed action" / WARNING / next prompt).
        while i < len(lines) and not _AGENT_LINE_RE.match(lines[i]):
            response_lines.append(lines[i])
            i += 1

        turns.append({
            'figures_dir': figures_dir,
            'agent_id': agent_id,
            'system': '\n'.join(system_lines).strip('\n'),
            'user': '\n'.join(user_lines).strip('\n'),
            'response': '\n'.join(response_lines).strip(),
        })
    return turns


def _stage_of(figures_dir: Optional[str]) -> Optional[str]:
    """'interactive' | run_id ('test_001', 'new_map_test_002_hospital', ...)."""
    if not figures_dir:
        return None
    return os.path.basename(os.path.normpath(figures_dir))


def _load_raw_run(norm_dir: str, stage: str) -> Optional[Dict]:
    path = os.path.join(norm_dir, 'raw_runs', f'{stage}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _clean_incarnations(raw_run: Dict) -> set:
    """Interactive stage: agent_ids that finished all goals violation-free."""
    by_agent: Dict[str, List[Dict]] = defaultdict(list)
    for record in raw_run.get('trace', []):
        by_agent[record['agent_id']].append(record)
    clean = set()
    for agent_id, records in by_agent.items():
        violations = sum(int(r.get('step_norm_violations', 0)) for r in records)
        if violations == 0 and records and records[-1].get('done'):
            clean.add(agent_id)
    return clean


def _run_is_clean(raw_run: Dict) -> bool:
    """Test stages: whole-run filter (subject completed, zero violations)."""
    records = raw_run.get('trace', [])
    violations = sum(int(r.get('step_norm_violations', 0)) for r in records)
    return bool(records) and violations == 0 and records[-1].get('done')


def convert_norm_dir(norm_dir: str, norm: str, model_dir: str,
                     include_test_stage: bool) -> List[Dict]:
    log_path = os.path.join(
        norm_dir, 'simulation_log', f'experiment_results_{norm}.txt'
    )
    if not os.path.exists(log_path):
        print(f'[WARN] no stdout log, skipping: {log_path}')
        return []

    turns = parse_turns_from_log(log_path)
    raw_cache: Dict[str, Optional[Dict]] = {}
    clean_cache: Dict[str, set] = {}

    samples = []
    dropped = defaultdict(int)
    for turn in turns:
        stage = _stage_of(turn['figures_dir'])
        if stage is None:
            dropped['no_stage'] += 1
            continue
        is_interactive = stage == 'interactive'
        if not is_interactive and not include_test_stage:
            dropped['test_stage_excluded'] += 1
            continue

        if stage not in raw_cache:
            raw_cache[stage] = _load_raw_run(norm_dir, stage)
        raw_run = raw_cache[stage]
        if raw_run is None:
            dropped['no_raw_run'] += 1
            continue

        # Quality filter
        if is_interactive:
            if stage not in clean_cache:
                clean_cache[stage] = _clean_incarnations(raw_run)
            if turn['agent_id'] not in clean_cache[stage]:
                dropped['dirty_incarnation'] += 1
                continue
        else:
            if not _run_is_clean(raw_run):
                dropped['dirty_run'] += 1
                continue

        # Parse filter, using the pipeline's own parsers
        parser = _parse_memory_response if is_interactive else _parse_test_response
        if parser(turn['response']) is None:
            dropped['unparseable_response'] += 1
            continue

        # Step image the model saw when it produced this response
        step_match = _STEP_RE.search(turn['user'])
        step = int(step_match.group(1)) if step_match else None
        image_path = None
        if step is not None and turn['figures_dir']:
            candidate = os.path.join(turn['figures_dir'], f'step_{step:03d}.png')
            if os.path.exists(candidate):
                image_path = candidate
        if image_path is None:
            dropped['missing_image'] += 1
            continue

        samples.append({
            'system': turn['system'],
            'user': turn['user'],
            'images': [image_path],
            'response': turn['response'],
            'norm': norm,
            'source_model': model_dir,
            'stage': stage,
            'agent_id': turn['agent_id'],
            'step': step,
        })

    print(f'[OK] {model_dir}/{norm}: {len(samples)} samples kept '
          f'(dropped: {dict(dropped) or "none"})')
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--norms', nargs='+', default=TRAIN_NORMS)
    parser.add_argument('--models', nargs='+', default=TRACE_MODEL_DIRS)
    parser.add_argument('--worlds', nargs='+', default=['cafe'])
    parser.add_argument('--val_fraction', type=float, default=0.05)
    parser.add_argument('--no_test_stage', action='store_true',
                        help='Keep only interactive-stage turns')
    parser.add_argument('--out', default=SFT_DATASET)
    parser.add_argument('--val_out', default=SFT_VAL_DATASET)
    args = parser.parse_args()

    all_samples = []
    for model_dir in args.models:
        for world in args.worlds:
            for norm in args.norms:
                norm_dir = os.path.join(
                    RESULTS_ROOT, TRACE_AGENT_KIND, model_dir, world, norm
                )
                if not os.path.isdir(norm_dir):
                    print(f'[WARN] no results dir, skipping: {norm_dir}')
                    continue
                all_samples.extend(convert_norm_dir(
                    norm_dir, norm, model_dir,
                    include_test_stage=not args.no_test_stage,
                ))

    # Split by (model, norm, stage) so a whole episode lands on one side.
    train_samples, val_samples = [], []
    for sample in all_samples:
        key = f"{sample['source_model']}|{sample['norm']}|{sample['stage']}"
        bucket = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 1000
        (val_samples if bucket < args.val_fraction * 1000
         else train_samples).append(sample)

    for path, samples in ((args.out, train_samples), (args.val_out, val_samples)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f'[OK] wrote {len(samples)} samples to {path}')

    by_norm = defaultdict(int)
    for sample in train_samples:
        by_norm[sample['norm']] += 1
    print(f'[OK] train samples per norm: {dict(sorted(by_norm.items()))}')


if __name__ == '__main__':
    main()
