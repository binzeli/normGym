"""Regrade the stored interactive-stage norm belief in every metrics summary.

Walks ``<results-root>/<agent>/<model>/<world>/<norm>/metrics_summary_<norm>.json``,
re-runs ``experiment.utils.norm_judge.judge_norm_belief`` on the belief already
recorded under ``norm_belief_judge`` (falling back to the interactive-stage
memory file when the section is missing) and rewrites that section under the
current rubric (7-point Likert: ``likert`` 1-7, ``score`` = (likert-1)/6).
The previous 0-1 score is kept as ``previous_score`` for comparison.

Usage:
    python experiment/evaluation/rejudge_beliefs.py                 # all memory_llm models
    python experiment/evaluation/rejudge_beliefs.py --models openai_gpt-5.6-sol --dry-run
"""
import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from experiment.utils.client import OpenRouterClient  # noqa: E402
from experiment.utils.norm_judge import (  # noqa: E402
    DEFAULT_JUDGE_MODEL, ground_truth_norm_description, judge_norm_belief)

RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'experiment', 'results')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', default='memory_llm')
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--world', default='cafe')
    ap.add_argument('--judge-model', default=DEFAULT_JUDGE_MODEL)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    base = os.path.join(RESULTS_ROOT, args.agent)
    models = args.models or sorted(os.listdir(base))
    client = None if args.dry_run else OpenRouterClient(model=args.judge_model)
    n = 0
    for model in models:
        for path in sorted(glob.glob(os.path.join(base, model, args.world, '*',
                                                  'metrics_summary_*.json'))):
            with open(path, encoding='utf-8') as f:
                summary = json.load(f)
            norm = summary['norm_type']
            old = summary.get('norm_belief_judge') or {}
            belief = old.get('belief') or ''
            if not belief:
                mem = os.path.join(os.path.dirname(path), 'simulation_log',
                                   'interactive_stage.json')
                if os.path.isfile(mem):
                    with open(mem, encoding='utf-8') as f:
                        belief = str(json.load(f).get('belief', '') or '')
            if not belief:
                print(f'[SKIP] {model}/{norm}: no belief recorded')
                continue
            if args.dry_run:
                print(f'{model}/{norm}: old score {old.get("score")}')
                continue
            verdict = judge_norm_belief(belief, ground_truth_norm_description(norm),
                                        client=client)
            if 'score' in old:
                verdict['previous_score'] = old['score']
            summary['norm_belief_judge'] = verdict
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
            n += 1
            print(f'{model}/{norm}: likert {verdict["likert"]} '
                  f'(score {verdict["score"]}, was {old.get("score")})', flush=True)
    if client:
        print(f'\nRegraded {n} summaries. ' + client.cost_summary_line())


if __name__ == '__main__':
    main()
