"""Greedy evaluation of a served model (base or the training job's LoRA) on
a record set — used for the test-norm generalisation check while a run is
training. Usage:
    python experiment/train/vigal/v3_violation_judge/eval_records.py \
        --records experiment/train/data/queuing/violation_judge/test \
        --rollout_url http://gh1xx:PORT/v1 --model policy --n 400
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from experiment.train.policy_client import OpenAIBackend
from experiment.train.vigal.v3_violation_judge.step_rloo_judge_train import (
    balanced_subset, load_records, run_eval,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--records', required=True)
    p.add_argument('--rollout_url', required=True)
    p.add_argument('--model', default='policy')
    p.add_argument('--n', type=int, default=400)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--num_workers', type=int, default=32)
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument('--show_feedback', type=int, default=1)
    p.add_argument('--min_step', type=int, default=50,
                   help='skip records before this env step (match training)')
    p.add_argument('--correct_reward', type=float, default=1.0)
    p.add_argument('--format_reward', type=float, default=0.1)
    p.add_argument('--top_p', type=float, default=1.0)
    args = p.parse_args()
    recs = balanced_subset(load_records(args.records, min_step=args.min_step),
                           args.n, random.Random(args.seed))
    backend = OpenAIBackend(args.rollout_url, args.model, top_p=args.top_p)
    out = run_eval(backend, recs, args, out_path=args.out)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}))


if __name__ == '__main__':
    main()
