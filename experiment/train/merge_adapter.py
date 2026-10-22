"""Merge a saved LoRA adapter into the base model for eval.

Usage (from the repo root, .venv active):
    python experiment/train/merge_adapter.py \
        --adapter experiment/train/checkpoints/v1_action_rloo/line11/adapter_best \
        --out experiment/train/checkpoints/v1_action_rloo/line11/merged_best

The merged directory is loadable by run_experiment.py --provider hf, same as
the merged_<tag> checkpoints ppo_train.py --merge_checkpoints writes inline.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.train.config import BASE_MODEL
from experiment.train.modeling import load_model, load_processor, save_merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--base_model', default=BASE_MODEL)
    args = parser.parse_args()

    model = load_model(args.base_model, adapter_path=args.adapter)
    processor = load_processor(args.base_model)
    out = save_merged(model, processor, args.out)
    print(f'[OK] merged checkpoint: {out}', flush=True)


if __name__ == '__main__':
    main()
