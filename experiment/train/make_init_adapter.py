"""Save a fresh, untrained LoRA adapter for a base model.

PEFT initialises lora_B to zero, so the adapter is an exact no-op on the base
weights. Used as the RLOO starting point (and the adapter vLLM first serves)
when a base model has no SFT cold start, e.g. Ministral 3:

    python experiment/train/make_init_adapter.py \
        --base_model /path/to/Ministral-3-8B-Instruct-2512-BF16 --out <dir>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.train.modeling import load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    model = load_model(args.base_model, train_lora=True)
    model.save_pretrained(args.out)
    print(f'[OK] fresh LoRA adapter -> {args.out}')


if __name__ == '__main__':
    main()
