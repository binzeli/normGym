"""Zero-shot baseline: the experiment's test runs without the interactive stage.

The subject goes straight to the same-world and new-world test runs of
run_experiment.py, with no interactive learning and no reference memory. Test
seeds, setup, background turnover, stopping rule and raw run IDs are identical
to the full experiment, so the two are directly comparable. --ground_truth
with|without controls whether an LLM subject sees the norm description.

Usage:
    python experiment/run/zero_shot/run_zero_shot.py --norm_type <norm> \
        --ground_truth with|without [--agent_type llm|memory_llm|rl] ...

Takes the same flags as run_experiment.py. Results go to
experiment/<output_dir>/zero_shot/<with|without>_gt/<agent_type>/...; compute
metrics with experiment/evaluation/eval.py as usual.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from experiment.run.run_experiment import cli  # noqa: E402


if __name__ == "__main__":
    cli(mode='zero_shot')
