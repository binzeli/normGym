# DQN training and evaluation

DQN with MLP/CNN encoders, optional norm/collision penalties, dense navigation
shaping, fixed-transition TD diagnostics, and deterministic final evaluation.

## Setup and training

Run from the repository root:

```bash
pip install -r requirements.txt
python -m experiment.run.rl.run_rl prepare \
  --tag baseline --manifest outputs/rl_manifest.json
python -m experiment.run.rl.run_rl train \
  --manifest outputs/rl_manifest.json --index 0 --output-root outputs/rl_runs
```

The default manifest has 300 runs: 50 norms × two encoders × three conditions:
`sparse_penalty`, `sparse_no_penalty`, and `dense_penalty`.
`--conditions` also accepts `dense_no_penalty`. Use `--norms`, `--policies`,
`--timesteps`, and `--eval-episodes` to reduce a run. Preparation does not train
or submit jobs; `train` executes one manifest index.

Defaults: 1M transitions, 200-step training episodes, Adam at 1e-4, gamma .99,
100k replay capacity, batch 32, learning after 1,000 transitions, one update per
four steps, target updates every 10k steps, and epsilon 1→.05 over the first
30%. Both encoders receive the same observations. Greedy selection and TD
targets mask walls/bounds; exploration samples all five actions.

Penalty removal affects only the learner's training reward. Dense shaping adds
signed BFS progress rewards with magnitude 6. Validation uses separate seeds
and two modified cafe layouts at fixed checkpoints, without selecting models
or updating hyperparameters. Manifests freeze source hashes, layouts, and seeds;
prepare a new manifest after editing source. Output tags must be new.

## Evaluation

After training all selected configurations for a norm:

```bash
python -m experiment.run.rl.run_rl final-test \
  --manifest outputs/rl_manifest.json --output-root outputs/rl_runs \
  --norm zigzag_right2_line
# Repeat for each norm, then aggregate:
python -m experiment.evaluation.rl.summarize --manifest outputs/rl_manifest.json \
  --output-root outputs/rl_runs --report outputs/rl_summary
```

Evaluation uses the final checkpoint on three seen-layout and three unseen-layout
episodes, each with a 100-step limit. Group assignments are seeded rather than
forced to rotate. Passive lifecycle records capture background completion times.
The exporter writes CSV, JSON, and LaTeX summaries: Task gives 1/0.5/0 for
two/one/no goals; Joint requires both goals and zero norm violations.

Replay-buffer snapshots are disabled by default. Logging is thinned to every
100 steps and episode ends; per-run outputs are bounded at 256 MiB with at least
1 GiB free disk space.
