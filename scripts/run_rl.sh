#!/bin/bash
# Trains and tests the DQN baseline (experiment/run/rl) on one norm, on this
# machine: prepares a manifest for that norm, trains every configuration in it
# one after another, runs the final test (3 same-layout + 3 new-layout
# episodes per checkpoint, plus the oracle reference), then writes the summary
# tables. CPU only; no GPU or API key needed. Settings are read from the
# environment, e.g.
#   NORM_TYPE=xcell_forbidden bash scripts/run_rl.sh
#   NORM_TYPE=edge_left_line POLICIES=mlp CONDITIONS=sparse_penalty TIMESTEPS=100000 bash scripts/run_rl.sh
#
# The paper default is 6 configurations (2 encoders x 3 reward conditions) of
# 1M transitions each, which takes many hours on CPU. To train all 50 norms
# in parallel on a cluster, see experiment/run/rl/README.md.

set -euo pipefail

# Run from the repo root
cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

NORM_TYPE="${NORM_TYPE:-edge_left_line}"
# Encoders: mlp and/or cnn.
POLICIES="${POLICIES:-mlp cnn}"
# Reward conditions: sparse_penalty sparse_no_penalty dense_penalty dense_no_penalty.
CONDITIONS="${CONDITIONS:-sparse_penalty sparse_no_penalty dense_penalty}"
# Environment transitions per configuration.
TIMESTEPS="${TIMESTEPS:-1000000}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiment/results/rl_dqn}"
# Tags must be new: a tag's runs are never resumed or overwritten.
TAG="${TAG:-${NORM_TYPE}_$(date +%Y%m%d_%H%M%S)}"
MANIFEST="$OUTPUT_ROOT/manifests/$TAG.json"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg

# Activate virtual environment
source .venv/bin/activate

echo "=== START: $(date) ==="
echo "norm=$NORM_TYPE policies=$POLICIES conditions=$CONDITIONS timesteps=$TIMESTEPS"
echo "tag=$TAG output=$OUTPUT_ROOT/$TAG"

mkdir -p "$OUTPUT_ROOT/manifests"
# shellcheck disable=SC2086  # POLICIES / CONDITIONS are space-separated lists
python -m experiment.run.rl.run_rl prepare \
    --tag "$TAG" --manifest "$MANIFEST" --norms "$NORM_TYPE" \
    --policies $POLICIES --conditions $CONDITIONS --timesteps "$TIMESTEPS"

NUM_TASKS="$(python -c "import json, sys; print(len(json.load(open(sys.argv[1]))['tasks']))" "$MANIFEST")"

for ((i = 0; i < NUM_TASKS; i++)); do
    echo
    echo "=== [$(date)] TRAIN $((i + 1))/$NUM_TASKS ==="
    python -m experiment.run.rl.run_rl train \
        --manifest "$MANIFEST" --index "$i" --output-root "$OUTPUT_ROOT" --device "$DEVICE"
done

echo
echo "=== [$(date)] FINAL TEST: $NORM_TYPE ==="
python -m experiment.run.rl.run_rl final-test \
    --manifest "$MANIFEST" --output-root "$OUTPUT_ROOT" --norm "$NORM_TYPE"

echo
echo "=== [$(date)] SUMMARY ==="
python -m experiment.evaluation.rl.summarize \
    --manifest "$MANIFEST" --output-root "$OUTPUT_ROOT" --report "$OUTPUT_ROOT/$TAG/summary"

echo
echo "=== END: $(date) ==="
echo "Results: $OUTPUT_ROOT/$TAG/ (summary tables in $OUTPUT_ROOT/$TAG/summary)"

deactivate
