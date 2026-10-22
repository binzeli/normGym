#!/bin/bash
# Runs the zero-shot baseline (run_zero_shot.py + eval.py): the test episodes
# only, without the interactive learning stage, one norm after another, every
# model per norm. Ends with the benchmark table from calculate_metrics.py.
# Settings are read from the environment, e.g.
#   MODELS="openai/gpt-5.6-sol" GROUND_TRUTH=with bash scripts/run_zero_shot.sh
#   CONFIG=experiment/run/zero_shot/experiment_config_same_map1.json bash scripts/run_zero_shot.sh

set -euo pipefail

# Run from the repo root
cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

# Activate virtual environment
source .venv/bin/activate

# Default: all norms, in src/norms/norm_descriptions.json order.
ALL_NORMS="$(python -c "import json; print(' '.join(n['name'] for n in json.load(open('src/norms/norm_descriptions.json'))['norms']))")"

NORM_TYPES="${NORM_TYPES:-$ALL_NORMS}"
# Space-separated model ids: OpenRouter slugs for PROVIDER=openrouter, HF ids
# or local checkpoint paths for PROVIDER=hf (1 GPU).
MODELS="${MODELS:-openai/gpt-5.6-sol}"
AGENT_TYPE="${AGENT_TYPE:-memory_llm}"
PROVIDER="${PROVIDER:-openrouter}"
# with = the subject is told the norm description; without = it must infer it.
GROUND_TRUTH="${GROUND_TRUTH:-without}"
# Shared config: 3 same-map + 3 new-map tests per norm. The same_map1 config
# runs one same-map test per norm and no new-map tests.
CONFIG="${CONFIG:-experiment/run/experiment_config.json}"
# Results go to experiment/<OUTPUT_DIR>/zero_shot/<GROUND_TRUTH>_gt/<agent>/<model>/<world>/<norm>/
OUTPUT_DIR="${OUTPUT_DIR:-results}"
WORLD_TYPE="$(python -c "import json; print(json.load(open('$CONFIG')).get('world_type', 'cafe'))")"
# SKIP_EXISTING=1 (default) skips any norm/model that already has its
# metrics_summary_<norm>.json, so a rerun only fills in what is missing.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

case "$OUTPUT_DIR" in
    /*) RESULTS_ROOT="$OUTPUT_DIR/zero_shot/${GROUND_TRUTH}_gt" ;;
    *)  RESULTS_ROOT="experiment/$OUTPUT_DIR/zero_shot/${GROUND_TRUTH}_gt" ;;
esac

echo "=== START: $(date) ==="
echo "models=$MODELS agent_type=$AGENT_TYPE provider=$PROVIDER ground_truth=$GROUND_TRUTH"
echo "config=$CONFIG world=$WORLD_TYPE results=$RESULTS_ROOT"

# Mirror run_experiment.py's model dir naming: '/' and ':' become '_'.
model_dir() { local m="${1//\//_}"; echo "${m//:/_}"; }

# One norm/model failing must not cancel the ones after it; collect and report.
failures=()

for NORM_TYPE in $NORM_TYPES; do
for MODEL in $MODELS; do
    NORM_DIR="$RESULTS_ROOT/$AGENT_TYPE/$(model_dir "$MODEL")/$WORLD_TYPE/$NORM_TYPE"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$NORM_DIR/metrics_summary_${NORM_TYPE}.json" ]; then
        echo "[SKIP] $NORM_DIR already has metrics_summary_${NORM_TYPE}.json"
        continue
    fi

    echo
    echo "=== [$(date)] NORM: $NORM_TYPE | MODEL: $MODEL ==="
    rc=0
    python -u experiment/run/zero_shot/run_zero_shot.py \
        --norm_type "$NORM_TYPE" \
        --ground_truth "$GROUND_TRUTH" \
        --agent_type "$AGENT_TYPE" \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR" || rc=$?
    if [ "$rc" -ne 0 ]; then
        # Exit code 3 = provider account out of credits; every remaining run
        # would fail the same way.
        if [ "$rc" -eq 3 ]; then
            echo "[FATAL] Out of LLM credits at $NORM_TYPE/$MODEL; aborting the whole run."
            exit 3
        fi
        echo "[FAIL] run_zero_shot.py failed for $NORM_TYPE/$MODEL; skipping its eval"
        failures+=("$NORM_TYPE/$MODEL (run)")
        continue
    fi

    echo "=== [$(date)] EVAL: $NORM_TYPE/$MODEL ==="
    if ! python -u experiment/evaluation/eval.py "$NORM_DIR/raw_runs"; then
        echo "[FAIL] eval.py failed for $NORM_TYPE/$MODEL"
        failures+=("$NORM_TYPE/$MODEL (eval)")
    fi
done
done

# Benchmark table per model; the new-map table is omitted when there are no
# new-map runs (e.g. the same_map1 config).
for MODEL in $MODELS; do
    echo
    echo "=== TABLE: $MODEL ==="
    python experiment/evaluation/calculate_metrics.py "$AGENT_TYPE/$(model_dir "$MODEL")" \
        --results-root "$RESULTS_ROOT" --world "$WORLD_TYPE" \
        | tee "$RESULTS_ROOT/$AGENT_TYPE/$(model_dir "$MODEL")/metrics_table.txt" \
        || failures+=("$MODEL (table)")
done

echo
if [ ${#failures[@]} -gt 0 ]; then
    echo "=== FAILURES: ${failures[*]} ==="
fi
echo "=== END: $(date) ==="

deactivate

if [ ${#failures[@]} -gt 0 ]; then
    exit 1
fi
