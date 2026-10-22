#!/bin/bash
# Runs the benchmark (run_experiment.py + eval.py) for open-weight models loaded
# locally with Hugging Face transformers (--provider hf), on this machine's GPU.
# A 48 GB GPU fits models up to ~20B in bf16 (the default 4B needs ~8 GB).
# Without NORM_TYPES it runs all 50 norms. Settings are read from the
# environment, e.g.
#   NORM_TYPES=bend_left2_line MODELS=google/gemma-3-4b-it bash scripts/run_experiment_hf.sh

set -euo pipefail

# Run from the repo root
cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

# Activate virtual environment
source .venv/bin/activate

# Norms run one after another in this order; each norm runs every model below.
# NORM_TYPE (singular) is still honoured for backwards compatibility.
# Default: all 50 norms, in src/norms/norm_descriptions.json order.
ALL_NORMS="$(python -c "import json; print(' '.join(n['name'] for n in json.load(open('src/norms/norm_descriptions.json'))['norms']))")"
NORM_TYPES="${NORM_TYPES:-${NORM_TYPE:-$ALL_NORMS}}"
AGENT_TYPE="${AGENT_TYPE:-memory_llm}"
# 'hf' loads the model in-process via HF transformers (needs a GPU).
# Works with any AutoModelForImageTextToText checkpoint or a local path.
PROVIDER="${PROVIDER:-hf}"
# Models run one after another in this order, each followed by its own eval.
# Space-separated HF ids, e.g. "Qwen/Qwen3-VL-4B-Instruct google/gemma-3-4b-it".
MODELS="${MODELS:-Qwen/Qwen3-VL-4B-Instruct}"
# Must match world_type in experiment/run/experiment_config.json; only
# used to locate existing results for the SKIP_EXISTING check below.
WORLD_TYPE="${WORLD_TYPE:-cafe}"
# SKIP_EXISTING=1 (default) skips any norm/model whose results directory
# already holds a metrics_summary_<norm>.json (i.e. run + eval both finished),
# so rerunning the script only runs what is still missing. Set 0 to rerun all.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# API keys (if any are needed) are loaded from .env by the Python entrypoint
# via force_load_repo_env(), which parses key=value lines safely. Do not
# `source .env` here: it is not a pure shell-assignment file and any malformed
# line would abort the run under `set -e`.

echo "=== START: $(date) ==="
echo "norm_types=$NORM_TYPES agent_type=$AGENT_TYPE provider=$PROVIDER models=$MODELS"

# Mirror run_experiment.py's model dir naming: '/' and ':' become '_'.
model_dir() { local m="${1//\//_}"; echo "${m//:/_}"; }

# One norm/model failing must not cancel the ones after it; collect and report.
failures=()

for NORM_TYPE in $NORM_TYPES; do
for MODEL in $MODELS; do
    echo
    echo "=== [$(date)] NORM: $NORM_TYPE | MODEL: $MODEL ==="

    # Skip norm/model combos that already finished (run + eval), so the script
    # can be rerun to fill in only the missing norms.
    RESULT_DIR="experiment/results/${AGENT_TYPE}/$(model_dir "$MODEL")/${WORLD_TYPE}/${NORM_TYPE}"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$RESULT_DIR/metrics_summary_${NORM_TYPE}.json" ]; then
        echo "[SKIP] $RESULT_DIR already has metrics_summary_${NORM_TYPE}.json"
        continue
    fi

    # Output defaults to experiment/results/<agent_type>/<model>/<world>/<norm_type>/
    # for LLM subjects (no --output_dir passed). Rather than rebuild that path
    # here (it depends on the world type from experiment_config.json), tee the
    # run and read back the raw_runs dir the run itself reports.
    RUN_LOG="$(mktemp)"
    if ! python -u experiment/run/run_experiment.py \
        --norm_type "$NORM_TYPE" \
        --agent_type "$AGENT_TYPE" \
        --provider "$PROVIDER" \
        --model "$MODEL" | tee "$RUN_LOG"; then
        echo "[FAIL] run_experiment.py failed for $NORM_TYPE/$MODEL; skipping its eval"
        failures+=("$NORM_TYPE/$MODEL (run)")
        rm -f "$RUN_LOG"
        continue
    fi

    RAW_RUNS="$(sed -n 's/^\[OK\] Raw run logs saved to: //p' "$RUN_LOG" | tail -1)"
    rm -f "$RUN_LOG"

    if [ -z "$RAW_RUNS" ] || [ ! -d "$RAW_RUNS" ]; then
        echo "[FAIL] could not locate raw_runs for $NORM_TYPE/$MODEL ('$RAW_RUNS'); skipping eval"
        failures+=("$NORM_TYPE/$MODEL (missing raw_runs)")
        continue
    fi

    echo "=== [$(date)] EVAL: $NORM_TYPE/$MODEL ($RAW_RUNS) ==="
    # Writes metrics_summary_<norm_type>.json next to raw_runs.
    if ! python -u experiment/evaluation/eval.py "$RAW_RUNS"; then
        echo "[FAIL] eval.py failed for $NORM_TYPE/$MODEL"
        failures+=("$NORM_TYPE/$MODEL (eval)")
    fi
done
done

echo
if [ ${#failures[@]} -gt 0 ]; then
    echo "=== FAILURES: ${failures[*]} ==="
fi
echo "=== END: $(date) ==="

# Deactivate the virtual environment when done
deactivate

if [ ${#failures[@]} -gt 0 ]; then
    exit 1
fi
