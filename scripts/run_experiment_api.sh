#!/bin/bash
# Runs the benchmark (run_experiment.py + eval.py) for API models via
# OpenRouter, one norm after another, every model per norm. Inference is
# remote, so no GPU is needed. For local Hugging Face models (GPU) use
# run_experiment_hf.sh instead. Settings are read from the environment, e.g.
#   NORM_TYPES=edge_left_line MODELS="openai/gpt-5.6-sol" bash scripts/run_experiment_api.sh

set -euo pipefail

# Run from the repo root
cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

# Norms run one after another in this order; each norm runs every model below.
# NORM_TYPE (singular) is still honoured for backwards compatibility.

# Default: all 50 norms, in src/norms/norm_descriptions.json norm_id order.
ALL_NORMS="zigzag_right2_line zigzag_left1_line bend_right1_line \
bend_left2_line edge_left_line zigzag_right1_line \
zigzag_left2_line bend_left1_line bend_right2_line \
edge_right_line square_right_line square_left_line \
staircase_right_line staircase_left_line lshape_right_line \
lshape_left_line movement_clockwise_move movement_counterclockwise_move \
movement_no_odd_turn movement_no_even_turn movement_no_turn_mod3_0 \
movement_no_turn_mod3_1 movement_no_turn_mod3_2 movement_move1_wait1 \
movement_move2_wait1 movement_move3_wait1 xcell_must_visit \
xcell_forbidden occupation_region_capacity_1 occupation_region_capacity_2 \
occupation_region_single_class_only occupation_door_all_exit_A occupation_door_A_forbidden \
occupation_door_class_specific occupation_door_all_exit_B occupation_door_all_exit_C \
occupation_door_B_forbidden occupation_door_C_forbidden occupation_seat_class_specific \
occupation_seat_bottom_left_only occupation_seat_no_bottom_left occupation_seat_in_order \
occupation_seat_top_left_only occupation_seat_bottom_right_only occupation_seat_no_bottom_right \
occupation_door_class_specific_BCA occupation_door_class_specific_CAB occupation_seat_class_specific_rot1 \
occupation_seat_class_specific_rot2 occupation_seat_in_order_right"

NORM_TYPES="${NORM_TYPES:-${NORM_TYPE:-$ALL_NORMS}}"
AGENT_TYPE="${AGENT_TYPE:-memory_llm}"
PROVIDER="${PROVIDER:-openrouter}"
# Models run one after another in this order, each followed by its own eval.
# Space-separated OpenRouter slugs, e.g. "openai/gpt-5.6-sol google/gemini-3.1-pro-preview".
MODELS="${MODELS:-google/gemini-3.1-pro-preview}"
# Must match world_type in experiment/run/experiment_config.json; only
# used to locate existing results for the SKIP_EXISTING check below.
WORLD_TYPE="${WORLD_TYPE:-cafe}"
# SKIP_EXISTING=1 (default) skips any norm/model whose results directory
# already holds a metrics_summary_<norm>.json (i.e. run + eval both finished),
# so rerunning the script only runs what is still missing. Set 0 to rerun all.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# API keys (OPENROUTER_API_KEY, etc.) are loaded from .env by the Python
# entrypoint via force_load_repo_env(), which parses key=value lines safely.
# Do not `source .env` here: it is not a pure shell-assignment file and any
# malformed line would abort the run under `set -e`.

# Activate virtual environment
source .venv/bin/activate

echo "=== START: $(date) ==="
echo "norm_types=$NORM_TYPES agent_type=$AGENT_TYPE provider=$PROVIDER models=$MODELS"

# Mirror run_experiment.py's model dir naming: '/' and ':' become '_'.
model_dir() { local m="${1//\//_}"; echo "${m//:/_}"; }

# Up-front plan: list which norm/model combos are already finished and which
# will actually run, so the log shows the work left before any API call.
if [ "$SKIP_EXISTING" = "1" ]; then
    for MODEL in $MODELS; do
        done_norms=(); todo_norms=()
        for NORM_TYPE in $NORM_TYPES; do
            RESULT_DIR="experiment/results/${AGENT_TYPE}/$(model_dir "$MODEL")/${WORLD_TYPE}/${NORM_TYPE}"
            if [ -f "$RESULT_DIR/metrics_summary_${NORM_TYPE}.json" ]; then
                done_norms+=("$NORM_TYPE")
            else
                todo_norms+=("$NORM_TYPE")
            fi
        done
        echo "=== PLAN for $MODEL: ${#done_norms[@]} done, ${#todo_norms[@]} to run ==="
        echo "    done: ${done_norms[*]:-<none>}"
        echo "    todo: ${todo_norms[*]:-<none>}"
    done
fi

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
    # Thinking models spend reasoning tokens out of the completion budget, so
    # cap their thinking and widen the budget; otherwise long thinks return
    # finish_reason=length with no JSON and the agent falls back to 'wait'.
    # Applied per model so other models keep exactly their previous requests.
    # Override from the environment with REASONING_EFFORT / REASONING_MAX_TOKENS /
    # MAX_COMPLETION_TOKENS (empty string = do not send).
    unset OPENROUTER_REASONING_EFFORT OPENROUTER_REASONING_MAX_TOKENS OPENROUTER_MAX_COMPLETION_TOKENS
    case "$MODEL" in
        google/gemini*)
            export OPENROUTER_REASONING_EFFORT="${REASONING_EFFORT-low}"
            export OPENROUTER_REASONING_MAX_TOKENS="${REASONING_MAX_TOKENS-}"
            export OPENROUTER_MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS-8192}"
            ;;
        *)
            export OPENROUTER_REASONING_EFFORT="${REASONING_EFFORT-}"
            export OPENROUTER_REASONING_MAX_TOKENS="${REASONING_MAX_TOKENS-}"
            export OPENROUTER_MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS-}"
            ;;
    esac
    echo "reasoning_effort='${OPENROUTER_REASONING_EFFORT}' reasoning_max_tokens='${OPENROUTER_REASONING_MAX_TOKENS}' max_completion_tokens='${OPENROUTER_MAX_COMPLETION_TOKENS}'"

    RUN_LOG="$(mktemp)"
    if ! python -u experiment/run/run_experiment.py \
        --norm_type "$NORM_TYPE" \
        --agent_type "$AGENT_TYPE" \
        --provider "$PROVIDER" \
        --model "$MODEL" | tee "$RUN_LOG"; then
        # --max_steps 50 \
        # --test_max_steps 3
        rc=${PIPESTATUS[0]}
        rm -f "$RUN_LOG"
        # Exit code 3 = provider account out of credits (EXIT_CODE_NO_CREDITS
        # in run_experiment.py). Every remaining norm/model would fail the same
        # way, so stop the whole run now instead of looping through them.
        if [ "$rc" -eq 3 ]; then
            echo "[FATAL] Out of LLM credits at $NORM_TYPE/$MODEL; aborting the whole run."
            echo "[FATAL] Add credits at https://openrouter.ai/settings/credits and rerun."
            if [ ${#failures[@]} -gt 0 ]; then
                echo "=== EARLIER FAILURES: ${failures[*]} ==="
            fi
            exit 3
        fi
        echo "[FAIL] run_experiment.py failed for $NORM_TYPE/$MODEL; skipping its eval"
        failures+=("$NORM_TYPE/$MODEL (run)")
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
