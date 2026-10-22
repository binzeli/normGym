#!/bin/bash

# Oracle-planner baselines for every norm, one norm after another: each runs
# both subject kinds back-to-back and evaluates them.
#
#   oracle_with_norm     planner knows the norm and plans around it   -> reward ceiling
#   oracle_without_norm  planner ignores the norm, shortest path      -> norm-blind floor
#
# Together they bracket what a subject can achieve on each norm. Each kind
# runs the same test episodes as the benchmark (3 same-layout + 3 new-layout,
# no learning stage: the oracle has nothing to learn).
#
# Results land in experiment/results/<subject_kind>/<world>/<norm>/, beside the
# LLM trees. No GPU or API key is needed: the oracle plans locally.
#
# All 50 norms:   bash scripts/run_oracle.sh
# Some norms:     NORM_TYPES="edge_left_line xcell_forbidden" bash scripts/run_oracle.sh

set -uo pipefail

cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

# All 50 norms, in src/norms/norm_descriptions.json order (what
# experiment/run/common.py norm_choices() returns).
ALL_NORMS=(
    zigzag_right2_line
    zigzag_left1_line
    bend_right1_line
    bend_left2_line
    edge_left_line
    zigzag_right1_line
    zigzag_left2_line
    bend_left1_line
    bend_right2_line
    edge_right_line
    square_right_line
    square_left_line
    staircase_right_line
    staircase_left_line
    lshape_right_line
    lshape_left_line
    movement_clockwise_move
    movement_counterclockwise_move
    movement_no_odd_turn
    movement_no_even_turn
    movement_no_turn_mod3_0
    movement_no_turn_mod3_1
    movement_no_turn_mod3_2
    movement_move1_wait1
    movement_move2_wait1
    movement_move3_wait1
    xcell_must_visit
    xcell_forbidden
    occupation_region_capacity_1
    occupation_region_capacity_2
    occupation_region_single_class_only
    occupation_door_all_exit_A
    occupation_door_A_forbidden
    occupation_door_class_specific
    occupation_door_all_exit_B
    occupation_door_all_exit_C
    occupation_door_B_forbidden
    occupation_door_C_forbidden
    occupation_seat_class_specific
    occupation_seat_bottom_left_only
    occupation_seat_no_bottom_left
    occupation_seat_in_order
    occupation_seat_top_left_only
    occupation_seat_bottom_right_only
    occupation_seat_no_bottom_right
    occupation_door_class_specific_BCA
    occupation_door_class_specific_CAB
    occupation_seat_class_specific_rot1
    occupation_seat_class_specific_rot2
    occupation_seat_in_order_right
)

# Default: every norm. NORM_TYPES overrides with an explicit list.
NORM_TYPES="${NORM_TYPES:-${ALL_NORMS[*]}}"
AGENT_TYPE="${AGENT_TYPE:-both}"

source .venv/bin/activate

echo "=== START: $(date) ==="
echo "norm_types=$NORM_TYPES agent_type=$AGENT_TYPE"

failures=()

for NORM_TYPE in $NORM_TYPES; do
    echo
    echo "=== [$(date)] ORACLE NORM: $NORM_TYPE (agent_type=$AGENT_TYPE) ==="

    # run_oracle.py clears and rewrites raw_runs/ for each subject kind it runs,
    # and prints one '[OK] Raw run logs saved to: <dir>' line per kind.
    RUN_LOG="$(mktemp)"
    if ! python -u experiment/run/run_oracle.py \
        --norm_type "$NORM_TYPE" \
        --agent_type "$AGENT_TYPE" | tee "$RUN_LOG"; then
        echo "[FAIL] run_oracle.py failed for $NORM_TYPE; skipping its eval"
        failures+=("$NORM_TYPE (run)")
        rm -f "$RUN_LOG"
        continue
    fi

    # One raw_runs dir per subject kind (two when --agent_type both).
    mapfile -t RAW_RUNS < <(sed -n 's/^\[OK\] Raw run logs saved to: //p' "$RUN_LOG")
    rm -f "$RUN_LOG"

    if [ ${#RAW_RUNS[@]} -eq 0 ]; then
        echo "[FAIL] could not locate raw_runs for $NORM_TYPE; skipping eval"
        failures+=("$NORM_TYPE (missing raw_runs)")
        continue
    fi

    for DIR in "${RAW_RUNS[@]}"; do
        if [ ! -d "$DIR" ]; then
            echo "[FAIL] raw_runs dir does not exist: $DIR"
            failures+=("$NORM_TYPE (missing $DIR)")
            continue
        fi
        echo "=== [$(date)] EVAL: $NORM_TYPE ($DIR) ==="
        # Writes metrics_summary_<norm_type>.json next to raw_runs.
        if ! python -u experiment/evaluation/eval.py "$DIR"; then
            echo "[FAIL] eval.py failed for $NORM_TYPE ($DIR)"
            failures+=("$NORM_TYPE (eval $DIR)")
        fi
    done
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
