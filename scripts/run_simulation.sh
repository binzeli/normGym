#!/usr/bin/env bash
# Quick-launch script for normGym simulations.
# Rule-based agents only (no LLM). Each run plays the same scene twice, with
# identical arrivals: once with norm-blind agents (no_norm_run<k>/) and once
# with agents that follow the norm (with_<norm>_run<k>/), each saved as
# step_*.png frames and simulation.mp4. Edit the config section below or set
# NORM_TYPE / WORLD_TYPE, then run from anywhere:
#   NORM_TYPE=xcell_forbidden bash scripts/run_simulation.sh

set -euo pipefail
cd "$(dirname "$0")/.."

WORLD_TYPE="${WORLD_TYPE:-cafe}"             # cafe | movie_theater | bank | pharmacy | grocery_store | clothing_store | library | post_office | hospital | barbershop | all
NORM_TYPE="${NORM_TYPE:-edge_left_line}"   # any "name" in src/norms/norm_descriptions.json, e.g. zigzag_left1_line | movement_counterclockwise_move
NUM_AGENTS=8
NUM_SIMULATIONS=1

# Display options
ISOMETRIC=false              # true to use 2.5D isometric rendering
INTERPOLATION_STEPS=4      # intermediate frames between steps for smooth rendering (0 = disabled)
# Output
OUTPUT_DIR="outputs_test"        # base output directory
FIGURES_SUFFIX=""           # optional suffix for figures subdir

# Use the project venv if present; otherwise fall back to whatever python is on PATH
if [[ -f ".venv/bin/python" ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

ARGS=(
    --world_type "$WORLD_TYPE"
    --norm_type "$NORM_TYPE"
    --num_agents "$NUM_AGENTS"
    --num_simulations "$NUM_SIMULATIONS"
    --output_dir "$OUTPUT_DIR"
)

[[ -n "$FIGURES_SUFFIX" ]] && ARGS+=(--figures_suffix "$FIGURES_SUFFIX")
if [[ "$ISOMETRIC" == "true" ]]; then
    ARGS+=(--isometric)
fi
[[ "$INTERPOLATION_STEPS" -gt 0 ]] && ARGS+=(--interpolation_steps "$INTERPOLATION_STEPS")

# Runner script
SCRIPT="runner/run_simulation.py"

# Build list of worlds to run
if [[ "$WORLD_TYPE" == "all" ]]; then
    WORLDS=(cafe movie_theater bank pharmacy grocery_store clothing_store library post_office hospital barbershop)
else
    WORLDS=("$WORLD_TYPE")
fi

for WORLD in "${WORLDS[@]}"; do
    # Replace --world_type in ARGS
    RUN_ARGS=()
    SKIP_NEXT=false
    for i in "${!ARGS[@]}"; do
        if $SKIP_NEXT; then
            SKIP_NEXT=false
            continue
        fi
        if [[ "${ARGS[$i]}" == "--world_type" ]]; then
            RUN_ARGS+=(--world_type "$WORLD")
            SKIP_NEXT=true
        else
            RUN_ARGS+=("${ARGS[$i]}")
        fi
    done

    # Separate output paths per render mode so frames don't overwrite each other
    if [[ "$ISOMETRIC" == "true" ]]; then
        RENDER_DIR="$OUTPUT_DIR/$WORLD/iso/$NORM_TYPE"
    else
        RENDER_DIR="$OUTPUT_DIR/$WORLD/flat/$NORM_TYPE"
    fi

    # Override --output_dir in RUN_ARGS with render-specific path
    FINAL_ARGS=()
    SKIP_NEXT=false
    for i in "${!RUN_ARGS[@]}"; do
        if $SKIP_NEXT; then
            SKIP_NEXT=false
            continue
        fi
        if [[ "${RUN_ARGS[$i]}" == "--output_dir" ]]; then
            FINAL_ARGS+=(--output_dir "$RENDER_DIR")
            SKIP_NEXT=true
        else
            FINAL_ARGS+=("${RUN_ARGS[$i]}")
        fi
    done

    echo "=== normGym Simulation ==="
    echo "  Runner:     $SCRIPT"
    echo "  World:      $WORLD"
    echo "  Norm:       $NORM_TYPE"
    echo "  Agents:     $NUM_AGENTS"
    echo "  Sims:       $NUM_SIMULATIONS"
    if [[ "$ISOMETRIC" == "true" ]]; then
        echo "  Render:     isometric"
    else
        echo "  Render:     flat"
    fi
    echo "  Output:     $RENDER_DIR/"
    echo "=========================="
    echo

    mkdir -p "$RENDER_DIR/simulation_log"
    $PYTHON "$SCRIPT" "${FINAL_ARGS[@]}" 2>&1 | tee "$RENDER_DIR/simulation_log/simulation_output_${NUM_AGENTS}agents.txt"
done