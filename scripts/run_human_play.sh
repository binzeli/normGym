#!/usr/bin/env bash
# Play normGym yourself in the browser: drive one character around a world under
# a norm, like the human study's learning round, with no step limit. When your
# character reaches its goal and leaves, you take over a new one and play goes
# on. Runs runner/human_play.py, which reuses the human study's game page and
# engine; nothing is logged.
#
# Edit the config section below, then run from anywhere:
#   bash scripts/run_human_play.sh
# and open http://localhost:5001 (or $PORT) in your browser.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

# Use the project venv if present; otherwise fall back to whatever python is on PATH
if [[ -f ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

NORM_TYPE="${NORM_TYPE:-edge_left_line}"  # any "name" in src/norms/norm_descriptions.json, or straight_line
WORLD_TYPE="${WORLD_TYPE:-cafe}"           # cafe | movie_theater | bank | pharmacy | grocery_store | clothing_store | library | post_office | hospital | barbershop
MAP_TYPE="default"          # default | navigation | affordance
CAPACITY=""                 # max characters on the map; empty = the norm's benchmark capacity
ISOMETRIC=false             # true to use 2.5D isometric rendering
INTERPOLATION_STEPS=4       # intermediate frames between steps for smooth motion (0 = disabled)

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5001}"

ARGS=(
  --norm_type "$NORM_TYPE"
  --world_type "$WORLD_TYPE"
  --map_type "$MAP_TYPE"
  --interpolation_steps "$INTERPOLATION_STEPS"
  --host "$HOST"
  --port "$PORT"
)
[[ -n "$CAPACITY" ]] && ARGS+=(--capacity "$CAPACITY")
[[ "$ISOMETRIC" == "true" ]] && ARGS+=(--isometric)

echo "=========================================="
echo "normGym free play"
echo "Norm: $NORM_TYPE  |  World: $WORLD_TYPE"
echo "Open http://localhost:${PORT} in your browser"
echo "=========================================="

exec "$PYTHON" runner/human_play.py "${ARGS[@]}"
