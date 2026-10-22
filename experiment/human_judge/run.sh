#!/usr/bin/env bash
# Launch the LLM-judge validation study server and (optionally) a public tunnel.
#
# Usage:
#   ./experiment/human_judge/run.sh                 # local only, http://localhost:5002
#   SHARE=1 ./experiment/human_judge/run.sh         # also open a public cloudflared/ngrok link
#   ADMIN_TOKEN=secret SHARE=1 ./experiment/human_judge/run.sh   # wave 2 interface
#   DESIGN=experiment/human_judge/study_design_wave2.json ADMIN_TOKEN=secret SHARE=1 ./experiment/human_judge/run.sh
#   MODE=two ADMIN_TOKEN=secret SHARE=1 ./experiment/human_judge/run.sh     # wave 1 interface
#
# Any extra args are passed to app.py (e.g. --port 5003).
# On a Mac this re-runs itself under `caffeinate` so the machine will not sleep
# while the study is live (CAFFEINATE=0 to skip).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PORT="${PORT:-5002}"

if [[ "${SHARE:-0}" == "1" && "${CAFFEINATE:-1}" == "1" && -z "${NORMGYM_CAFFEINATED:-}" ]] \
   && command -v caffeinate >/dev/null 2>&1; then
  export NORMGYM_CAFFEINATED=1
  echo "[judge] keeping this machine awake until you stop the study (CAFFEINATE=0 to skip)"
  exec caffeinate -is "$HERE/run.sh" "$@"
fi

cd "$ROOT"
# Which design to serve (DESIGN=... overrides; e.g. the wave-2 follow-up
# DESIGN=experiment/human_judge/study_design_wave2.json). Refuse to start on a
# design that fails its own checks.
DESIGN="${DESIGN:-$HERE/study_design.json}"
python "$HERE/make_study_design.py" --check --quiet --out "$DESIGN"

# Interface mode: MODE=two (wave 1: own rating, then judge + agreement) or
# MODE=agree (wave 2, default: judge shown up front, agreement only).
MODE="${MODE:-agree}"

# Prolific completion code shown on the final page (COMPLETION_CODE=... overrides).
export COMPLETION_CODE="${COMPLETION_CODE:-XXXXXXXX}"
echo "[judge] starting the LLM-judge study server on port $PORT (completion code $COMPLETION_CODE)"
echo "[judge] design: $DESIGN   mode: $MODE"
python "$HERE/app.py" --host 0.0.0.0 --port "$PORT" --design_file "$DESIGN" --mode "$MODE" "$@" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
sleep 2

if [[ "${SHARE:-0}" == "1" ]]; then
  if command -v cloudflared >/dev/null 2>&1; then
    echo "[judge] opening public link via cloudflared (look for the https://*.trycloudflare.com URL below)…"
    cloudflared tunnel --url "http://localhost:$PORT"
  elif command -v ngrok >/dev/null 2>&1; then
    echo "[judge] opening public link via ngrok…"
    ngrok http "$PORT"
  else
    echo "[judge] SHARE=1 but neither 'cloudflared' nor 'ngrok' is installed; open a tunnel manually:"
    echo "        cloudflared tunnel --url http://localhost:$PORT"
    wait $SERVER_PID
  fi
else
  echo "[judge] local only — open http://localhost:$PORT   (SHARE=1 for a public link)"
  wait $SERVER_PID
fi
