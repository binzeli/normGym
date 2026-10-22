#!/usr/bin/env bash
# Launch the normGym human-experiment server and (optionally) a public tunnel so
# you can send the link to remote participants.
#
# Usage:
#   ./experiment/human/run.sh --norm_types straight_line movement_clockwise_move occupation_seat_in_order
#   SHARE=1 ./experiment/human/run.sh --norm_types straight_line movement_clockwise_move  # also open a public link
#
# Any extra args are passed straight to app.py (see: python experiment/human/app.py --help)
set -euo pipefail

PORT="${PORT:-5001}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

cd "$ROOT"

echo "[run] starting normGym experiment server on port $PORT"
python experiment/human/app.py --host 0.0.0.0 --port "$PORT" "$@" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Give the server a moment to bind.
sleep 2

if [[ "${SHARE:-0}" == "1" ]]; then
  if command -v cloudflared >/dev/null 2>&1; then
    echo "[run] opening public link via cloudflared (look for the https://*.trycloudflare.com URL below)…"
    cloudflared tunnel --url "http://localhost:$PORT"
  elif command -v ngrok >/dev/null 2>&1; then
    echo "[run] opening public link via ngrok…"
    ngrok http "$PORT"
  else
    echo "[run] SHARE=1 but neither 'cloudflared' nor 'ngrok' is installed."
    echo "[run] Install one (see README) or open the tunnel manually in another terminal:"
    echo "        cloudflared tunnel --url http://localhost:$PORT"
    wait $SERVER_PID
  fi
else
  echo "[run] local only — open http://localhost:$PORT"
  echo "[run] (set SHARE=1 to also open a public link for remote participants)"
  wait $SERVER_PID
fi
