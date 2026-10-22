#!/usr/bin/env bash
# Launch THE REAL STUDY: the 25-norm validation set (half of each category, see
# norms_25.txt), 25 participants, 2 norms each.
#
# The assignment is the fixed design in study_design_25.json (built and verified
# by make_study_design.py):
#   * every norm is played by exactly 2 participants — once as scenario 1 and
#     once as scenario 2 (order counterbalanced per norm);
#   * a participant's 2 norms come from DIFFERENT categories (queue / region /
#     trajectory / allocation — the paper's four families);
#   * 25 participants fill the design exactly; a 26th wraps back to slot 1.
# Tutorial/practice norm is always straight_line (not one of the 25).
#
# The first wave (2026-09-12/13) ran the full 50-norm design, study_design.json;
# pass DESIGN=experiment/human/study_design.json to run that one instead.
#
# Usage:
#   ./experiment/human/run_study.sh            # server + public cloudflared link
#   SHARE=0 ./experiment/human/run_study.sh    # local only (http://localhost:5001)
#   CAFFEINATE=0 ./experiment/human/run_study.sh   # don't hold the Mac awake
#
# On a Mac this re-runs itself under `caffeinate` so the machine will not sleep
# while the study is live (sleeping kills the tunnel and the participants' link).
# Closing the lid on battery still sleeps — leave the lid open, or plug in.
#
# Extra args are passed through to app.py, e.g.:
#   ./experiment/human/run_study.sh --port 5002
#
# Participant -> slot assignment is handed out in ARRIVAL ORDER (lowest free
# slot) and persisted to experiment/human/data/_assignments.json, so restarting
# this script continues the design instead of restarting it. To see who has
# been assigned what, and which slots are still free:
#   cat experiment/human/data/_assignments.json
#   http://localhost:5001/admin/assignments?token=$ADMIN_TOKEN
# A dropout who will not finish should have their slot freed so the next
# arrival takes it and the design still fills:
#   http://localhost:5001/admin/release?token=$ADMIN_TOKEN&pid=PARTICIPANT_ID
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep the Mac awake for the whole session — if the laptop sleeps, the tunnel dies
# and the participants' link goes dead (and cloudflared hands out a NEW url on the
# next start, so the old link is gone for good). Re-exec ourselves under
# `caffeinate` once; the guard var keeps that from looping, and CAFFEINATE=0 skips
# it. Non-macOS hosts have no `caffeinate`, so this is simply a no-op there.
if [[ "${CAFFEINATE:-1}" == "1" && -z "${NORMGYM_CAFFEINATED:-}" ]] \
   && command -v caffeinate >/dev/null 2>&1; then
  export NORMGYM_CAFFEINATED=1
  echo "[study] keeping this machine awake until you stop the study (CAFFEINATE=0 to skip)"
  exec caffeinate -is "$HERE/run_study.sh" "$@"
fi

# Which fixed design to serve (DESIGN=... overrides). Refuse to start on a
# design file that fails its own checks.
DESIGN="${DESIGN:-$HERE/study_design_25.json}"
python "$HERE/make_study_design.py" --check --out "$DESIGN" >/dev/null

# Shown to participants on the final thank-you page (Prolific completion code).
export COMPLETION_CODE="${COMPLETION_CODE:-XXXXXXXX}"

SHARE="${SHARE:-1}" exec "$HERE/run.sh" \
  --assignment design \
  --design_file "$DESIGN" \
  --tutorial_norm straight_line \
  "$@"
