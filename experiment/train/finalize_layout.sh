#!/usr/bin/env bash
# One-off: finish the per-norm-group layout (2026-08-28) once the three
# queuing jobs that still use the pre-group paths have ended.
#
# While jobs 115799 (v1), 119037 (v2) and 125557 (v3) run, their data /
# output dirs stay at the old paths and the new layout points at them with
# symlinks. This script replaces each symlink with the real directory.
# Safe to re-run; refuses to touch anything a running job still uses.
#
#   bash experiment/train/finalize_layout.sh          # dry run (prints)
#   bash experiment/train/finalize_layout.sh --apply
set -euo pipefail
cd "$(dirname "$0")"          # experiment/train
APPLY="${1:-}"

running=" $(squeue -h -u "$USER" -o "%i" 2>/dev/null | tr '\n' ' ') "

# link (new path)                                   real dir (old path)                 jobs that still use it
moves=(
  "data/queuing/snapshots_train                     data/snapshots_line11               115799,119037"
  "data/queuing/violation_judge/train               data/violation_judge/line11         125557"
  "checkpoints/sft/queuing/legacy_4norm             checkpoints/sft_4b                  115799,119037"
  "checkpoints/v1_action_rloo/queuing/line11         checkpoints/step_rloo_line11_4b     115799"
  "checkpoints/v2_belief_rloo/queuing/line11        checkpoints/v2_belief_rloo/line11   119037"
  "checkpoints/v3_violation_judge/queuing/line11    checkpoints/v3_violation_judge/line11 125557"
)
for m in "${moves[@]}"; do
    set -- $m; link=$1; real=$2; jobs=$3
    busy=""
    for j in ${jobs//,/ }; do [[ "$running" == *" $j "* ]] && busy="$busy $j"; done
    if [ -n "$busy" ]; then
        echo "skip (job$busy still running): $link"; continue
    fi
    if [ -L "$link" ]; then
        echo "mv $real -> $link"
        if [ "$APPLY" = "--apply" ]; then rm "$link"; mv "$real" "$link"; fi
    elif [ -d "$link" ]; then
        echo "already final: $link"
    else
        echo "missing: $link (and $real)"; exit 1
    fi
done
if [ "$APPLY" = "--apply" ]; then
    # test-eval jobs 169768/169769 wrote through this symlink (results moved into train/ on 2026-08-28)
    [ -L ../results_test ] && [ -z "$(squeue -h -j 169768,169769 2>/dev/null)" ] && rm ../results_test && echo "removed experiment/results_test symlink"
    if [ -z "$(squeue -h -j 119037 2>/dev/null)" ]; then
        # code-layout compatibility symlinks for job 119037 (2026-08-28 move of v1/v2/v3 under vigal/)
        for l in vigal/step_rloo_train.py vigal/harvest_snapshots.py vigal/snapshot_verifier.py v2_belief_rloo; do
            [ -L "$l" ] && rm "$l" && echo "removed symlink $l"
        done
    fi
    rmdir data/violation_judge 2>/dev/null || true
    echo "done. Remaining old-path dirs (should be none once every job has ended):"; ls -d data/snapshots_* data/violation_judge checkpoints/sft_4b checkpoints/step_rloo_line11_4b checkpoints/v2_belief_rloo/line11 checkpoints/v3_violation_judge/line11 2>/dev/null || true
else
    echo "(dry run — add --apply)"
fi
