#!/usr/bin/env bash
# Evaluate a finished train_sweep.py run's per-epoch checkpoints with
# sweep_checkpoints.py (fixed OV evaluator, imgsz 1280) inside the Thor container.
#
# ultralytics names per-epoch files 0-based (epoch4.pt = weights after 5 epochs), so
# this links e<N>.pt -> epoch<N-1>.pt and evaluates those; results label points e<N>.
#
# Usage: scripts/eval_run.sh <run name> [epochs...]   (default: all epochs <= 10,
#        then every 5th, plus the last)
# Output: eval_results/sweep_<run name>/sweep_results.json
set -euo pipefail

NAME="$1"; shift
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WDIR="runs/sweep/$NAME/weights"
LINKS="runs/sweep/$NAME/eval_ckpts"
cd "$REPO"

n_epochs=$(find "$WDIR" -maxdepth 1 -name 'epoch*.pt' | wc -l)
if [[ "$n_epochs" -eq 0 ]]; then
    echo "[eval] no epoch*.pt in $WDIR"; exit 1
fi
if [[ $# -gt 0 ]]; then
    epochs=("$@")
else
    epochs=()
    for ((e = 1; e <= n_epochs; e++)); do
        if (( e <= 10 || e % 5 == 0 || e == n_epochs )); then epochs+=("$e"); fi
    done
fi

mkdir -p "$LINKS"
ckpts=()
for e in "${epochs[@]}"; do
    src="$WDIR/epoch$((e - 1)).pt"
    [[ -f "$src" ]] || { echo "[eval] missing $src, skipping"; continue; }
    ln -sfn "../weights/epoch$((e - 1)).pt" "$LINKS/e$e.pt"
    ckpts+=("$LINKS/e$e.pt")
done

echo "[eval] $NAME: ${#ckpts[@]} checkpoints (${epochs[*]})"
scripts/thor_run.sh python3 scripts/sweep_checkpoints.py \
    --checkpoint "${ckpts[@]}" --output "eval_results/sweep_$NAME"
