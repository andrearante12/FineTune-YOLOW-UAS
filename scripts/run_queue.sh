#!/usr/bin/env bash
# Run train_sweep.py configs one after another inside the Thor container.
# Each non-comment line of the queue file is the argument list for one run and
# must include --name; logs go to runs/sweep/<name>.log. Runs whose
# weights/last.pt already exists are skipped, so the queue can be restarted.
# Usage: scripts/run_queue.sh configs/thor_sweeps/paper_fixed_optim.txt
set -uo pipefail

QUEUE="$1"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/runs/sweep"

grep -vE '^\s*(#|$)' "$QUEUE" | while read -r line; do
    name="$(sed -nE 's/.*--name[ =]([^ ]+).*/\1/p' <<<"$line")"
    if [[ -z "$name" ]]; then
        echo "[queue] skipping line without --name: $line"; continue
    fi
    if [[ -f "$REPO/runs/sweep/$name/weights/last.pt" ]] && \
       grep -q "epochs completed" "$REPO/runs/sweep/$name.log" 2>/dev/null; then
        echo "[queue] $name already finished, skipping"; continue
    fi
    echo "[queue] $(date '+%F %T') start $name"
    # shellcheck disable=SC2086
    "$REPO/scripts/thor_run.sh" python3 scripts/train_sweep.py $line \
        > "$REPO/runs/sweep/$name.log" 2>&1 < /dev/null
    echo "[queue] $(date '+%F %T') end $name (exit $?)"
done
