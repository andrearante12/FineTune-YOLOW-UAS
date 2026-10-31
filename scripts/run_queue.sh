#!/usr/bin/env bash
# Run train_sweep.py configs one after another inside the Thor container.
# Each non-comment line of the queue file is either
#   - the argument list for one training run (must include --name); its log goes to
#     runs/sweep/<name>.log and, when it finishes, scripts/eval_run.sh evaluates its
#     checkpoints into eval_results/sweep_<name>/ (log: runs/sweep/<name>.eval.log); or
#   - "! <shell command>", run as-is from the repo root on the host (e.g. a benchmark
#     that must not overlap with training).
# A training run whose log is silent for STALL_SECONDS (default 20 min) is killed and
# retried with --resume from its last.pt, up to 3 attempts in total.
# The queue file is read once at launch; to drop a pending run later, create
# runs/sweep/SKIP_<name> before the queue reaches it.
# Finished runs (completion line in their log) and "!" steps that already succeeded
# (marker in runs/sweep/.queue_done) are skipped, so the queue can be restarted.
# Usage: scripts/run_queue.sh configs/thor_sweeps/paper_fixed_optim.txt
set -uo pipefail

QUEUE="$1"
STALL_SECONDS="${STALL_SECONDS:-1200}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DONE="$REPO/runs/sweep/.queue_done"
mkdir -p "$REPO/runs/sweep"
touch "$DONE"
cd "$REPO"

grep -vE '^\s*(#|$)' "$QUEUE" | while read -r line; do
    if [[ "$line" == '!'* ]]; then
        cmd="${line#!}"
        if grep -qxF "$cmd" "$DONE"; then
            echo "[queue] already done, skipping: $cmd"; continue
        fi
        echo "[queue] $(date '+%F %T') run: $cmd"
        if bash -c "$cmd" < /dev/null; then
            echo "$cmd" >> "$DONE"
            echo "[queue] $(date '+%F %T') ok: $cmd"
        else
            echo "[queue] $(date '+%F %T') FAILED (exit $?): $cmd"
        fi
        continue
    fi

    name="$(sed -nE 's/.*--name[ =]([^ ]+).*/\1/p' <<<"$line")"
    if [[ -z "$name" ]]; then
        echo "[queue] skipping line without --name: $line"; continue
    fi
    log="runs/sweep/$name.log"
    if [[ -e "runs/sweep/SKIP_$name" ]]; then
        echo "[queue] $(date '+%F %T') runs/sweep/SKIP_$name present, skipping $name"
        continue
    fi
    if grep -q "epochs completed" "$log" 2>/dev/null; then
        echo "[queue] $name already trained, skipping training"
    else
        for attempt in 1 2 3; do
            extra=""
            if [[ $attempt -gt 1 ]]; then
                extra="--resume"
                sleep 30  # let docker remove the killed container and free its name
            fi
            echo "[queue] $(date '+%F %T') start $name (attempt $attempt) $extra"
            # shellcheck disable=SC2086
            THOR_NAME="train_$name" scripts/thor_run.sh python3 scripts/train_sweep.py \
                $line $extra >> "$log" 2>&1 < /dev/null &
            pid=$!
            # Watchdog: a driver-side allocation stall leaves the run alive but silent.
            while kill -0 "$pid" 2>/dev/null; do
                sleep 60
                age=$(( $(date +%s) - $(stat -c %Y "$log") ))
                if (( age > STALL_SECONDS )); then
                    echo "[queue] $(date '+%F %T') $name STALLED (log silent ${age}s), killing"
                    sg docker -c "docker kill train_$name" >/dev/null 2>&1 || \
                        docker kill "train_$name" >/dev/null 2>&1
                    break
                fi
            done
            wait "$pid"; rc=$?
            echo "[queue] $(date '+%F %T') end $name attempt $attempt (exit $rc)"
            grep -q "epochs completed" "$log" && break
            [[ -f "runs/sweep/$name/weights/last.pt" ]] || break  # nothing to resume
        done
    fi
    if [[ -f "eval_results/sweep_$name/sweep_results.json" ]]; then
        echo "[queue] $name already evaluated"
    elif grep -q "epochs completed" "$log" 2>/dev/null; then
        echo "[queue] $(date '+%F %T') eval $name"
        scripts/eval_run.sh "$name" > "runs/sweep/$name.eval.log" 2>&1 < /dev/null
        echo "[queue] $(date '+%F %T') eval done $name (exit $?)"
    fi
done
