#!/usr/bin/env bash
# WiSE-FT across the three 5-epoch seed runs (paper_e5, paper_e5_s1, paper_e5_s2):
# interpolate each run's final weights with the base model, evaluate every alpha,
# then pick alpha by VisDrone mAP only (select_wise_alpha.py) and report OVB
# mean +- std across seeds. Run from run_queue.sh as a "!" step after the seeds.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

results=()
for run in paper_e5 paper_e5_s1 paper_e5_s2; do
    ft="runs/sweep/$run/weights/last.pt"
    [[ -f "$ft" ]] || { echo "[wise] $ft missing, skipping $run"; continue; }
    out="eval_results/sweep_wise_$run/sweep_results.json"
    if [[ ! -f "$out" ]]; then
        (source ~/venvs/yoloworld-trt/bin/activate &&
         python scripts/wise_ft.py --finetuned-weights "$ft" --output "runs/wise_ft/$run")
        scripts/thor_run.sh python3 scripts/sweep_checkpoints.py \
            --checkpoint runs/wise_ft/"$run"/wise_last_a*.pt \
            --output "eval_results/sweep_wise_$run"
    fi
    results+=("$out")
done

python3 scripts/select_wise_alpha.py "${results[@]}" \
    --output eval_results/wise_seeds_summary.json
