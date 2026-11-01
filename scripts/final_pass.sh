#!/usr/bin/env bash
# Final deliverables pass (after all training): standard benchmarks for the late models,
# clean checkpoint exports, TensorRT benchmark of the recommended models, annotated
# drone videos, the LVIS fixed-AP / YOLO-World-prompt cross-check, and paired-bootstrap
# CIs vs base. Steps are ordered by priority; each is independent, so a failure in one
# is logged and the rest still run. Run from run_queue.sh as a "!" step.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source ~/venvs/yoloworld-trt/bin/activate
PY="python -W ignore"
VIDEO="${VIDEO:-$HOME/vlm_on_jetson/data/videos/DJI_0162.MP4}"

step() { echo; echo "=== [final] $(date '+%F %T') $*"; }
alpha() {  # alpha* chosen (VisDrone-only rule) for the first run in a wise summary json
    python3 -c "import json,sys; r=json.load(open(sys.argv[1]))['rows'][0]; print(f\"{r['alpha']:.2f}\")" "$1" 2>/dev/null
}

A40=$(alpha eval_results/wise_lr1e-4_e40_bias_summary.json)
A10=$(alpha eval_results/wise_seeds_lr1e-4_bias_summary.json)
echo "[final] alpha*: lr1e-4_e40_bias=${A40:-n/a} lr1e-4_e10_bias=${A10:-n/a}"
WISE40="runs/wise_ft/lr1e-4_e40_bias/wise_last_a${A40}.pt"
WISE10="runs/wise_ft/lr1e-4_e10_bias/wise_last_a${A10}.pt"

# label=weights for every final candidate (missing files are skipped by each tool)
CANDIDATES=(
    base=yolov8m-worldv2.pt
    ft5_paper=5_epoch.pt
    wise5_a0.7=runs/wise_ft/wise_5_epoch_a0.70.pt
    wise5bias_a0.7=runs/wise_ft/bias_e5/wise_last_a0.70.pt
    lr1e-4_e10_bias=runs/sweep/lr1e-4_e10_bias/weights/last.pt
    wise_lr1e-4_e10_bias=$WISE10
    lr1e-4_e40_bias=runs/sweep/lr1e-4_e40_bias/weights/last.pt
    wise_lr1e-4_e40_bias=$WISE40
    bias_e40=runs/sweep/bias_e40/weights/last.pt
    wise40_a0.9=runs/wise_ft/paper_e40/wise_last_a0.90.pt
)

step "1. standard benchmarks (LVIS/COCO/ODinW) for late models (cached ones skipped)"
$PY scripts/evaluate_standard_ov.py --datasets lvis coco odinw_aerial_all odinw_aerial \
    --models "${CANDIDATES[@]}" || echo "[final] step 1 FAILED"

step "2. export clean checkpoints -> checkpoints/"
$PY scripts/export_checkpoints.py "${CANDIDATES[@]:1}" || echo "[final] step 2 FAILED"

step "3. TensorRT benchmark of recommended models (FP16, 640 + 1280, with accuracy)"
$PY scripts/benchmark_thor.py --accuracy --runtimes torch-fp16 trt-fp16 --imgsz 640 1280 \
    --iters 300 --output eval_results/thor_benchmark_final \
    --weights checkpoints/wise5_a0.7.pt checkpoints/wise5bias_a0.7.pt \
              checkpoints/lr1e-4_e10_bias.pt checkpoints/wise_lr1e-4_e40_bias.pt \
    || echo "[final] step 3 FAILED"

step "4. annotated videos ($VIDEO)"
$PY scripts/annotate_video.py --video "$VIDEO" --output deliverables/videos \
    --models base=yolov8m-worldv2.pt paper_5ep=5_epoch.pt \
             wise_a0.7=checkpoints/wise5_a0.7.pt lr1e-4_bias=checkpoints/lr1e-4_e10_bias.pt \
    || echo "[final] step 4 FAILED"

step "5. LVIS cross-check: fixed AP + YOLO-World prompts (base, paper 5ep, WiSE)"
$PY scripts/evaluate_standard_ov.py --datasets lvis --fixed-ap --max-det 1000 \
    --prompt-file data/texts/lvis_v1_class_texts.json --tag _fixedap_ywprompts \
    --models base=yolov8m-worldv2.pt ft5_paper=5_epoch.pt \
             wise5_a0.7=runs/wise_ft/wise_5_epoch_a0.70.pt \
    || echo "[final] step 5 FAILED"

step "6. paired bootstrap (OVB) vs base for the recommended models"
scripts/thor_run.sh python3 scripts/evaluate_open_vocab.py \
    --finetuned-weights checkpoints/wise5bias_a0.7.pt checkpoints/lr1e-4_e10_bias.pt \
                        checkpoints/wise_lr1e-4_e40_bias.pt \
    --paired-bootstrap 1000 --output eval_results/ov_paired_bootstrap_final \
    || echo "[final] step 6 FAILED"

step "done"
