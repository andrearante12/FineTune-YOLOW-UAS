# Experiments index (Jetson Thor, Oct 30 – Nov 23 2026)

All experiments for the final manuscript (due 2026-12-01) and the journal extension, in one place.
Manuscript package (story, all paper numbers, LaTeX edits): the other files in `docs/manuscript/`. The chronological log with every per-epoch table,
incident and decision is `docs/THOR_PROGRESS.md` (committed). Raw metrics: `eval_results/`.

## Conventions

- **VisDrone mAP**: COCO mAP@[.5:.95] on the 548-image VisDrone-DET val set, imgsz 1280.
- **OVB**: 195-image open-vocab benchmark (own UAS footage), 8 classes, box counts:
  pedestrian 571, backpack 106, orange circle landing pad 104, foreign object on orange circle
  landing pad 91, dog 57, soccer ball 37, foreign object on driveway 20, car 1.
- **OVB novel**: the 6 classes not in VisDrone (excludes car, pedestrian). **Headline OV metric.**
  All-class OVB is fragile because "car" has a single box (it can swing all-class mAP by ±0.056).
- **LVIS / COCO / ODinW**: zero-shot, imgsz 640, `scripts/evaluate_standard_ov.py`. Quote LVIS as
  **% of base**: our absolute LVIS is below YOLO-World's published numbers (ultralytics port,
  one prompt string per class, standard AP).
- Epoch labels = epochs completed (ultralytics `epochN.pt` is 0-based; `eval_run.sh` relinks).
- Run names: `paper_*` = paper recipe (AdamW 7.14e-4); `lrX_eN` = AdamW lr X for N epochs;
  `_bias` = pretrained contrastive-head biases restored (`--restore-head-bias`);
  `freeze10` = backbone frozen, `freeze22` = only the detection head trains;
  `_s1/_s2` = seeds 1/2 (no suffix = seed 0); `wise*` = WiSE-FT weight average with base.

## Common training settings (unless the name says otherwise)

YOLOv8m-WorldV2 init, ultralytics 8.4.194 (`yoloworld-thor` container via `scripts/thor_run.sh`),
single Thor GPU, batch 24, imgsz 1280, AdamW (β1 0.9), lrf 0.01 linear decay, weight decay 0.05,
warmup 3 epochs (warmup_bias_lr 0), AMP, mosaic on with `close_mosaic=10` (so 5/10-epoch runs never
use mosaic; 40-epoch runs use it for epochs 1–30), workers 4, `freeze: None` (backbone + neck + head
train). CLIP text encoder frozen (cached text embeddings). Args per run: `runs/sweep/<name>/args.yaml`.

## Bugs found in the original paper pipeline

1. **OVB category-id off-by-one** in `scripts/evaluate_open_vocab.py` (predictions 0–7 vs GT ids 1–8).
   Reproduces the paper's 0.034 exactly. Fixed (commit fa7ec42). Every OV number in the abstract
   paper / `docs/RESULTS.md` is invalid; VisDrone numbers are unaffected.
2. **`optimizer=auto` ignored lr0=2e-4.** 5-epoch run actually used AdamW 7.14e-4; 40/80-epoch runs
   would have used SGD 0.01. The original epoch comparison is confounded.
3. **CLIP was never trainable** on the ultralytics path (cached embeddings, 0 CLIP params in the
   checkpoint); the `txt_model` freeze loop in `finetune.py` matches nothing.
4. **Pretrained cv4 (contrastive head) biases dropped** at fine-tune start (0-d vs shape (1,)), reset
   to −10 instead of −11.96 / −10.45 / −8.97. They barely move during training.
5. **Mosaic confound**: `close_mosaic=10` means short and long runs also differ in augmentation.

## Reproduction (`eval_results/thor_repro/`)

| Model | VisDrone mAP / @50 | OVB all | OVB novel |
|---|---|---|---|
| base yolov8m-worldv2 | 0.131 / 0.198 | 0.263 [0.221, 0.341] | 0.222 [0.206, 0.248] |
| `5_epoch.pt` (paper) | 0.308 / 0.492 | 0.196 [0.149, 0.265] | 0.129 [0.112, 0.151] |
| `paper_e5` (Thor re-run, pinned AdamW 7.14e-4) | 0.299 / 0.479 | 0.161 | 0.142 |

[ ] = 95% percentile bootstrap CI, 1000 image resamples (`eval_results/ov_bootstrap/`).

## Training runs (final epoch)

| Run | LR | Ep | Bias | Freeze | VisDrone mAP / @50 | OVB novel | Hours | Eval dir |
|---|---|---|---|---|---|---|---|---|
| paper_e5 | 7.14e-4 | 5 | – | – | 0.299 / 0.479 | 0.142 | 0.93 | sweep_paper_e5 |
| paper_e5_s1 | 7.14e-4 | 5 | – | – | 0.307 | 0.130 | 0.73 | sweep_paper_e5_s1 |
| paper_e5_s2 | 7.14e-4 | 5 | – | – | 0.302 | 0.102 | 0.73 | sweep_paper_e5_s2 |
| paper_e40 | 7.14e-4 | 40 | – | – | 0.341 / 0.538 | 0.007 | 6.24 | sweep_paper_e40 |
| bias_e5 | 7.14e-4 | 5 | ✓ | – | 0.303 / 0.484 | 0.162 | 0.74 | sweep_bias_e5 |
| bias_e5_s1 | 7.14e-4 | 5 | ✓ | – | 0.304 | 0.126 | 0.71 | sweep_bias_e5_s1 |
| bias_e5_s2 | 7.14e-4 | 5 | ✓ | – | 0.306 | 0.131 | 0.71 | sweep_bias_e5_s2 |
| bias_e40 | 7.14e-4 | 40 | ✓ | – | 0.345 / 0.544 | 0.023 | 5.88 | sweep_bias_e40 |
| lr2.5e-5_e10 | 2.5e-5 | 10 | – | – | 0.271 / 0.431 | 0.181 | 1.42 | sweep_lr2.5e-5_e10 |
| lr1e-4_e10 | 1e-4 | 10 | – | – | 0.304 / 0.484 | 0.202 | 1.42 | sweep_lr1e-4_e10 |
| lr2e-4_e10 | 2e-4 | 10 | – | – | 0.319 / 0.504 | 0.172 | 1.42 | sweep_lr2e-4_e10 |
| lr7.14e-4_e10 | 7.14e-4 | 10 | – | – | 0.327 / 0.519 | 0.131 | 1.44 | sweep_lr7.14e-4_e10 |
| lr1.5e-3_e10 | 1.5e-3 | 10 | – | – | 0.323 / 0.510 | 0.000 | ~1.5 (1 stall, auto-resumed) | sweep_lr1.5e-3_e10 |
| lr1e-4_e10_bias | 1e-4 | 10 | ✓ | – | 0.305 / 0.484 | 0.221 | 1.48 | sweep_lr1e-4_e10_bias |
| lr1e-4_e10_bias_s1 | 1e-4 | 10 | ✓ | – | 0.303 / 0.481 | 0.208 | 1.42 | sweep_lr1e-4_e10_bias_s1 |
| lr1e-4_e10_bias_s2 | 1e-4 | 10 | ✓ | – | 0.305 / 0.484 | 0.216 | 1.47 | sweep_lr1e-4_e10_bias_s2 |
| lr1e-4_e40_bias | 1e-4 | 40 | ✓ | – | 0.338 / 0.532 | 0.171 | 6.03 | sweep_lr1e-4_e40_bias |
| freeze10_e5_bias | 7.14e-4 | 5 | ✓ | backbone | 0.274 | 0.173 | 0.50 | sweep_freeze10_e5_bias |
| freeze22_e5_bias | 7.14e-4 | 5 | ✓ | all but head | 0.227 | 0.209 | 0.38 | sweep_freeze22_e5_bias |
| freeze10_e40_bias | 7.14e-4 | 40 | ✓ | backbone | 0.320 | 0.001 | 4.20 | sweep_freeze10_e40_bias |
| mosaic_e5 | 7.14e-4 | 5 | – | – (mosaic on all epochs) | 0.310 / 0.496 | 0.100 | 0.75 | sweep_mosaic_e5 |
| mosaic_e5_s1 | 7.14e-4 | 5 | – | – (mosaic on) | 0.304 / 0.487 | 0.128 | 0.75 | sweep_mosaic_e5_s1 |
| mosaic_e5_s2 | 7.14e-4 | 5 | – | – (mosaic on) | 0.315 / 0.500 | 0.103 | 0.70 | sweep_mosaic_e5_s2 |
| lr1e-4_e40_bias_s1 | 1e-4 | 40 | ✓ | – | 0.336 / 0.529 | 0.162 | 5.53 | sweep_lr1e-4_e40_bias_s1 |
| lr1e-4_e40_bias_s2 | 1e-4 | 40 | ✓ | – | 0.336 / 0.530 | 0.162 | 5.42 | sweep_lr1e-4_e40_bias_s2 |

Not run: `paper_e80` (dropped by decision 2026-10-31: paper_e40 already shows the OV collapse and
the VisDrone plateau), stock `freeze*` runs (replaced by `_bias` versions), LoRA/replay (journal).

### Seed summaries (3 seeds, mean ± std)

| Recipe | VisDrone mAP | OVB novel |
|---|---|---|
| paper recipe (7.14e-4, 5 ep) | 0.302 ± 0.004 | 0.125 ± 0.021 |
| + bias fix | 0.304 ± 0.002 | 0.139 ± 0.019 |
| + WiSE α0.7 | 0.299 ± 0.002 | 0.209 ± 0.031 |
| + bias fix + WiSE α0.7 | 0.301 ± 0.002 | 0.220 ± 0.024 |
| lr 1e-4 + bias, 10 ep | 0.304 ± 0.001 (@50 0.483 ± 0.002) | 0.215 ± 0.007 |
| lr 1e-4 + bias, 40 ep | 0.337 ± 0.001 (@50 0.530 ± 0.002) | 0.165 ± 0.005 |
| paper recipe, mosaic on all 5 ep (`--close-mosaic 0`) | 0.310 ± 0.006 (@50 0.494 ± 0.007) | 0.110 ± 0.015 |
| **lr 1e-4 + bias, 10 ep + WiSE α0.8 (recommended)** | **0.298 ± 0.000** (@50 0.469 ± 0.001) | **0.253 ± 0.013** |
| base | 0.131 | 0.222 |

Summaries: `eval_results/wise_seeds_summary.json`, `wise_seeds_bias_summary.json`,
`wise_seeds_lr1e-4_bias_summary.json`.

## LR grid (AdamW, 10 epochs, stock bias, seed 0)

| lr0 | VisDrone mAP / @50 | OVB novel / @50 |
|---|---|---|
| 2.5e-5 | 0.271 / 0.431 | 0.181 / 0.374 |
| 1e-4 | 0.304 / 0.484 | 0.202 / 0.371 |
| 2e-4 | 0.319 / 0.504 | 0.172 / 0.288 |
| 7.14e-4 (paper effective) | 0.327 / 0.519 | 0.131 / 0.201 |
| 1.5e-3 | 0.323 / 0.510 | 0.000 / 0.000 |

LR is a smooth VisDrone↔OV dial up to the paper's LR; above it OV collapses with no VisDrone gain.
Novel OV tracks the *current* LR (drops when warmup peaks, partially recovers as LR decays).

## WiSE-FT (θ = (1−α)·base + α·fine-tuned)

α rule (pre-registered, VisDrone only): smallest α whose VisDrone mAP is within 0.01 of α=1.
Picks 0.7 for paper-LR models (6/6 seeds), 0.8 for lr1e-4 models (3/3 seeds).
Weights: `runs/wise_ft/`. Script: `scripts/select_wise_alpha.py`, `scripts/wise_seeds.sh`.

**base ↔ 5_epoch.pt** (`eval_results/sweep_wise_ft/`)

| α | 0.0 | 0.3 | 0.5 | 0.6 | **0.7** | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| VisDrone | 0.130 | 0.217 | 0.275 | 0.295 | **0.308** | 0.315 | 0.315 | 0.308 |
| OVB novel | 0.224 | 0.220 | 0.225 | 0.222 | **0.209** | 0.175 | 0.152 | 0.129 |

**base ↔ paper_e40** (`eval_results/sweep_wise_paper_e40/`, CPU inference)

| α | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| VisDrone | 0.231 | 0.294 | 0.317 | 0.334 | 0.344 | **0.346** | 0.341 |
| OVB novel | 0.185 | 0.143 | 0.109 | 0.092 | 0.085 | 0.045 | 0.007 |

**base ↔ lr1e-4_e40_bias** (`eval_results/sweep_wise_lr1e-4_e40_bias/`)

| α | 0.5 | 0.6 | 0.7 | 0.8 (rule) | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| VisDrone | 0.265 | 0.293 | 0.315 | 0.332 | **0.340** | 0.338 |
| OVB novel | 0.188 | 0.190 | 0.196 | 0.190 | **0.179** | 0.171 |

**base ↔ lr1e-4_e10_bias**, 3 seeds at α0.8: VisDrone 0.298 / 0.298 / 0.299, novel 0.246 / 0.267 / 0.245.

Post-hoc bias restore on 5_epoch.pt (`5_epoch_biasfix`): VisDrone 0.308 → 0.279, novel 0.129 → 0.137
(weights co-adapted to −10; restoring at training start is what works).

## Paired bootstrap vs base (novel OVB mAP difference, 95% CI, 1000 shared resamples)

`eval_results/ov_paired_bootstrap/`, `eval_results/ov_paired_bootstrap_final/` (CPU inference →
only novel-only is reliable).

| Model − base | Novel mAP diff CI | Verdict |
|---|---|---|
| 5_epoch.pt (paper) | [−0.124, −0.070] | significantly worse |
| paper_e5 re-run | [−0.104, −0.062] | significantly worse |
| WiSE α0.7 (5_epoch) | [−0.035, +0.011] | ≈ base |
| WiSE α0.7 + bias | [−0.015, +0.025] | ≈ base |
| lr1e-4_e10_bias | [−0.024, +0.020] | ≈ base |
| **wise_lr1e-4_e10_bias** | **[+0.004, +0.046]** (98.9% ≥ 0) | **significantly better** |
| wise_lr1e-4_e40_bias | [−0.054, −0.008] | slightly worse |

## Standard zero-shot benchmarks (`eval_results/standard_ov/results.json`, imgsz 640)

| Model | LVIS AP (% base) | APr | COCO AP (% base) | ODinW-aerial all / novel (74 img) |
|---|---|---|---|---|
| base | 0.162 (100%) | 0.070 | 0.419 (100%) | 0.049 / 0.034 |
| ft5_paper (5_epoch.pt) | 0.031 (19%) | 0.010 | 0.074 (18%) | 0.118 / 0.061 |
| wise5_a0.7 | 0.088 (55%) | 0.036 | 0.248 (59%) | 0.110 / 0.064 |
| wise5bias_a0.7 | 0.088 (55%) | 0.040 | 0.254 (61%) | 0.083 / 0.043 |
| lr1e-4_e10 | 0.075 (46%) | 0.021 | 0.226 (54%) | 0.093 / 0.057 |
| lr1e-4_e10_bias | 0.080 (49%) | 0.023 | 0.246 (59%) | 0.092 / 0.057 |
| **wise_lr1e-4_e10_bias (α0.8)** | **0.125 (77%)** | **0.043** | **0.325 (78%)** | 0.097 / 0.060 |
| lr2e-4_e10 | 0.057 (35%) | 0.029 | 0.175 (42%) | 0.086 / 0.046 |
| bias_e5 | 0.031 (19%) | 0.012 | 0.082 (19%) | 0.075 / 0.037 |
| lr1e-4_e40_bias | 0.039 (24%) | 0.008 | 0.124 (30%) | 0.103 / 0.047 |
| wise_lr1e-4_e40_bias (α0.8) | 0.088 (54%) | 0.032 | 0.249 (59%) | 0.104 / 0.054 |
| freeze10_e5_bias | 0.067 (41%) | 0.041 | 0.224 (53%) | 0.059 / 0.028 |
| freeze22_e5_bias | 0.094 (58%) | 0.026 | 0.320 (76%) | 0.051 / 0.034 |
| paper_e40 | 0.001 (1%) | 0.000 | 0.007 (2%) | 0.057 / 0.014 |
| wise40_a0.9 | 0.002 (2%) | 0.000 | 0.010 (2%) | 0.061 / 0.014 |
| bias_e40 | 0.002 (1%) | 0.000 | 0.007 (2%) | 0.053 / 0.007 |
| freeze10_e40_bias | 0.005 (3%) | 0.000 | 0.010 (2%) | 0.046 / 0.015 |

ODinW = AerialMaritimeDrone (boat/car/dock/jetski/lift), train+valid+test pooled; novel excludes car.
Small set → directional only. Official 7-image test split is in the JSON too.

LVIS cross-check (fixed AP + YOLO-World synonym prompts, first synonym): base 0.190, ft5_paper 0.028
(15%), wise5_a0.7 0.086 (45%). Ranking holds under both protocols.

## Thor deployment benchmark

120 W mode, batch 1, frames in memory, end-to-end (preprocess + inference + NMS), 500 iters after 30
warm-up, board power from tegrastats. Native venv `~/venvs/yoloworld-trt`, TensorRT 10.16.1.11
(pip wheel ultralytics auto-installed), torch 2.14.1+cu130.
`eval_results/thor_benchmark/` (base, 5_epoch, all runtimes/sizes) and
`eval_results/thor_benchmark_final/` (recommended models).

**imgsz 1280 (paper setting; paper: Xavier 9.4, Orin Nano 12.5 FPS)**

| Model | Runtime | FPS | p99 ms | Board W | VisDrone | OVB novel |
|---|---|---|---|---|---|---|
| base | torch-fp32 / fp16 / trt-fp16 / trt-int8 | 35 / 49 / 57 / 59 | 34.3 / 22.7 / 21.5 / 20.8 | 66 / 57 / 58 / 48 | 0.131 / 0.131 / 0.130 / 0.123 | 0.221 / 0.222 / 0.223 / 0.198 |
| 5_epoch | torch-fp32 / fp16 / trt-fp16 / trt-int8 | 35 / 48 / 57 / 59 | 32.1 / 23.8 / 22.1 / 20.8 | 63 / 59 / 59 / 49 | 0.308 / 0.307 / 0.308 / 0.300 | 0.126 / 0.127 / 0.128 / 0.119 |
| wise5_a0.7 | torch-fp16 / trt-fp16 | 48 / 57 | 23.5 / 22.0 | | 0.306 / 0.307 | 0.210 / 0.200 |
| wise5bias_a0.7 | torch-fp16 / trt-fp16 | 48 / 56 | 23.1 / 22.1 | | 0.298 / 0.298 | 0.229 / 0.231 |
| lr1e-4_e10_bias | torch-fp16 / trt-fp16 | 49 / 55 | 23.3 / 22.0 | | 0.303 / 0.303 | 0.222 / 0.221 |
| **wise_lr1e-4_e10_bias** | torch-fp16 / trt-fp16 | 49 / **56** | 23.4 / 22.0 | | 0.297 / **0.297** | 0.246 / **0.247** |
| lr1e-4_e40_bias | torch-fp16 / trt-fp16 | 49 / 56 | 23.5 / 21.7 | | 0.336 / 0.337 | 0.171 / 0.161 |
| wise_lr1e-4_e40_bias | torch-fp16 / trt-fp16 | 48 / 56 | 23.7 / 22.3 | | 0.329 / 0.330 | 0.194 / 0.185 |

- TensorRT FP16 is lossless; INT8 adds only 2–4% FPS (CPU letterbox + NMS dominate at ~16 ms/frame,
  TRT inference is 8.9 ms) and costs ~0.008 VisDrone / 0.01–0.025 novel. Its upside is ~20% less power.
- 640 px: 159–169 FPS TensorRT; 960 px: ~115 FPS. Fine-tuned models trade novel OV for VisDrone as
  resolution rises (5_epoch novel 0.167 @640 → 0.128 @1280), base is flat. Single model, unconfirmed.
- Measured the paper's way (`evaluate.py`, images read from disk), Thor PyTorch FP32 @1280 ≈ 21 FPS.

## Prompt / landing-decision analyses (2026-11-23, no training)

`scripts/prompt_obstruction_analysis.py` → `eval_results/prompt_analysis/results.json` (cached
predictions in `preds/`). 7 models: base, ft5_paper, paper_e40, wise5_a0.7, lr1e-4_e10_bias,
wise_lr1e-4_e10_bias, lr1e-4_e40_bias.

- **Generic-prompt recall:** vocabulary without backpack/dog/soccer ball prompts. Foreign-object
  detections almost never box the unprompted object tightly (≤ 0.11 recall at IoU 0.5). They do cover
  backpacks lying on the pad (base 84%, recommended 57%), while dogs and balls elsewhere are missed.
  Own-prompt recall at conf 0.1: base 0.53, ft5 0.04, recommended **0.59**.
- **Is the pad obstructed?** (91 obstructed vs. 101 clear images). AUROC, descriptive prompt only:
  base **0.835**, ft5 0.668, wise5 0.759, lr1e-4_e10_bias 0.505, recommended 0.682, paper_e40 0.421.
  Combined with object prompts inside the detected pad (defined post hoc): base 0.866, **recommended
  0.861** (false alarms 3% vs. 11% at the same 54% miss rate, conf 0.25), wise5 0.818, ft5 0.696.
- Per-class OVB AP: "foreign object on pad" base 0.477 → recommended 0.261 (most fragile class);
  "foreign object on driveway" ≈ 0 for every model, base included.

## Models and deliverables

- `checkpoints/` (gitignored .pt, `index.json` committed with sha256): ft5_paper, wise5_a0.7,
  wise5bias_a0.7, lr1e-4_e10_bias, **wise_lr1e-4_e10_bias (recommended)**, lr1e-4_e40_bias,
  wise_lr1e-4_e40_bias, bias_e40, wise40_a0.9. `checkpoints/*.onnx` have the OVB vocabulary baked in.
- `deliverables/videos/`: DJI_0162 annotated per model + `DJI_0162_comparison_4models.mp4`,
  `DJI_0162_comparison_lowLR_e40_vs_wise.mp4` (1080p, conf 0.10, imgsz 1280).
- Figures: `eval_results/pareto_novel.png`, `eval_results/pareto_ov_all.png`,
  `eval_results/figures/dji0162_frame2076_4models.jpg` (frame 2076: lr1e-4_bias_e10 gets pad 0.96,
  pedestrian 0.60, ball 0.77, bag as "foreign object on driveway" 0.24, no false positives; paper
  5-epoch misses the ball, calls the bag "car", hallucinates "dog").
- USB backup: `/mnt/thor_backup/FineTune-YOLOW-UAS_backup/` (git bundle, OVB images, all final run
  weights, eval_results, provenance, SHA256SUMS). Per-epoch checkpoints (~65 GB) are only on the Thor.

## Key findings (for the manuscript)

1. Corrected OVB: base 0.263 / novel 0.222; paper 5-epoch 0.196 / novel 0.129 (significant loss).
   On LVIS/COCO the paper model keeps only 18–19% of base.
2. With the optimizer pinned, 40 epochs → VisDrone 0.341 but novel OV 0.007, LVIS/COCO 1–2%.
3. Learning rate is the lever. lr 1e-4 for 40 epochs → 0.338 VisDrone with novel 0.171.
4. Freezing the backbone only delays the collapse; head-only keeps OV but caps VisDrone at 0.227.
5. WiSE-FT recovers most of the loss for free and stacks with low LR. Recommended model: VisDrone
   0.298, novel 0.253 (> base), LVIS 77% / COCO 78% of base, 56 FPS TRT-FP16 @1280 on Thor.
6. Bias fix: small, significant only at low LR (+0.02 novel in 9/10 epochs; LVIS/COCO +0.005/+0.02).
7. Short/low-LR VisDrone fine-tuning improves zero-shot detection of unseen aerial classes (ODinW).

## Not yet done

(Done 2026-11-23: mosaic control, lr1e-4_e40_bias seeds, prompt/landing-decision analyses.)

- YOLOv8 / RT-DETR closed-set comparison with Vu et al. (promised in the extended abstract).
- LoRA / replay / VLM prompt generation (journal).
