# Results for the final manuscript

All numbers come from the Jetson Thor runs with the fixed open-vocab evaluator. Source paths are
given per table. Unless noted:
- **VisDrone mAP** = COCO mAP@[.5:.95] on the 548-image val set, imgsz 1280.
- **OVB novel** = mAP@[.5:.95] on the 6 OVB classes absent from VisDrone (backpack, dog, soccer
  ball, orange circle landing pad, foreign object on orange circle landing pad, foreign object
  on driveway).
- ± = std over 3 seeds; [a, b] = 95% bootstrap CI.

## Model names used below

| Short name | What it is | Checkpoint |
|---|---|---|
| base | pretrained YOLOv8m-WorldV2 | `yolov8m-worldv2.pt` |
| paper 5-ep | abstract's model (AdamW 7.14e-4, 5 ep) | `5_epoch.pt` / `checkpoints/ft5_paper.pt` |
| paper 40-ep | same recipe, 40 ep (controlled re-run) | `runs/sweep/paper_e40/weights/last.pt` |
| low-LR 10-ep | AdamW 1e-4, 10 ep, head biases restored | `checkpoints/lr1e-4_e10_bias.pt` |
| low-LR 40-ep | AdamW 1e-4, 40 ep, head biases restored | `checkpoints/lr1e-4_e40_bias.pt` |
| WiSE α0.7 | 0.3·base + 0.7·paper 5-ep | `checkpoints/wise5_a0.7.pt` |
| **recommended** | 0.2·base + 0.8·low-LR 10-ep | `checkpoints/wise_lr1e-4_e10_bias.pt` |

---

## R1. Main table (replaces abstract Table 3)

| Model | VisDrone mAP | VisDrone mAP@50 | OVB mAP (all 8) | OVB novel mAP |
|---|---|---|---|---|
| Base | 0.131 | 0.198 | 0.263 | 0.222 |
| Fine-tuned, 5 epoch | 0.308 | 0.492 | 0.196 | 0.129 |
| Fine-tuned, 40 epoch | 0.341 | 0.538 | 0.037 | 0.007 |

- 5-epoch vs. base: VisDrone +135% (mAP), +149% (mAP@50); OVB novel −42%, OVB all −25%.
- Paired bootstrap of the novel-mAP drop (1000 shared image resamples): **[−0.124, −0.070]**.
- Per-model bootstrap CIs: base novel [0.206, 0.248], 5-ep novel [0.112, 0.151] (do not overlap).
- 3-seed re-runs of the 5-epoch recipe: VisDrone 0.302 ± 0.004, novel 0.125 ± 0.021.
  The original `5_epoch.pt` sits inside this spread.
- Source: `eval_results/thor_repro/`, `eval_results/sweep_paper_e40/`, `eval_results/ov_bootstrap/`,
  `eval_results/ov_paired_bootstrap/`.

## R2. Per-class VisDrone AP (abstract Table 2): unchanged

The VisDrone evaluation was unaffected by the bug and reproduces to 3 decimals on Thor.

## R3. Hyperparameters (fixes abstract Table 1)

AdamW (β1 0.9), lr0 **7.14e-4**, lrf 0.01 (linear), weight decay 0.05, batch 24, imgsz 1280,
warmup 3 epochs (bias warmup LR 0), AMP, mosaic disabled for the final 10 epochs (ultralytics
default `close_mosaic=10`). Backbone, neck and head train (`freeze: None`); the CLIP text
encoder is frozen. Original training: 2× RTX 5090; re-runs: one Jetson Thor GPU, same global
batch. The Thor re-run of the 5-epoch recipe reproduces `5_epoch.pt` within 0.01 on both axes.

## R4. Learning rate × duration

**LR grid** (AdamW, 10 epochs, seed 0, stock head bias; `eval_results/sweep_lr*_e10/`):

| lr0 | VisDrone mAP | mAP@50 | OVB novel | novel@50 |
|---|---|---|---|---|
| 2.5e-5 | 0.271 | 0.431 | 0.181 | 0.374 |
| 1e-4 | 0.304 | 0.484 | 0.202 | 0.371 |
| 2e-4 | 0.319 | 0.504 | 0.172 | 0.288 |
| 7.14e-4 (abstract) | 0.327 | 0.519 | 0.131 | 0.201 |
| 1.5e-3 | 0.323 | 0.510 | 0.000 | 0.000 |

**Duration at two LRs** (final epoch):

| Recipe | 5 ep | 10 ep | 40 ep |
|---|---|---|---|
| lr 7.14e-4: VisDrone / novel | 0.302 ± 0.004 / 0.125 ± 0.021 | 0.327 / 0.131 | 0.341 / 0.007 |
| lr 1e-4 + bias: VisDrone / novel | — | 0.304 ± 0.001 / 0.215 ± 0.007 | **0.337 ± 0.001 / 0.165 ± 0.005** |

- lr 1e-4, 40 ep, 3 seeds: VisDrone mAP@50 0.530 ± 0.002, novel@50 0.270 ± 0.007
  (`eval_results/sweep_lr1e-4_e40_bias{,_s1,_s2}/`; seeds 1–2 finished 2026-11-23).
- Novel OV tracks the *current* LR. In the abstract recipe it drops in one step (epoch 1 → 2,
  0.229 → 0.132) as warmup reaches peak LR, then stays flat while VisDrone climbs. At the abstract
  LR over 40 epochs it reaches ~0 by epoch 20 and never recovers.

## R5. Mosaic control (closes the augmentation confound)

The 5-epoch recipe never uses mosaic (`close_mosaic=10` > 5 epochs); the 40-epoch run uses it for
epochs 1–30. Control: the 5-epoch recipe with mosaic on throughout (`close_mosaic=0`), 3 seeds
(`eval_results/sweep_mosaic_e5{,_s1,_s2}/`).

| 5-epoch recipe, 3 seeds | VisDrone mAP | mAP@50 | OVB novel | novel@50 |
|---|---|---|---|---|
| mosaic off (default) | 0.302 ± 0.004 | 0.483 ± 0.003 | 0.125 ± 0.021 | 0.202 ± 0.032 |
| mosaic on | 0.310 ± 0.006 | 0.494 ± 0.007 | 0.110 ± 0.015 | 0.163 ± 0.021 |

Mosaic adds ~+0.008 VisDrone and costs ~0.015 novel, both within about one std. Augmentation
cannot explain the 40-epoch collapse (novel 0.007). Training length at a high LR does.

## R6. Layer freezing (lr 7.14e-4, head biases restored)

| Trainable | Epochs | VisDrone mAP | OVB novel | LVIS (% base) | COCO (% base) |
|---|---|---|---|---|---|
| all (backbone+neck+head), 3 seeds | 5 | 0.304 ± 0.002 | 0.139 ± 0.019 | 19% | 19% |
| neck + head (backbone frozen) | 5 | 0.274 | 0.173 | 41% | 53% |
| head only | 5 | 0.227 | 0.209 | 58% | 76% |
| all | 40 | 0.345 | 0.023 | 1% | 2% |
| neck + head (backbone frozen) | 40 | 0.320 | 0.001 | 3% | 2% |

Freezing trades VisDrone for OV at 5 epochs, but at 40 epochs a frozen backbone still collapses.
It delays the loss (novel 0.125 at epoch 10 vs. 0.016 unfrozen) and does not prevent it. Neither
freezing level beats WiSE-FT at equal VisDrone accuracy. Training time: frozen backbone −30%,
head only −50%.

## R7. Retention methods — main comparison table

| Method | VisDrone mAP | OVB novel | LVIS AP (% base) | COCO AP (% base) | ODinW-aerial novel |
|---|---|---|---|---|---|
| Base (no fine-tuning) | 0.131 | 0.222 | 0.162 (100%) | 0.419 (100%) | 0.034 |
| Fine-tune, 5 ep (abstract) | 0.302 ± 0.004 | 0.125 ± 0.021 | 0.031 (19%) | 0.074 (18%) | 0.061 |
| Fine-tune, 40 ep | 0.341 | 0.007 | 0.001 (1%) | 0.007 (2%) | 0.014 |
| Head-only, 5 ep | 0.227 | 0.209 | 0.094 (58%) | 0.320 (76%) | 0.034 |
| Low LR (1e-4), 10 ep | 0.304 ± 0.001 | 0.215 ± 0.007 | 0.080 (49%) | 0.246 (59%) | 0.057 |
| Low LR (1e-4), 40 ep | 0.337 ± 0.001 | 0.165 ± 0.005 | 0.039 (24%) | 0.124 (30%) | 0.047 |
| 5 ep + WiSE-FT α0.7 | 0.299 ± 0.002 | 0.209 ± 0.031 | 0.088 (55%) | 0.248 (59%) | 0.064 |
| **Low LR 10 ep + WiSE-FT α0.8 (recommended)** | **0.298 ± 0.000** | **0.253 ± 0.013** | **0.125 (77%)** | **0.325 (78%)** | 0.060 |
| Low LR 40 ep + WiSE-FT α0.8 | 0.332 | 0.190 | 0.088 (54%) | 0.249 (59%) | 0.054 |

- LVIS = minival 4809 images, standard AP, official API; COCO = val2017; ODinW = AerialMaritimeDrone
  (boat/dock/jetski/lift; car excluded), 74 images pooled; all zero-shot at imgsz 640.
  LVIS / COCO / ODinW are single-seed (seed 0). Source: `eval_results/standard_ov/results.json`.
- Recommended model, LVIS APr (rare) 0.043 vs. base 0.070; paper 5-ep 0.010.
- Paired bootstrap, novel OVB difference vs. base (`eval_results/ov_paired_bootstrap{,_final}/`):

| Model − base | 95% CI | |
|---|---|---|
| paper 5-ep | [−0.124, −0.070] | worse |
| 5 ep + WiSE α0.7 | [−0.035, +0.011] | ≈ base |
| low LR 10 ep | [−0.024, +0.020] | ≈ base |
| **recommended** | **[+0.004, +0.046]** (98.9% of resamples ≥ 0) | **better** |
| low LR 40 ep + WiSE α0.8 | [−0.054, −0.008] | slightly worse |

- WiSE-FT α rule (fixed before the seed runs): smallest α whose VisDrone mAP is within 0.01 of
  α=1, chosen **without looking at OVB**. It picked 0.7 for every paper-LR seed (6/6) and 0.8 for
  every low-LR seed (3/3).
- WiSE α sweep on `5_epoch.pt` (VisDrone / novel): α0.5 0.275 / 0.225, α0.6 0.295 / 0.222,
  α0.7 0.308 / 0.209, α0.8 0.315 / 0.175, α1.0 0.308 / 0.129. α0.7–0.8 beat the plain
  fine-tune on *both* axes.
- Best VisDrone with OV kept: WiSE α0.9 on low-LR 40-ep → 0.340 / 0.531, novel 0.179.
- LVIS cross-check (fixed AP + YOLO-World synonym prompts): base 0.190, paper 5-ep 15%, WiSE α0.7
  45%. Same ranking as standard AP.

## R8. Head-bias restoration (implementation detail)

Ultralytics' weight transfer drops the three pretrained contrastive-head bias scalars (0-d in the
checkpoint vs. shape (1,) in the model) and resets them to −10 instead of −11.96 / −10.45 / −8.97.
They barely move during training. Restoring them at the start of training:
- at the abstract LR (3 seeds): +0.014 novel (0.139 ± 0.019 vs. 0.125 ± 0.021), not significant;
- at lr 1e-4: novel higher in 9/10 epochs (mean +0.020), LVIS 0.080 vs. 0.075, COCO 0.246 vs. 0.226,
  VisDrone identical;
- restoring after training hurts (VisDrone 0.308 → 0.279).

One sentence in Methods is enough. It is a loader bug fix, not a method.

## R9. Per-class OVB AP (descriptive-prompt fragility)

Source: `eval_results/sweep_*/sweep_results.json` (per_class).

| Model | backpack | dog | soccer ball | landing pad | FO on pad | FO on driveway | pedestrian | car* |
|---|---|---|---|---|---|---|---|---|
| Base | 0.076 | 0.360 | 0.016 | 0.398 | **0.477** | 0.005 | 0.275 | 0.500 |
| Paper 5-ep | 0.014 | 0.011 | 0.013 | 0.424 | 0.310 | 0.001 | 0.347 | 0.450 |
| Paper 40-ep | 0.000 | 0.000 | 0.000 | 0.042 | 0.000 | 0.000 | 0.216 | 0.040 |
| WiSE α0.7 | 0.050 | 0.178 | 0.025 | 0.547 | 0.451 | 0.001 | 0.379 | 1.000 |
| Low LR 10 ep | 0.153 | 0.438 | 0.017 | 0.580 | 0.110 | 0.028 | 0.361 | 0.500 |
| **Recommended** | **0.174** | 0.423 | 0.017 | 0.577 | 0.261 | 0.026 | 0.377 | 1.000 |
| Low LR 40 ep | 0.058 | 0.262 | 0.022 | 0.627 | 0.056 | 0.000 | 0.354 | 0.043 |

\* car has one ground-truth box, so its AP jumps between a few values. Don't interpret it.

- Concrete nouns recover or improve with the low-LR recipes (backpack 2.3×, landing pad +45%).
- The compositional prompt "foreign object on orange circle landing pad" is the most fragile class:
  it drops under every recipe except WiSE α0.7 on the paper model.
- "foreign object on driveway" is near zero for **every** model, base included (20 boxes), so
  this prompt does not work in our setting.
- Soccer ball is near zero everywhere (small objects, median ~1700 px² at 1920×1080).

## R10. Generic-prompt recall (does a descriptive prompt catch unprompted objects?)

Setup: vocabulary = {landing pad, pedestrian, car, foreign object on pad, foreign object on
driveway}, **no** backpack / dog / soccer-ball prompt. Count how many of the 200 held-out boxes
(106 backpack, 57 dog, 37 soccer ball) a foreign-object detection finds (conf ≥ 0.1).
- "Tight" = a foreign-object box overlaps the object at IoU ≥ 0.5.
- "Covering" = a foreign-object box no larger than 25% of the image contains ≥ 50% of the object.
- Reference = recall when the object's own prompt is given.

Source: `eval_results/prompt_analysis/results.json`, script `scripts/prompt_obstruction_analysis.py`.

| Model | Own prompt (ref.) | Generic, tight | Generic, tight or covering | — backpack / dog / ball |
|---|---|---|---|---|
| Base | 0.53 | 0.01 | 0.46 | 0.84 / 0.00 / 0.08 |
| Paper 5-ep | 0.04 | 0.01 | 0.26 | 0.46 / 0.00 / 0.08 |
| Paper 40-ep | 0.00 | 0.00 | 0.00 | 0 / 0 / 0 |
| WiSE α0.7 | 0.27 | 0.05 | 0.34 | 0.52 / 0.00 / 0.32 |
| Low LR 10 ep | 0.46 | 0.06 | 0.15 | 0.17 / 0.19 / 0.03 |
| **Recommended** | **0.59** | 0.04 | 0.34 | 0.57 / 0.12 / 0.03 |
| Low LR 40 ep | 0.24 | 0.11 | 0.27 | 0.31 / 0.23 / 0.19 |

Foreign-object false detections per clear-pad image (conf 0.1): base 0.49, paper 5-ep 0.31,
recommended 0.31, low LR 40 ep 0.47.

Interpretation:
- Generic prompts almost never produce a tight box around an unprompted object (≤ 11%).
- What they do is flag the **pad region** when something lies on it. 84% of backpacks (which lie on
  the pad) are covered by a base-model "foreign object on pad" box. Loose objects elsewhere (dogs,
  most balls) are missed.
- The descriptive prompt is a region-level "pad is obstructed" signal, not an object detector.
  Specific prompts remain necessary for localization, and the recommended model has the best
  specific-prompt recall (0.59 vs. base 0.53).

## R11. Image-level landing decision: "is the pad obstructed?"

Label from OVB boxes: positive = image has a "foreign object on pad" box (91); negative = image
has a landing-pad box and no foreign-object-on-pad box (101); 3 pedestrian-only images excluded.
Bootstrap CI over images (1000).

- **Descriptive-only score** (pre-specified) = max confidence of the "foreign object on pad" prompt.
- **Combined score** (defined after seeing the descriptive-only result; report as secondary) = max
  of that and the confidence of any backpack / dog / soccer ball / FO-on-driveway detection whose
  centre lies inside a pad or FO-on-pad detection (conf ≥ 0.1).
- Full 8-prompt vocabulary for both.

| Model | AUROC, descriptive only | AUROC, combined | Combined: miss / false alarm @ conf 0.25 |
|---|---|---|---|
| Base | **0.835** [0.774, 0.893] | **0.866** [0.813, 0.920] | 0.54 / 0.11 |
| Paper 5-ep | 0.668 [0.590, 0.746] | 0.696 [0.620, 0.773] | 0.81 / 0.02 |
| Paper 40-ep | 0.421 [0.354, 0.491] | 0.421 [0.354, 0.491] | 1.00 / 0.00 |
| WiSE α0.7 | 0.759 [0.687, 0.826] | 0.818 [0.755, 0.874] | 0.76 / 0.01 |
| Low LR 10 ep | 0.505 [0.424, 0.584] | 0.711 [0.636, 0.778] | 0.77 / 0.03 |
| **Recommended** | 0.682 [0.602, 0.756] | **0.861** [0.806, 0.908] | **0.54 / 0.03** |
| Low LR 40 ep | 0.572 [0.489, 0.657] | 0.683 [0.605, 0.761] | 0.96 / 0.02 |

- Descriptive-only: base is best, and every fine-tuned model is worse (the per-class FO-on-pad
  drop in R9).
- Combined: the recommended model matches base (0.861 vs. 0.866), with the same miss rate and a
  **3.6× lower false-alarm rate** (3% vs. 11%) at conf 0.25. The abstract's model reaches 0.696.
- With the generic-only vocabulary (no object prompts), descriptive-only AUROC: base 0.858,
  recommended 0.694, paper 5-ep 0.636, so removing the specific prompts doesn't change the picture.
- Miss rates of 54%+ at conf 0.25 mean a safety system would run a lower threshold (e.g. conf
  0.1: recommended, descriptive only, misses 59% with 9% false alarms). AUROC is the
  threshold-free comparison. A threshold should be calibrated on more data before any claim about
  operational safety.

## R12. Jetson Thor deployment (extends abstract Table 4)

Method: 120 W power mode, batch 1, imgsz 1280, frames decoded into memory, end-to-end
(preprocess + inference + NMS), 300–500 timed iterations after 30 warm-up. Board power from
tegrastats (whole board). Accuracy re-scored with each runtime. TensorRT 10.16, torch 2.14.1.
Source: `eval_results/thor_benchmark/`, `eval_results/thor_benchmark_final/`.

| Model | Runtime | FPS | p99 latency (ms) | VisDrone mAP | OVB novel |
|---|---|---|---|---|---|
| Paper 5-ep | PyTorch FP32 | 35 | 32.1 | 0.308 | 0.126 |
| Paper 5-ep | PyTorch FP16 | 48 | 23.8 | 0.307 | 0.127 |
| Paper 5-ep | TensorRT FP16 | 57 | 22.1 | 0.308 | 0.128 |
| Paper 5-ep | TensorRT INT8 | 59 | 20.8 | 0.300 | 0.119 |
| **Recommended** | PyTorch FP16 | 49 | 23.4 | 0.297 | 0.246 |
| **Recommended** | **TensorRT FP16** | **56** | 22.0 | **0.297** | **0.247** |

Platform comparison at 1280 (5-epoch model): Xavier 9.4 FPS, Orin Nano 12.5 FPS (abstract,
PyTorch), Thor 35 FPS (PyTorch FP32) / 57 FPS (TensorRT FP16).
- Caveat: the Xavier/Orin numbers probably include image loading from disk. Measured that way,
  Thor PyTorch FP32 ≈ 21 FPS, still 1.7× Orin Nano. Say so, or present Thor separately.
- TensorRT FP16 is lossless (VisDrone identical, novel within 0.01). INT8 adds 2–4% FPS for
  −0.008 VisDrone / −0.01–0.025 novel, because CPU letterbox and NMS dominate (TensorRT inference
  is 8.9 of ~16 ms per frame). INT8 does use ~20% less board power.
- At 640 px: ~160 FPS TensorRT FP16. Board power ~55–60 W at 1280 (0.97–1.04 FPS/W).
- TensorRT engines are built per vocabulary (re-export when the prompt list changes, ~2.5 min at
  FP16).

## Figures

| Figure | File | Use |
|---|---|---|
| Pareto: VisDrone mAP vs. OVB novel, all runs | `eval_results/pareto_novel.png` | Main results figure. Regenerate to add the 3-seed points and the mosaic runs if you want them |
| Pareto, all-class OVB | `eval_results/pareto_ov_all.png` | Appendix only (car artifact) |
| 4-model qualitative frame | `eval_results/figures/dji0162_frame2076_4models.jpg` | Base / paper 5-ep / WiSE α0.7 / low-LR 10-ep on frame 2076. Low-LR: pad 0.96, pedestrian 0.60, ball 0.77, bag flagged "foreign object on driveway" 0.24, no false positives; paper 5-ep misses the ball, calls the bag "car", hallucinates "dog". Single frame, illustrative only. The recommended model's video exists (`deliverables/videos/DJI_0162_wise_lr1e-4_bias_e10.mp4`), so a frame could be pulled from it |
| LVIS-style showcase collage | `eval_results/figures/qualitative/showcase__wise_lr1e-4_e10_bias.jpg` (+ same tiles for `ft5_paper`) | 21 tiles from OVB/VisDrone/ODinW/COCO, curated for the recommended model (0 empty tiles by construction); on the same tiles paper 5-ep misses 10/21 (elephant, zebra, pizza, dog, backpack, pad, …). Captions and caveats: `eval_results/figures/qualitative/README.md` |
| Model comparison grids | `eval_results/figures/qualitative/compare_{ovb_gain,coco_gain,visdrone_ft,ovb_fail}.jpg` | 4 models × 4 images, auto-ranked by per-image F1 difference; OVB/COCO show the paper model's open-vocab loss, VisDrone the fine-tuning gain, `ovb_fail` the shared failure mode (high altitude + shadow) |
| Existing abstract figures | `baseline_4.png`, `finetuned_4.png`, `landing_pad_backpack_*` | Keep. The backpack figure's message (confidence drops after fine-tuning) is still correct |

## Numbers for the abstract / conclusion

- Recommended vs. base: VisDrone mAP 0.131 → 0.298 (**+127%**), mAP@50 0.198 → 0.469 (**+137%**).
- Novel OVB 0.222 → 0.253 (+14%, paired CI [+0.004, +0.046]).
- LVIS 77% / COCO 78% of base (abstract model: 19% / 18%).
- 56 FPS TensorRT FP16 @1280 on Jetson Thor (4.5× Orin Nano's 12.5 FPS, different runtime).
- Pad-obstruction AUROC 0.861 vs. base 0.866, with false alarms 3% vs. 11%.
