# Jetson Thor experiment log

Unattended run, Fri 2026-10-30 evening → Mon 2026-11-02 morning, branch `thor-experiments`.
Goal: corrected numbers and controlled re-runs for the Dec 1 final manuscript.
All open-vocab (OV) numbers here use the **fixed** evaluator (see Finding 1). OVB = the
195-image open-vocab benchmark; "novel" = its 6 classes not in VisDrone (excludes car,
pedestrian). VisDrone mAP is COCO mAP@[.5:.95] on the 548-image val set. imgsz 1280.

## Current results

| Model | VisDrone mAP | VisDrone mAP@50 | OV mAP | OV mAP novel | Notes |
|---|---|---|---|---|---|
| Base yolov8m-worldv2 | 0.131 | 0.198 | 0.263 [0.221, 0.341] | 0.222 [0.206, 0.248] | paper reported OV 0.034; [95% bootstrap CI] |
| 5_epoch.pt (paper) | 0.308 | 0.492 | 0.196 [0.149, 0.265] | 0.129 [0.112, 0.151] | paper reported OV 0.034 |
| WiSE-FT α=0.6 (base↔5_epoch) | 0.295 | 0.462 | 0.338 | 0.222 | novel = base level |
| **WiSE-FT α=0.7** | **0.308** | **0.485** | **0.329** | **0.209** | α picked by VisDrone-only rule (≤0.01 below α=1); novel **not significantly below base** (paired CI [−0.035, +0.011]) |
| WiSE-FT α=0.8 | 0.315 | 0.497 | 0.291 | 0.175 | beats 5_epoch on both axes |
| 5_epoch + post-hoc restored cv4 biases | 0.279 | 0.454 | 0.204 | 0.137 | restoring biases after training hurts VisDrone, ~no OV gain |
| **paper_e5** (re-run, fixed AdamW 7.14e-4, Thor) | 0.299 | 0.479 | 0.161 | 0.142 | reproduces 5_epoch.pt within ~0.01 |
| bias_e5 (paper_e5 + pretrained cv4 biases kept) | 0.303 | 0.484 | 0.165 | 0.162 | +0.004 VisDrone, +0.020 novel vs paper_e5; seed noise TBD |
| **lr1e-4_e10** (AdamW lr 1e-4, 10 ep, no averaging) | 0.304 | 0.484 | 0.309 | 0.202 | **matches WiSE-FT α0.7 frontier with plain training** |
| lr2e-4_e10 (AdamW lr 2e-4, 10 ep) | 0.319 | 0.504 | 0.297 | 0.172 | on frontier (≈ WiSE α0.8) |
| lr7.14e-4_e10 (paper LR, 10 ep) | 0.327 | 0.519 | 0.251 | 0.131 | 2× epochs vs paper_e5: +0.025 VisDrone, same novel OV |
| lr1.5e-3_e10 (2× paper LR, 10 ep) | 0.323 | 0.510 | 0.090 | 0.000 | open-vocab gone, no VisDrone gain |
| **paper_e40** (re-run, same optimizer, 40 ep) | **0.341** | **0.538** | 0.037 | **0.007** | paper: 40 ep 0.298, 80 ep 0.330; novel OV collapsed |
| WiSE-FT base↔paper_e40 α=0.8 | 0.344 | 0.537 | 0.160 | 0.085 | beats paper_e40 on both axes (CPU eval) |
| **WiSE-FT base↔paper_e40 α=0.9** | **0.346** | **0.543** | 0.086 | 0.045 | **best VisDrone so far** (CPU eval) |
| paper_e5, 3 seeds (mean ± std) | 0.302 ± 0.004 | 0.483 ± 0.003 | 0.181 ± 0.063 | 0.125 ± 0.021 | novel OV varies a lot across seeds |
| **WiSE-FT α=0.7 on paper_e5, 3 seeds** | **0.299 ± 0.002** | — | 0.303 ± 0.056 | **0.209 ± 0.031** | α chosen on VisDrone only (=0.7 in all 3 seeds); OVB held out → **headline result** |

**Thor deployment (paper Table 4 setting: imgsz 1280, batch 1, 120 W mode, frames in memory):**
PyTorch FP32 **35 FPS**, PyTorch FP16 48–49 FPS, **TensorRT FP16 57 FPS with no accuracy
loss** (5_epoch VisDrone 0.308, novel 0.128), TensorRT INT8 59 FPS (−0.008 VisDrone,
−0.01 to −0.025 novel). Paper: Xavier 9.4, Orin Nano 12.5 FPS. Full matrix below (02:25 entry).

Pareto plots: `eval_results/pareto_novel.png`, `eval_results/pareto_ov_all.png`.
_(re-runs and benchmark rows added as they finish)_

## Findings so far (affect the paper)

1. **OV evaluator category-id bug.** `evaluate_open_vocab.py` scored predictions with
   category_id = class index 0–7 against annotations using ids 1–8, so every OV
   prediction was matched to the wrong class. Reproduces the paper's 0.034 exactly.
   Fixed in commit fa7ec42. Affects every OV number in the paper / RESULTS.md, including
   40/80-epoch and the LR-sensitivity table (whose early stopping used this metric).
   VisDrone numbers are unaffected and reproduce to 3 decimals on Thor.
2. **`optimizer='auto'` ignored lr0.** `5_epoch.pt` train_args: optimizer=auto,
   lr0=2e-4, ultralytics 8.4.18. auto picks AdamW lr=0.002·5/(4+nc)=7.14e-4 below 10k
   iterations and SGD-family lr=0.01 above. So 5-epoch (~1.35k it) ran AdamW 7.14e-4 but
   40/80-epoch (~10.8k/21.6k it) would have run SGD lr 0.01: the epoch comparison is
   confounded with optimizer and a ~14× LR change.
3. **CLIP was never trainable on the ultralytics path.** The trainer uses cached text
   embeddings; the checkpoint has 0 CLIP params. The `txt_model` freeze loop in
   `finetune.py` matches nothing. "CLIP frozen" is true by construction, not an ablated
   choice; partial CLIP unfreezing isn't possible here without a custom trainer.
4. **Pretrained contrastive-head biases are dropped at fine-tune start.** ultralytics
   transfers 529/532 tensors: the 3 `model.22.cv4.*.bias` are 0-d in the pretrained
   checkpoint and shape (1,) in the new model, so they reset to the −10 default instead
   of the pretrained −11.96 / −10.45 / −8.97. `5_epoch.pt` still has −10.0 / −10.0 /
   −10.008, shifting class logits by up to ~+2 at the finest scale. Plausibly part of
   the OV degradation and the 4.4× more OVB detections (47.5k vs 10.7k). Ablations:
   `runs/wise_ft/5_epoch_biasfix.pt` (post-hoc restore) and `--restore-head-bias` runs.
5. **OVB includes 2 VisDrone classes** (car, pedestrian); novel-only mAP reported separately.

## Decisions

- **Optimizer for re-runs:** pin exactly what `auto` chose for `5_epoch.pt` (AdamW,
  lr0 7.14e-4, beta1 0.9, warmup_bias_lr 0) for all of 5/40/80 epochs, so only
  training length varies. Other args from `finetune.py` (batch 24, imgsz 1280,
  wd 0.05, warmup 3). Single GPU batch 24 = same global batch as the 2-GPU original.
- **Ultralytics 8.4.194** (container + native venv) vs 8.4.18 for the original run;
  same version everywhere on Thor.
- **Engines/PyTorch timing** runs natively (`~/venvs/yoloworld-trt`, JetPack TensorRT
  10.16); training/eval runs in the `yoloworld-thor` container (its TensorRT 11 doesn't
  support Jetson).

## Log

- **2026-10-30 20:29** Datasets ready on Thor: VisDrone DET 6471 train / 548 val
  (converted with `convert_visdrone.py` + `yolo_to_coco.py`), OVB 195 images.
- **20:33** Reproduction on Thor (`eval_results/thor_repro/`): VisDrone base 0.131 →
  5-ep 0.308 (matches paper). OV corrected: base 0.263 / novel 0.222, 5-ep 0.196 /
  novel 0.129 (−25% / −42%). PyTorch FP32 end-to-end @1280: 17–21 FPS.
- **20:41** Benchmark smoke (yolov8m @640, 120 W): PyTorch FP32 105 FPS, FP16 120 FPS,
  TensorRT FP16 164 FPS (2.8 ms inference), 2.48 FPS/W.
- **20:45** Smoke training run (1 epoch, batch 24, imgsz 1280, 8 workers): 1.9 s/it
  steady state (~8.5 min/epoch); first batch ~2 min (cuDNN autotune); 42 GB GPU mem.
- **20:57** Smoke run **stalled** at iteration 144/270: kernel log full of NVRM
  `NV_ERR_NO_MEMORY` from `_memdescAllocInternal` despite ~56 GB MemAvailable.
  Likely driver-side allocation failure (fragmented unified memory / expandable_segments
  mappings) plus 8 workers × ~3.6 GB mosaic buffers. Killed at 21:04. **Fix:** dropped
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` from `train_sweep.py`, workers 8 → 4.
- **21:05** Committed + pushed fixes and new scripts (fa7ec42) to `thor-experiments`.
- **21:06** Smoke run restarted with the mitigations.
- **21:17** Smoke epoch OK: **1.8 s/it with 4 workers** (no slowdown vs 8), ~8.1 min
  train + ~1.5 min VisDrone val/OVB eval per epoch; 45 GB reserved; **no driver OOMs**.
  1 epoch AdamW lr 2e-4 (ultralytics val): VisDrone mAP50 0.420, mAP50-95 0.253.
  Then crashed in my OV callback: ultralytics 8.4.194 `trainer.tloss` is a dict, not a
  tensor. Fixed (read losses by name).
- **21:18** cv4 biases in the smoke checkpoint after 1 stock epoch: **exactly −10.0 on
  all three scales** — confirms Finding 4, and shows the bias is effectively not updated
  during training (it is a normal trainable `nn.Parameter` in the forward path and isn't
  in the freeze list, so the cause is not yet understood; AdamW would move a trained
  scalar ~0.05 over 270 steps). The `--restore-head-bias` runs test the effect directly.
- **21:20–21:33** Added robustness for the unattended weekend: `run_queue.sh` watchdog
  (kills a run whose log is silent for 20 min) + up to 3 attempts with `--resume` from
  `last.pt`; per-run evaluation via `eval_run.sh` after each training run; "!" host
  steps for the benchmark. **Resume tested**: 2-epoch @640 run killed after epoch 1,
  resumed from last.pt with optimizer state (LR continued at the decayed value), finished
  and logged normally. Fixed a duplicate final row (ultralytics re-fires
  `on_fit_epoch_end` after final validation).
  Side observation from that test (imgsz 640, lr 2e-4): OV mAP 0.254 → **0.111 after a
  single epoch** (novel 0.215 → 0.111). OV loss starts within the first epoch.
- **Decision:** ultralytics names per-epoch checkpoints 0-based (`epoch4.pt` = after
  5 epochs). `eval_run.sh` links `e<N>.pt → epoch<N-1>.pt`; all labels here are
  epochs completed. Evaluated epochs: all ≤10, then every 5th, plus the last.
- **Decision:** INT8 calibration uses ~8% of VisDrone **train** images
  (`configs/thor_sweeps/visdrone_calib.yaml`), not the val images that the accuracy check
  scores.
- **21:45** Per your request, appended **selective-freezing runs** at the end of the
  queue (after `bias_e40`): `freeze10_e5` (backbone frozen, neck + head train),
  `freeze22_e5` (only the WorldDetect head trains), `freeze10_e40`. Verified
  layers 0–9 = backbone (ends at SPPF), `model.22` = WorldDetect. ~8.7 h extra →
  projected queue end ~Mon 03:00. LoRA not queued (needs custom adapter code; journal).
- **21:34** Started WiSE-FT sweep eval (11 alphas + `5_epoch_biasfix.pt`) and
  bootstrap CIs (1000 resamples) for base and 5_epoch.
- **21:45 WiSE-FT results** (`eval_results/sweep_wise_ft/`), θ(α) = (1−α)·base + α·5_epoch:

  | α | VisDrone mAP | mAP@50 | OV mAP | OV mAP@50 | OV novel |
  |---|---|---|---|---|---|
  | 0.0 | 0.130 | 0.195 | 0.264 | 0.435 | 0.224 |
  | 0.1 | 0.158 | 0.238 | 0.264 | 0.436 | 0.219 |
  | 0.2 | 0.188 | 0.285 | 0.330 | 0.504 | 0.222 |
  | 0.3 | 0.217 | 0.332 | 0.331 | 0.506 | 0.220 |
  | 0.4 | 0.248 | 0.381 | 0.331 | 0.508 | 0.217 |
  | 0.5 | 0.275 | 0.428 | 0.339 | 0.519 | 0.225 |
  | 0.6 | 0.295 | 0.462 | 0.338 | 0.512 | 0.222 |
  | 0.7 | 0.308 | 0.485 | 0.329 | 0.479 | 0.209 |
  | 0.8 | 0.315 | 0.497 | 0.291 | 0.428 | 0.175 |
  | 0.9 | 0.315 | 0.499 | 0.272 | 0.392 | 0.152 |
  | 1.0 | 0.308 | 0.492 | 0.196 | 0.294 | 0.129 |

  - α=1.0 reproduces 5_epoch exactly. α=0.0 is within 0.001–0.002 of base (fp16 re-save
    of the interpolated weights), i.e. the endpoints check out.
  - Frontier = α 0.5–0.8; α 0.7 and 0.8 dominate plain 5_epoch on both axes.
  - Non-monotone OV(all) is driven by car/pedestrian (VisDrone classes inside OVB): they
    improve with α while the novel classes stay flat until α≈0.6, then fall.
  - **Caveat:** single fine-tuned checkpoint; α was swept on OVB. The VisDrone-only
    selection rule (`scripts/select_wise_alpha.py`) picks α=0.7 without reading OVB;
    `scripts/wise_seeds.sh` (queued after the seed runs) repeats this on 3 seeds for
    mean ± std.
- **21:45 Post-hoc bias restore** (`5_epoch_biasfix`): VisDrone 0.308 → 0.279, OV 0.196 →
  0.204, novel 0.129 → 0.137. The fine-tuned weights co-adapted to the −10 biases, so
  swapping the pretrained ones back afterwards mostly hurts. The *training-time* restore
  (`bias_e5`) is the real test of Finding 4.
- **21:50 Base bootstrap CIs** (1000 image resamples): OV mAP 0.263 [0.221, 0.341],
  mAP@50 0.440 [0.412, 0.529], novel 0.222 [0.206, 0.248], novel mAP@50 0.394
  [0.372, 0.442]. The 5_epoch novel 0.129 is far outside the base CI. (5_epoch CIs still
  computing — 4.4× more detections make each resample slower; run left going in
  parallel with training since it is CPU-only now.)
- **21:55** Queue launched (`runs/sweep/queue.log`), detached from the session.
- **22:10 5_epoch bootstrap CIs** (1000 resamples): OV mAP 0.196 [0.149, 0.265],
  mAP@50 0.294 [0.248, 0.372], **novel 0.129 [0.112, 0.151]**, novel mAP@50 0.185
  [0.162, 0.215]. Novel-only CIs of base vs 5_epoch **do not overlap** → the novel-class
  loss is significant. All-class OV CIs **do overlap** (car/pedestrian dilute it), so the
  manuscript should lead with the novel-only metric. (`eval_results/ov_bootstrap/`)
  Caveat: both point estimates sit near the *low* end of their percentile intervals
  (resampling duplicates images, which skews AP upward), so the intervals are biased.
  TODO: paired bootstrap on the base − fine-tuned *difference* (same resample for both
  models) is the cleaner significance test for the manuscript.
- **Decision (α selection rule, 0.01 tolerance):** α\* = smallest α whose VisDrone mAP
  is within 0.01 (absolute) of α=1. The tolerance is a judgment call made before seeing
  the seed runs; it picks α=0.7 on 5_epoch.
- **22:08 paper_e5 epoch 1:** OV mAP 0.346 / novel 0.229 — *above* base — at VisDrone
  mAP50 0.416. Unlike the 640 smoke test (−56% after 1 epoch, 1 warmup epoch at lr 2e-4,
  warmup_bias_lr 0.1); here epoch 1 sits inside a 3-epoch warmup with bias warmup from 0.
  Expect the OV drop as LR peaks in epochs 2–3.
- **22:52 paper_e5 finished** (0.93 h for 5 epochs, 1 attempt, no driver errors).
  Evaluated 23:01 (`eval_results/sweep_paper_e5/`, pycocotools, imgsz 1280):

  | epoch | VisDrone mAP | mAP@50 | OV mAP | OV mAP@50 | OV novel | novel@50 |
  |---|---|---|---|---|---|---|
  | 1 | 0.220 | 0.361 | 0.346 | 0.531 | 0.229 | 0.407 |
  | 2 | 0.219 | 0.362 | 0.175 | 0.297 | 0.132 | 0.215 |
  | 3 | 0.257 | 0.415 | 0.158 | 0.285 | 0.145 | 0.251 |
  | 4 | 0.278 | 0.449 | 0.145 | 0.252 | 0.132 | 0.199 |
  | 5 | 0.299 | 0.479 | 0.161 | 0.283 | 0.142 | 0.232 |

  - Epoch 5 reproduces the original `5_epoch.pt` (0.308 / 0.129 novel) within ~0.01
    on both axes, despite 1 GPU vs 2 and ultralytics 8.4.194 vs 8.4.18. So the original
    5-epoch run's *effective* recipe is confirmed as AdamW lr 7.14e-4.
  - The novel-class loss happens in **one step, epoch 1 → 2** (0.229 → 0.132), when the
    LR ramps up through warmup; epochs 2–5 are flat (0.13–0.15) while VisDrone keeps
    climbing (0.219 → 0.299). So training *length* within 2–5 epochs barely matters for
    OV; what matters is reaching the peak LR.
  - Epoch 1 (VisDrone 0.220, OV 0.346, novel 0.229) is itself a useful operating point:
    above base on OV, +68% VisDrone. Comparable to WiSE α≈0.3.
- **23:01** Full benchmark started (between trainings).
- **2026-10-31 02:25 Benchmark finished**, 24/24 configs, no failures
  (`eval_results/thor_benchmark/benchmark_120W.{csv,json}`). Native venv (TensorRT
  **10.16.1.11 from the `tensorrt-cu13` pip wheel** — auto-installed by ultralytics at
  the first engine export on 10-30 20:43, shadowing JetPack's 10.16.2.10; noticed and
  corrected here 10-31 18:40 — torch 2.14.1+cu130), power mode 120 W (not changed). Latency = 500
  batch-1 predictions on 50 VisDrone val frames decoded into memory, end-to-end
  (preprocess + inference + NMS), after 30 warm-up. Power = tegrastats VIN (whole board)
  mean during timing. Accuracy = same evaluators as the paper, re-run with that runtime
  (TensorRT engines are exported per vocabulary: VisDrone classes / OVB classes).

  | Model | Runtime | imgsz | FPS | p50 ms | p99 ms | infer ms | board W | FPS/W | VisDrone mAP | mAP@50 | OV mAP | OV novel | engine build s |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | base | torch-fp32 | 640 | 106 | 9.5 | 10.1 | 7.7 | 62 | 1.70 | 0.083 | 0.125 | 0.254 | 0.215 | - |
  | base | torch-fp16 | 640 | 124 | 8.0 | 8.9 | 6.3 | 50 | 2.48 | 0.082 | 0.125 | 0.252 | 0.212 | - |
  | base | trt-fp16 | 640 | 161 | 6.2 | 7.3 | 2.7 | 53 | 3.05 | 0.082 | 0.125 | 0.234 | 0.216 | ~200 |
  | base | trt-int8 | 640 | 169 | 6.0 | 6.9 | 2.4 | 42 | 4.00 | 0.080 | 0.120 | 0.298 | 0.210 | 490 |
  | base | torch-fp32 | 960 | 69 | 14.8 | 16.8 | 11.9 | 63 | 1.09 | 0.116 | 0.174 | 0.268 | 0.229 | - |
  | base | torch-fp16 | 960 | 91 | 11.1 | 11.8 | 8.8 | 61 | 1.49 | 0.116 | 0.174 | 0.268 | 0.229 | - |
  | base | trt-fp16 | 960 | 116 | 9.2 | 10.5 | 4.8 | 66 | 1.74 | 0.116 | 0.174 | 0.268 | 0.230 | 133 |
  | base | trt-int8 | 960 | 118 | 9.1 | 10.6 | 4.6 | 48 | 2.44 | 0.110 | 0.166 | 0.240 | 0.201 | 694 |
  | base | torch-fp32 | 1280 | 35 | 29.3 | 34.3 | 21.1 | 66 | 0.53 | 0.131 | 0.197 | 0.263 | 0.221 | - |
  | base | torch-fp16 | 1280 | 49 | 19.4 | 22.7 | 14.4 | 57 | 0.87 | 0.131 | 0.197 | 0.264 | 0.222 | - |
  | base | trt-fp16 | 1280 | 57 | 16.3 | 21.5 | 8.9 | 58 | 0.97 | 0.130 | 0.197 | 0.264 | 0.223 | 152 |
  | base | trt-int8 | 1280 | 59 | 15.5 | 20.8 | 8.3 | 48 | 1.22 | 0.123 | 0.185 | 0.241 | 0.198 | 1055 |
  | 5_epoch | torch-fp32 | 640 | 104 | 9.6 | 10.0 | 7.7 | 56 | 1.85 | 0.197 | 0.327 | 0.278 | 0.167 | - |
  | 5_epoch | torch-fp16 | 640 | 110 | 9.1 | 9.7 | 7.1 | 50 | 2.20 | 0.196 | 0.327 | 0.291 | 0.168 | - |
  | 5_epoch | trt-fp16 | 640 | 158 | 6.3 | 7.9 | 2.6 | 55 | 2.88 | 0.197 | 0.327 | 0.277 | 0.166 | 132 |
  | 5_epoch | trt-int8 | 640 | 165 | 6.1 | 7.1 | 2.5 | 41 | 4.01 | 0.192 | 0.318 | 0.268 | 0.159 | 481 |
  | 5_epoch | torch-fp32 | 960 | 68 | 14.8 | 16.4 | 11.9 | 66 | 1.03 | 0.271 | 0.439 | 0.286 | 0.154 | - |
  | 5_epoch | torch-fp16 | 960 | 90 | 11.2 | 11.9 | 8.9 | 57 | 1.57 | 0.270 | 0.438 | 0.285 | 0.153 | - |
  | 5_epoch | trt-fp16 | 960 | 115 | 9.2 | 10.7 | 4.8 | 71 | 1.63 | 0.271 | 0.438 | 0.284 | 0.152 | 135 |
  | 5_epoch | trt-int8 | 960 | 116 | 9.2 | 11.3 | 4.7 | 53 | 2.16 | 0.262 | 0.425 | 0.294 | 0.168 | 693 |
  | 5_epoch | torch-fp32 | 1280 | 35 | 27.2 | 32.1 | 20.5 | 63 | 0.55 | 0.308 | 0.493 | 0.250* | 0.126 | - |
  | 5_epoch | torch-fp16 | 1280 | 48 | 19.7 | 23.8 | 14.6 | 59 | 0.82 | 0.307 | 0.491 | 0.251* | 0.127 | - |
  | 5_epoch | trt-fp16 | 1280 | 57 | 16.1 | 22.1 | 8.8 | 59 | 0.97 | 0.308 | 0.492 | 0.196 | 0.128 | 151 |
  | 5_epoch | trt-int8 | 1280 | 59 | 15.2 | 20.8 | 8.1 | 49 | 1.21 | 0.300 | 0.482 | 0.188 | 0.119 | 1052 |

  - **Table 4 replacement:** at the paper's 1280 setting Thor does 35 FPS (PyTorch FP32)
    / **57 FPS (TensorRT FP16)** vs Xavier 9.4 / Orin Nano 12.5. The paper's numbers
    probably include reading images from disk; measured that way (`evaluate.py`), Thor
    PyTorch FP32 @1280 is ~21 FPS — still 1.7× the Orin Nano.
  - **TensorRT FP16 is lossless** (VisDrone and novel OV within 0.002 of PyTorch) and
    1.6× PyTorch FP32 at 1280. **Recommended deployment config.**
  - **INT8 isn't worth it:** only +2–4% FPS over FP16 (pre/post-processing dominates),
    costs ~0.007 VisDrone mAP and 0.01–0.025 novel OV at 960/1280; its gain is ~20% less
    board power (best FPS/W). Engine builds take 8–18 min (calibration on 518 train imgs).
  - At 1280, TensorRT inference is only 8.9 of ~16 ms per frame: CPU letterbox + NMS are
    now the bottleneck.
  - **Inference resolution trades VisDrone vs novel OV for the fine-tuned model:**
    5_epoch novel OV 0.167 @640 / 0.154 @960 / 0.128 @1280 while VisDrone is 0.197 /
    0.271 / 0.308. Base novel OV is ~flat (0.215 / 0.229 / 0.221). One model; re-check
    on re-runs before claiming it.
  - \* **All-class OV mAP is fragile: OVB "car" has exactly 1 ground-truth box.** Its AP
    can only take a few values; one detection crossing IoU thresholds moves it 0.45 ↔
    0.90, i.e. ±0.056 on 8-class OV mAP. That is the whole 0.196 vs 0.250 gap between
    the container eval and the native PyTorch rows above (every other class within
    0.004; novel-only 0.129 vs 0.126). Explains the wide/skewed all-class bootstrap CIs
    too. → **Use novel-only OV mAP as the headline metric**; report all-class with this
    caveat. OVB box counts: pedestrian 571, backpack 106, landing pad 104, foreign object
    on pad 91, dog 57, soccer ball 37, foreign object on driveway 20, **car 1**.
- **02:25** `paper_e40` started.
- **02:28–02:59 Driver OOM incident (handled, training unaffected).** I started a paired
  bootstrap (base vs 5_epoch / WiSE α0.7 / paper_e5) in a second GPU container while
  `paper_e40` trained. Two `NV_ERR_NO_MEMORY` kernel lines followed (02:3x, 02:5x);
  training kept progressing, but the bootstrap process turned out to hold ~54 GB (CUDA
  context + accumulated pycocotools objects). Killed it at 02:59 (MemAvailable 33 → 88 GB)
  and restarted it **CPU-only** (`--device cpu`, `CUDA_VISIBLE_DEVICES=`), native venv,
  `ulimit -v 24G`, nice 15. Rule from here: nothing else touches the GPU while training.
  Note: CPU inference can flip OVB's single "car" box, so trust the paired *novel-only*
  differences from that run.
- **Added** `--paired-bootstrap N` to `evaluate_open_vocab.py`: CI on mAP(model) −
  mAP(base) with both scored on the same image resamples (cancels the duplicate-image
  bias of per-model percentile CIs), plus the fraction of resamples with diff ≥ 0.
- **05:50 Paired bootstrap done** (`eval_results/ov_paired_bootstrap/`, 1000 resamples,
  each model scored on the same resamples as base; CPU inference, so only novel-only is
  reliable — its point estimates match the GPU evals within 0.003):

  | model − base | novel mAP diff, 95% CI | novel mAP@50 diff, 95% CI | resamples with diff ≥ 0 |
  |---|---|---|---|
  | 5_epoch.pt (paper) | **[−0.124, −0.070]** | [−0.260, −0.179] | 0.0% |
  | paper_e5 (re-run) | **[−0.104, −0.062]** | [−0.209, −0.138] | 0.0% |
  | **WiSE-FT α=0.7** | **[−0.035, +0.011]** | [−0.097, −0.019] | 14.3% |

  → Plain 5-epoch fine-tuning **significantly** reduces novel-class mAP; WiSE-FT α=0.7
  is **not significantly different from base** on novel mAP@[.5:.95] (small but
  significant cost at mAP@50) while matching the fine-tuned VisDrone mAP. This is the
  clean manuscript claim. (All-class diffs are in the JSON but unreliable — car, 1 box.)
- **08:41 paper_e40 finished** (6.24 h, 1 attempt; occasional single driver-OOM lines,
  no stall). Evaluated 09:06 (`eval_results/sweep_paper_e40/`):

  | epoch | VisDrone mAP | mAP@50 | OV mAP | OV novel | novel@50 |
  |---|---|---|---|---|---|
  | 1 | 0.231 | 0.376 | 0.233 | 0.122 | 0.246 |
  | 2 | 0.236 | 0.385 | 0.164 | 0.143 | 0.221 |
  | 3 | 0.256 | 0.409 | 0.101 | 0.091 | 0.146 |
  | 4 | 0.249 | 0.404 | 0.103 | 0.081 | 0.113 |
  | 5 | 0.265 | 0.434 | 0.054 | 0.040 | 0.082 |
  | 6 | 0.249 | 0.414 | 0.097 | 0.082 | 0.126 |
  | 7 | 0.295 | 0.472 | 0.055 | 0.030 | 0.055 |
  | 8 | 0.292 | 0.470 | 0.084 | 0.079 | 0.118 |
  | 9 | 0.304 | 0.489 | 0.042 | 0.021 | 0.037 |
  | 10 | 0.303 | 0.487 | 0.058 | 0.039 | 0.063 |
  | 15 | 0.312 | 0.500 | 0.101 | 0.041 | 0.060 |
  | 20 | 0.332 | 0.532 | 0.087 | 0.004 | 0.006 |
  | 25 | 0.321 | 0.509 | 0.032 | 0.002 | 0.004 |
  | 30 | 0.333 | 0.527 | 0.078 | 0.015 | 0.024 |
  | 35 | 0.341 | 0.538 | 0.043 | 0.015 | 0.024 |
  | 40 | **0.341** | **0.538** | 0.037 | **0.007** | 0.012 |

  - **VisDrone 0.341 at 40 epochs beats every number in the paper** (paper 40-ep 0.298,
    80-ep best 0.330). The paper's long runs were on `optimizer=auto` → SGD lr 0.01; with
    the 5-epoch run's AdamW 7.14e-4 held fixed, 40 epochs is clearly better on VisDrone.
  - **Novel OV collapses to ~0 by epoch 20** and stays there (mosaic off from epoch 31
    did not bring it back). Confirms the paper's qualitative "long training destroys
    open-vocab", now with a controlled optimizer and the fixed metric.
  - Same-epoch comparison with paper_e5 shows schedule + augmentation matter, not just
    epochs: at epoch 5, paper_e5 novel 0.142 (LR decayed to ~1%, no mosaic) vs paper_e40
    novel 0.040 (LR still ~88% of peak, mosaic on).
  - Next: WiSE-FT base ↔ paper_e40 (α 0.3–0.9) — the strongest VisDrone endpoint may give
    a better frontier than base ↔ 5-epoch. Running **CPU-only** (no GPU while training;
    6 threads, nice 15, 24 GB cap) → `eval_results/sweep_wise_paper_e40/`.
- **09:06** `bias_e5` started.
- **09:52 bias_e5 finished** (0.74 h). Evaluated 10:01 (`eval_results/sweep_bias_e5/`):

  | epoch | paper_e5 VisDrone / novel | bias_e5 VisDrone / novel |
  |---|---|---|
  | 1 | 0.220 / 0.229 | 0.222 / 0.180 |
  | 2 | 0.219 / 0.132 | 0.225 / 0.170 |
  | 3 | 0.257 / 0.145 | 0.257 / 0.136 |
  | 4 | 0.278 / 0.132 | 0.285 / 0.145 |
  | 5 | 0.299 / 0.142 | **0.303 / 0.162** |

  - Final: +0.004 VisDrone, **+0.020 novel OV** (novel@50 0.242 vs 0.232). Same direction
    on both axes, but small — needs the seed spread (s1, s2 next) before it can be called
    an effect.
  - Biases after training: paper_e5 −10.008 ×3; bias_e5 −11.961 / −10.445 / −8.977 (≈
    restored values). **In both runs the cv4 bias moves < 0.01 in 5 epochs**, so it is
    effectively fixed at its starting value — the ultralytics transfer bug fully
    determines it in stock fine-tuning.
- **10:01** `paper_e5_s1` started.
- **10:47 paper_e5_s1 finished** (0.73 h), evaluated 10:55 (`eval_results/sweep_paper_e5_s1/`):
  epochs 1–5 VisDrone 0.228 / 0.226 / 0.251 / 0.279 / **0.307**, novel OV 0.158 / 0.133 /
  0.108 / 0.133 / **0.130** (all-class OV at e5 0.251 is the 1-box car flipping).
  Seed 0 e5 was 0.299 / 0.142 → seed spread so far ±0.004 VisDrone, ±0.006 novel;
  bias_e5's novel 0.162 sits above both.
- **10:55** `paper_e5_s2` started.
- **11:41 paper_e5_s2 finished**, evaluated 11:50 (`eval_results/sweep_paper_e5_s2/`):
  e5 VisDrone 0.302 / novel 0.102 / novel@50 0.168. **3-seed summary at epoch 5:**
  VisDrone 0.299 / 0.307 / 0.302 → **0.302 ± 0.004**; novel OV 0.142 / 0.130 / 0.102 →
  **0.125 ± 0.021**; novel@50 0.202 ± 0.032. VisDrone is very stable across seeds;
  novel OV is not. bias_e5 (0.303 / 0.162) is +1.8 σ on novel — suggestive, not
  conclusive from one bias run (would need bias seeds to claim it).
- **11:45 WiSE-FT base ↔ paper_e40** (`eval_results/sweep_wise_paper_e40/`, CPU-only
  inference; base gives identical numbers on CPU and GPU, 0.131 / 0.221):

  | α | VisDrone mAP | mAP@50 | OV mAP | OV novel | novel@50 |
  |---|---|---|---|---|---|
  | 0.3 | 0.231 | 0.352 | 0.307 | 0.185 | 0.314 |
  | 0.4 | 0.263 | 0.406 | 0.294 | 0.168 | 0.278 |
  | 0.5 | 0.294 | 0.456 | 0.262 | 0.143 | 0.222 |
  | 0.6 | 0.317 | 0.494 | 0.237 | 0.109 | 0.159 |
  | 0.7 | 0.334 | 0.520 | 0.223 | 0.092 | 0.129 |
  | 0.8 | 0.344 | 0.537 | 0.160 | 0.085 | 0.118 |
  | 0.9 | **0.346** | **0.543** | 0.086 | 0.045 | 0.063 |
  | 1.0 (paper_e40) | 0.341 | 0.538 | 0.037 | 0.007 | 0.012 |

  - α=0.8–0.9 **beat the 40-epoch model on both axes**; α=0.9 is the best VisDrone
    model so far (0.346 / 0.543), with some novel OV back (0.045 vs 0.007).
  - Combined frontier: base↔5-epoch WiSE dominates for VisDrone ≤ ~0.315 (e.g. α0.8:
    0.315 / novel 0.175 vs this sweep's α0.6: 0.317 / 0.109); base↔40-epoch WiSE owns
    the high-VisDrone end (0.33–0.346). Weight averaging helps at both ends.
- **11:50** `wise_seeds.sh` (WiSE-FT on the 3 paper_e5 seeds, VisDrone-only α rule) started.
- **12:41 WiSE-FT on 3 seeds done** (`eval_results/wise_seeds_summary.json`,
  `eval_results/sweep_wise_paper_e5{,_s1,_s2}/`). α\* = smallest α with VisDrone within
  0.01 of the seed's own fine-tuned model, chosen **without reading OVB**:

  | seed | α\* | VisDrone (WiSE / ft) | OV novel (WiSE / ft) |
  |---|---|---|---|
  | 0 | 0.7 | 0.296 / 0.299 | 0.243 / 0.142 |
  | 1 | 0.7 | 0.299 / 0.307 | 0.203 / 0.130 |
  | 2 | 0.7 | 0.301 / 0.302 | 0.182 / 0.102 |
  | **mean ± std** | 0.7 | **0.299 ± 0.002 / 0.302 ± 0.004** | **0.209 ± 0.031 / 0.125 ± 0.021** |

  **Headline:** WiSE-FT α=0.7 costs 0.003 VisDrone mAP and recovers novel-class OV from
  0.125 to 0.209 (+67%; base 0.222 → ~94% of base), on every seed, with α selected on
  VisDrone only. Same α (0.7) as on the original 5_epoch.pt.
- **12:41** LR grid started (`lr2.5e-5_e10` first).
- **14:08 lr2.5e-5_e10 finished** (1.42 h), evaluated 14:25 (`eval_results/sweep_lr2.5e-5_e10/`):

  | epoch | VisDrone mAP | mAP@50 | OV mAP | OV novel | novel@50 |
  |---|---|---|---|---|---|
  | 1 | 0.168 | 0.267 | 0.154 | 0.155 | 0.308 |
  | 2 | 0.212 | 0.339 | 0.154 | 0.154 | 0.319 |
  | 3 | 0.232 | 0.370 | 0.166 | 0.164 | 0.337 |
  | 4 | 0.245 | 0.387 | 0.176 | 0.165 | 0.348 |
  | 5 | 0.255 | 0.406 | 0.195 | 0.181 | 0.377 |
  | 6 | 0.260 | 0.413 | 0.211 | 0.174 | 0.370 |
  | 7 | 0.264 | 0.420 | 0.216 | 0.180 | 0.368 |
  | 8 | 0.268 | 0.426 | 0.217 | 0.181 | 0.380 |
  | 9 | 0.268 | 0.427 | 0.293 | 0.182 | 0.375 |
  | 10 | 0.271 | 0.431 | 0.293 | 0.181 | 0.374 |

  - 28× lower LR than the paper recipe: novel OV drops once (0.222 → 0.155 at epoch 1),
    then stays flat ~0.18 for 10 epochs while VisDrone climbs slowly to 0.271.
  - **Dominated by WiSE-FT:** base↔5_epoch α=0.5 gives 0.275 / novel 0.225 — better on
    both axes. Lowering the LR is a worse way to buy open-vocab retention than weight
    averaging. Novel mAP@50 stays high (0.374 vs base 0.394): the low LR mostly keeps
    loose-IoU detections.
- **14:25** `lr1e-4_e10` started.
- **15:52 lr1e-4_e10 finished** (1.42 h), evaluated 16:08 (`eval_results/sweep_lr1e-4_e10/`):

  | epoch | VisDrone mAP | mAP@50 | OV mAP | OV novel | novel@50 |
  |---|---|---|---|---|---|
  | 1 | 0.215 | 0.343 | 0.165 | 0.164 | 0.333 |
  | 2 | 0.236 | 0.381 | 0.201 | 0.187 | 0.381 |
  | 3 | 0.267 | 0.426 | 0.295 | 0.183 | 0.372 |
  | 4 | 0.272 | 0.435 | 0.199 | 0.200 | 0.387 |
  | 5 | 0.286 | 0.459 | 0.221 | 0.207 | 0.385 |
  | 6 | 0.286 | 0.461 | 0.244 | 0.192 | 0.365 |
  | 7 | 0.294 | 0.470 | 0.256 | 0.203 | 0.382 |
  | 8 | 0.297 | 0.473 | 0.316 | 0.209 | 0.379 |
  | 9 | 0.303 | 0.480 | 0.310 | 0.203 | 0.378 |
  | 10 | **0.304** | **0.484** | 0.309 | **0.202** | 0.371 |

  - **Plain training reaches the WiSE-FT frontier:** 0.304 / novel 0.202 vs WiSE α0.7
    0.308 / 0.209 (single) and 0.299 ± 0.002 / 0.209 ± 0.031 (3 seeds). Novel OV is
    stable 0.19–0.21 across epochs 4–10 instead of dropping to ~0.13.
  - So the paper recipe's *effective* LR (7.14e-4, from optimizer=auto) is what costs
    the open-vocab ability: at 1e-4, 10 epochs keep ~91% of base novel OV with the same
    VisDrone mAP as 5 epochs at 7.14e-4. Single seed — the effect size (0.202 vs 0.125 ±
    0.021) is large enough that it's very unlikely to be seed noise, but a second seed
    would make it airtight.
  - Next idea (post-queue): WiSE-FT base ↔ lr1e-4_e10 to see if averaging stacks on top.
- **16:08** `lr2e-4_e10` started.
- **17:35 lr2e-4_e10 finished** (1.42 h), evaluated 17:51 (`eval_results/sweep_lr2e-4_e10/`):

  | epoch | VisDrone mAP | mAP@50 | OV mAP | OV novel | novel@50 |
  |---|---|---|---|---|---|
  | 1 | 0.228 | 0.366 | 0.208 | 0.169 | 0.355 |
  | 2 | 0.246 | 0.394 | 0.318 | 0.196 | 0.400 |
  | 3 | 0.277 | 0.443 | 0.301 | 0.194 | 0.352 |
  | 4 | 0.285 | 0.455 | 0.207 | 0.201 | 0.355 |
  | 5 | 0.278 | 0.451 | 0.305 | 0.192 | 0.313 |
  | 6 | 0.294 | 0.470 | 0.216 | 0.179 | 0.304 |
  | 7 | 0.306 | 0.481 | 0.234 | 0.179 | 0.302 |
  | 8 | 0.309 | 0.487 | 0.299 | 0.187 | 0.315 |
  | 9 | 0.314 | 0.497 | 0.290 | 0.178 | 0.304 |
  | 10 | **0.319** | **0.504** | 0.297 | **0.172** | 0.288 |

  - On the frontier: ≈ WiSE base↔5_epoch α0.8 (0.315 / 0.175); clearly above WiSE
    base↔paper_e40 α0.6 (0.317 / 0.109).
  - LR grid so far (10 epochs, final): 2.5e-5 → 0.271 / 0.181; 1e-4 → 0.304 / 0.202;
    2e-4 → 0.319 / 0.172; paper recipe (7.14e-4, 5 ep) → 0.302 / 0.125. LR is a smooth
    VisDrone-vs-open-vocab dial; the paper's effective LR sits *inside* the frontier.
- **17:51** `lr7.14e-4_e10` started.
- **19:19 lr7.14e-4_e10 finished** (1.44 h), evaluated 19:36 (`eval_results/sweep_lr7.14e-4_e10/`):

  | epoch | VisDrone mAP | mAP@50 | OV mAP | OV novel | novel@50 |
  |---|---|---|---|---|---|
  | 1 | 0.220 | 0.361 | 0.346 | 0.229 | 0.407 |
  | 2 | 0.217 | 0.366 | 0.159 | 0.136 | 0.229 |
  | 3 | 0.228 | 0.365 | 0.173 | 0.152 | 0.227 |
  | 4 | 0.268 | 0.432 | 0.137 | 0.125 | 0.196 |
  | 5 | 0.278 | 0.453 | 0.083 | **0.058** | 0.116 |
  | 6 | 0.275 | 0.443 | 0.113 | 0.106 | 0.154 |
  | 7 | 0.304 | 0.478 | 0.144 | 0.128 | 0.207 |
  | 8 | 0.313 | 0.501 | 0.158 | 0.144 | 0.217 |
  | 9 | 0.320 | 0.506 | 0.108 | 0.097 | 0.143 |
  | 10 | **0.327** | **0.519** | 0.251 | **0.131** | 0.201 |

  - Same LR/seed as paper_e5: epoch 1 is identical (0.220 / 0.229) — training is
    deterministic on Thor.
  - **Twice the epochs at the paper LR: +0.025 VisDrone (0.327 vs 0.302 ± 0.004) at the
    same open-vocab cost** (novel 0.131 vs 0.125 ± 0.021).
  - Novel OV dips to 0.058 at epoch 5 (LR still ~50% of peak) and partially recovers as
    the LR decays (0.13–0.14 by epochs 8–10). Novel OV tracks the *current* LR more than
    the epoch count — consistent with paper_e40 (near zero while LR stays high).
- **19:36** `lr1.5e-3_e10` started.
- **20:22 First unattended stall, recovered automatically.** `lr1.5e-3_e10` froze at
  epoch 3, 94% (~20:01) with **no** driver `NV_ERR_NO_MEMORY` lines (the single one at
  20:22:07 coincides with the kill) and 119 GB MemAvailable afterwards — a silent hang
  (dataloader or GPU), not the Friday memory-pressure stall. The watchdog killed it after
  1247 s of silence; attempt 2 resumed from `last.pt` (end of epoch 2) at 20:22:43, so
  only the partial epoch 3 (~10 min) plus the 20-min watchdog window were lost. Watching
  whether it recurs.
- **21:31 lr1.5e-3_e10 finished** (attempt 2 after the stall), evaluated 21:47
  (`eval_results/sweep_lr1.5e-3_e10/`): VisDrone 0.194 / 0.140 / 0.202 / 0.224 / 0.245 /
  0.263 / 0.273 / 0.300 / 0.315 / **0.323**; novel OV 0.173 / 0.035 / 0.028 / 0.001 …
  **0.000** at epoch 10. Epoch 2 shows instability (VisDrone *drops* 0.194 → 0.140 as
  the LR peaks).
- **LR grid complete** (AdamW, 10 epochs, final epoch; same seed, stock biases):

  | lr0 | VisDrone mAP | mAP@50 | OV novel | novel@50 |
  |---|---|---|---|---|
  | 2.5e-5 | 0.271 | 0.431 | 0.181 | 0.374 |
  | **1e-4** | **0.304** | 0.484 | **0.202** | 0.371 |
  | 2e-4 | 0.319 | 0.504 | 0.172 | 0.288 |
  | 7.14e-4 (paper, effective) | 0.327 | 0.519 | 0.131 | 0.201 |
  | 1.5e-3 | 0.323 | 0.510 | **0.000** | 0.000 |
  | *ref: paper_e5 (7.14e-4, 5 ep, 3 seeds)* | *0.302 ± 0.004* | | *0.125 ± 0.021* | |

  - LR is a monotone VisDrone ↔ open-vocab dial up to the paper's LR; above it, open-vocab
    collapses completely with no further VisDrone gain (0.323 < 0.327).
  - 1e-4 and 2e-4 lie on the WiSE-FT frontier; 2.5e-5 is dominated; 7.14e-4 and 1.5e-3
    trade all their extra VisDrone for open-vocab.
  - Single seed per LR (the paper-LR seed spread is ±0.004 VisDrone, ±0.021 novel).
- **21:47** `bias_e40` started.
- **20:35 Re-prioritized to the bias-fixed baseline (your call: upgrade, not
  replicate).** Restoring the pretrained cv4 biases is correct by construction (it
  undoes a loading bug), cost nothing on VisDrone in bias_e5, and at worst does nothing,
  so every *new* run uses `--restore-head-bias`. Changes:
  - Main queue: `lr1.5e-3_e10` (running) and `bias_e40` continue. The stock
    `freeze10_e5`, `freeze22_e5`, `freeze10_e40` are skipped (`runs/sweep/SKIP_*`) →
    the main queue ends ~Sun 04:00 and the follow-up queue starts ~7 h earlier.
  - Follow-up queue (`configs/thor_sweeps/standard_ov_then_e80.txt`, rewritten):
    standard benchmarks on 9 existing models → `lr1e-4_e10_bias` → `bias_e5_s1`,
    `bias_e5_s2` → WiSE-FT on the 3 bias seeds (`wise_seeds.sh` now takes run names) →
    `freeze10_e5_bias`, `freeze22_e5_bias` → `lr1e-4_e40_bias` → `freeze10_e40_bias` →
    standard benchmarks for the new models → `paper_e80` (stock, lowest priority).
  - The stock `lr1e-4_e40` added at 19:55 is replaced by `lr1e-4_e40_bias` (reference:
    `lr1e-4_e10_bias`).
  - Controlled comparisons: freeze*_e5_bias vs bias_e5; freeze10_e40_bias vs bias_e40;
    bias seeds vs paper_e5 seeds; lr1e-4_*_bias vs lr1e-4_e10 (stock) for the bias
    effect at low LR.
  - Estimated: benchmarks Sun ~04:00–07:00, bias 10/5-epoch runs + WiSE ~07:00–13:00,
    lr1e-4_e40_bias ~13:00–20:00, freeze10_e40_bias ~20:00–02:00 Mon, new-model
    benchmarks ~02:00–03:00, paper_e80 ~03:00 → ~16:00 Mon.
- **19:55 Added `lr1e-4_e40` (your request) before paper_e80** — superseded at 20:35 by
  the bias-fixed `lr1e-4_e40_bias`. in
  `configs/thor_sweeps/standard_ov_then_e80.txt`: same recipe as lr1e-4_e10, 40 epochs,
  followed by its standard-benchmark scores. Question it answers: does a low LR only
  *slow* the open-vocab loss or avoid it, and can it reach paper_e40-level VisDrone
  (0.34) while keeping novel OV ~0.20? New timeline: standard benchmarks Sun
  ~11:30–15:00 → lr1e-4_e40 Sun ~15:00–21:45 → paper_e80 Sun ~22:00 → **Mon ~11:00**
  (no longer before Monday morning; e80 is last, so nothing else waits on it).
- **18:20–18:45 Standard open-vocab benchmarks added (your request), to run before
  paper_e80.**
  - Data (official sources, `datasets/`, gitignored): COCO val2017 images + 2017
    annotations (cocodataset.org); LVIS v1 minival as used by YOLO-World/GLIP
    (`lvis_v1_minival_inserted_image_name.json`, 4809 images, 1203 classes, from the
    GLIPModel/GLIP Hugging Face repo); ODinW AerialMaritimeDrone (GLIP odinw_35 zip).
  - `scripts/evaluate_standard_ov.py`: zero-shot, imgsz 640 (YOLO-World protocol),
    conf 0.001, ≤300 dets/image. LVIS via the official `lvis` API (AP, APr/APc/APf —
    APr is the standard "rare/novel" proxy); COCO + ODinW via pycocotools. Prompts =
    category names with "_" → " ". Same sorted-category-id mapping as the fixed OVB
    evaluator (COCO ids are non-contiguous). Shim: `np.float = float` for the
    unmaintained lvis 0.5.3 under NumPy 2 (alias of the builtin, no behaviour change).
  - **Decision:** this is standard *LVIS AP* from the official API, not the "fixed AP"
    variant some YOLO-World tables use, so absolute numbers can differ slightly from the
    paper; comparisons between our models are apples-to-apples.
  - **Decision:** ODinW AerialMaritimeDrone's official test split is 7 images / 78
    boxes. Since we never train on ODinW, I also report `odinw_aerial_all` = large
    train+valid+test pooled (74 images, 1151 boxes; boat/car/dock/jetski/lift), with a
    novel-only score that excludes car. Merge snippet: images re-id'd, `file_name` =
    `<split>/<file>` → `datasets/odinw/AerialMaritimeDrone/large/all_zero_shot.json`.
  - CPU smoke test (base, 20 images): LVIS AP 0.432, COCO AP 0.596, ODinW runs — OK.
  - **Ordering:** `runs/sweep/SKIP_paper_e80` makes the main queue skip e80;
    `runs/sweep/chain_queue2.sh` (detached) waits for the main queue to exit, removes the
    SKIP file and runs `configs/thor_sweeps/standard_ov_then_e80.txt`: the benchmarks
    on 12 models (GPU, nothing else running), then paper_e80. Estimated: benchmarks
    ~Sun 11:30–15:00, paper_e80 done ~Mon 04:00–05:00.
- **paper_e40 during training:** epoch 1 novel OV 0.123 (vs paper_e5 epoch 1: 0.229), epoch 2
  0.143, epoch 3 **0.091** (VisDrone mAP50 0.431 / 0.425 / 0.461). Likely cause of the
  epoch-1 difference: **mosaic augmentation.** ultralytics `close_mosaic=10` disables
  mosaic for the last 10 epochs, so a 5-epoch run never uses mosaic while a 40-epoch run
  uses it for epochs 1–30. The "5 vs 40 epochs" comparison therefore also differs in
  augmentation (true of the paper's original runs too — ultralytics default). A
  `close_mosaic` control would isolate it if the paper needs that.
- **Queue order (as edited during your side session):** paper_e5 → benchmark →
  paper_e40 → bias_e5 → seeds s1, s2 → WiSE-FT on 3 seeds → LR grid → bias_e40 →
  freeze10_e5, freeze22_e5, freeze10_e40 → **paper_e80 last** (40-epoch curve informs
  whether 80 is still needed; drop it by creating `runs/sweep/SKIP_paper_e80`).
