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
| **WiSE-FT α=0.7** | **0.308** | **0.485** | **0.329** | **0.209** | α picked by VisDrone-only rule (≤0.01 below α=1) |
| WiSE-FT α=0.8 | 0.315 | 0.497 | 0.291 | 0.175 | beats 5_epoch on both axes |
| 5_epoch + post-hoc restored cv4 biases | 0.279 | 0.454 | 0.204 | 0.137 | restoring biases after training hurts VisDrone, ~no OV gain |

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
- **22:08 paper_e5 epoch 1:** OV mAP 0.346 / novel 0.229 — *above* base — at VisDrone
  mAP50 0.416. Unlike the 640 smoke test (−56% after 1 epoch, 1 warmup epoch at lr 2e-4,
  warmup_bias_lr 0.1); here epoch 1 sits inside a 3-epoch warmup with bias warmup from 0.
  Expect the OV drop as LR peaks in epochs 2–3.
- **Queue order (as edited during your side session):** paper_e5 → benchmark →
  paper_e40 → bias_e5 → seeds s1, s2 → WiSE-FT on 3 seeds → LR grid → bias_e40 →
  freeze10_e5, freeze22_e5, freeze10_e40 → **paper_e80 last** (40-epoch curve informs
  whether 80 is still needed; drop it by creating `runs/sweep/SKIP_paper_e80`).
