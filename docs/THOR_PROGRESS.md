# Jetson Thor experiment log

Unattended run, Fri 2026-10-30 evening → Mon 2026-11-02 morning, branch `thor-experiments`.
Goal: corrected numbers and controlled re-runs for the Dec 1 final manuscript.
All open-vocab (OV) numbers here use the **fixed** evaluator (see Finding 1). OVB = the
195-image open-vocab benchmark; "novel" = its 6 classes not in VisDrone (excludes car,
pedestrian). VisDrone mAP is COCO mAP@[.5:.95] on the 548-image val set. imgsz 1280.

## Current results

| Model | VisDrone mAP | VisDrone mAP@50 | OV mAP | OV mAP novel | Notes |
|---|---|---|---|---|---|
| Base yolov8m-worldv2 | 0.131 | 0.198 | 0.263 | 0.222 | paper reported OV 0.034 |
| 5_epoch.pt (paper) | 0.308 | 0.492 | 0.196 | 0.129 | paper reported OV 0.034 |

_(updated as runs finish; CIs, WiSE-FT, re-runs and benchmark rows below)_

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
- **21:34** Started WiSE-FT sweep eval (11 alphas + `5_epoch_biasfix.pt`) and
  bootstrap CIs (1000 resamples) for base and 5_epoch.
