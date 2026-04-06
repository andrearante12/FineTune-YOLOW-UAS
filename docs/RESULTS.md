# Results

## Summary

Fine-tuning YOLO-World (YOLOv8m-WorldV2) on the VisDrone dataset for 5 epochs with a frozen CLIP text encoder yields a **2.3x improvement** in VisDrone detection (mAP 0.131 to 0.308) while fully preserving open-vocabulary capability (OV mAP 0.034 baseline vs. 0.034 after fine-tuning). Extended training (40 or 80 epochs) produces diminishing VisDrone gains while progressively destroying open-vocab performance.

## VisDrone Closed-Vocabulary Performance

Evaluated on the 548-image VisDrone validation set using standard COCO mAP (IoU 0.50:0.95).

| Model | mAP | mAP@50 | mAP@75 | Small | Medium | Large | FPS |
|-------|-----|--------|--------|-------|--------|-------|-----|
| Base (YOLOv8m-WorldV2) | 0.131 | 0.198 | 0.139 | 0.066 | 0.184 | 0.359 | 87 |
| Fine-tuned (5 epoch) | **0.308** | **0.492** | **0.322** | **0.219** | **0.432** | **0.579** | 95 |
| Fine-tuned (40 epoch) | 0.298 | 0.476 | 0.310 | 0.212 | 0.418 | 0.478 | 95 |
| Fine-tuned (80 epoch, best) | 0.330 | 0.525 | 0.346 | 0.244 | 0.449 | 0.527 | 96 |

The 5-epoch checkpoint achieves the best balance. The 80-epoch model achieves marginally higher mAP (0.330) but at the cost of total open-vocabulary collapse (see below). FPS is consistent or slightly improved across all fine-tuned variants.

## Per-Class VisDrone AP

| Class | Base | 5-epoch | 40-epoch | Change (5-ep) |
|-------|------|---------|----------|---------------|
| pedestrian | 0.081 | 0.325 | 0.309 | +301% |
| people | 0.044 | 0.215 | 0.200 | +389% |
| bicycle | 0.064 | 0.155 | 0.134 | +143% |
| car | 0.469 | 0.622 | 0.620 | +33% |
| van | 0.149 | 0.317 | 0.352 | +113% |
| truck | 0.156 | 0.338 | 0.302 | +117% |
| tricycle | 0.023 | 0.230 | 0.217 | +900% |
| awning-tricycle | 0.036 | 0.123 | 0.097 | +242% |
| bus | 0.293 | 0.454 | 0.458 | +55% |
| motor | 0.000 | 0.303 | 0.288 | -- (from zero) |

Every class improved. The largest gains are on small, hard-to-detect classes (pedestrian, people, tricycle, motor) that the base open-vocab model struggled with on drone imagery.

## Open-Vocabulary Performance

Evaluated on the 195-image Open-Vocabulary Benchmark (OVB) set using COCO mAP against ground-truth annotations for 8 novel classes not present in VisDrone training.

| Model | OV mAP | OV mAP@50 | OV mAP@75 |
|-------|--------|-----------|-----------|
| Base | 0.034 | 0.051 | 0.046 |
| Fine-tuned (5 epoch) | 0.034 | 0.055 | 0.039 |
| Fine-tuned (40 epoch) | 0.017 | 0.027 | 0.019 |
| Fine-tuned (80 epoch, best) | 0.001 | 0.001 | 0.000 |

The 5-epoch model retains virtually identical open-vocab mAP to the base model. The 40-epoch model loses ~50% of open-vocab performance, and the 80-epoch model loses it entirely.

## Open-Vocabulary Confidence Retention

OV Confidence Retention measures mean detection confidence on OVB classes relative to the base model (upper bound = 1.0). From the sweep evaluation:

| Model | Retention | OV Detections |
|-------|-----------|---------------|
| Base | 1.000 | Normal |
| Fine-tuned (5 epoch) | 0.358 | Reduced but present |
| Fine-tuned (80 epoch, best) | 0.000 | Zero detections on all OV classes |

The 80-epoch model produces **zero detections** across all 6 monitored open-vocab classes (backpack, dog, foreign objects, landing pad, soccer ball). This confirms total open-vocabulary collapse.

## Learning Rate Sensitivity

Two learning rates were tested with early stopping based on open-vocab ratio monitoring:

| LR | Epochs Before Stop | Best VisDrone mAP@50 | OV Ratio at Stop | Stop Reason |
|----|-------------------|---------------------|------------------|-------------|
| 1e-4 | 3 | 0.472 | 0.93 | No OV improvement for 3 epochs |
| 5e-5 | 2 | 0.442 | 0.60 | OV ratio dropped below 70% threshold |

Both learning rates show rapid VisDrone improvement in the first 1-2 epochs, with open-vocab degradation beginning by epoch 2-3.

## Key Findings

1. **Frozen CLIP encoder is essential.** Keeping the text encoder frozen during fine-tuning is the primary mechanism for preserving open-vocabulary capability. Only the detection head and neck are updated.

2. **Short training is optimal.** 5 epochs provides the best trade-off between VisDrone improvement and open-vocab retention. Beyond ~5 epochs, the model increasingly specializes to VisDrone classes at the expense of novel class detection.

3. **Open-vocab collapse is progressive.** Performance degrades gradually from 5 to 40 epochs, then is effectively zero by 80 epochs. This is not a sharp cliff but a steady decline.

4. **Descriptive prompts enable general obstacle detection.** Prompts like "foreign object on driveway" and "foreign object on orange circle landing pad" allow the model to detect objects outside any predefined class list, which is critical for UAS safety applications.

5. **No speed penalty.** Fine-tuning does not degrade inference speed; all models run at 87-96 FPS on the evaluation hardware.

## Issues Encountered

### Overfitting to VisDrone classes with extended training

The longer we fine-tuned on VisDrone, the more the model would prefer to detect only VisDrone classes and ignore prompted open-vocab classes.

Solutions:
- Three durations were evaluated (5, 40, 80 epochs). The 5-epoch variant showed the best balance.
- During fine-tuning, the CLIP encoder's weights were frozen to prevent them from changing. Only the detection head/neck of the YOLO-World architecture were updated.

### Unable to detect general obstacles not explicitly prompted for

Solutions:
- Use of descriptive, general prompts to detect "foreign" objects in relevant areas.
