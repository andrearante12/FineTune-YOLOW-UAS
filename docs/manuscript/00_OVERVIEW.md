# Final manuscript package — overview

AIAA final manuscript, due **2026-12-01**. Built from the extended abstract that was accepted
(PDF in repo root) and all Jetson Thor experiments (Oct 30 – Nov 23 2026).

| File | What's in it |
|---|---|
| `00_OVERVIEW.md` (this) | The new story, what changed vs. the abstract, the claims you can make, a checklist |
| `01_RESULTS.md` | Every number, table and figure the paper needs, with caveats and source paths |
| `02_LATEX_EDITS.md` | Section-by-section LaTeX: the 4 must-fixes plus every addition, ready to paste |

Full experiment index: `EXPERIMENTS.md` (this folder). Chronological log: `docs/THOR_PROGRESS.md`.
Raw metrics: `eval_results/`.

## The story in one paragraph

Fine-tuning YOLO-World on VisDrone more than doubles aerial detection accuracy but erodes
open-vocabulary ability, and the extended abstract measured that erosion wrongly (an
evaluator bug) and with a confounded design (the optimizer changed with training length).
With both fixed, **learning rate, not epoch count or layer freezing, controls forgetting**.
A low learning rate plus weight-space ensembling (WiSE-FT) keeps 77–78% of the base model's
general open-vocab accuracy (LVIS/COCO, vs. 18–19% for the abstract's model) and slightly
*beats* the base model on novel UAS obstruction classes, at +127% VisDrone mAP and 56 FPS on
Jetson Thor. The descriptive "foreign object" prompt is the most fragile capability. On its
own it degrades under every fine-tuning recipe. Combined with object prompts inside the
detected landing pad, the recommended model matches the base model at telling an obstructed
pad from a clear one, with about 3.6× fewer false alarms.

## What changed vs. the extended abstract

| Abstract said | Now |
|---|---|
| OVB mAP 0.034 for base and 5-epoch → "no measurable degradation" | Evaluator bug. Base 0.263 / novel 0.222; 5-epoch 0.196 / novel 0.129, a **significant 42% novel loss** (paired CI [−0.124, −0.070]) |
| 5 / 40 / 80 epochs, "identical hyperparameters", lr 2e-4 | `optimizer=auto` ignored lr0. 5 epochs ran AdamW 7.14e-4, and longer runs would switch to SGD 0.01. Re-ran 40 epochs with the optimizer pinned; 80 dropped (40 already shows the collapse and the plateau) |
| 40 ep VisDrone 0.298, 80 ep 0.330 | Controlled 40 ep: **0.341**, novel OV 0.007, LVIS/COCO 1–2% of base |
| Weight updates restricted to head and neck | Backbone, neck and head all train; CLIP text encoder frozen |
| Expected outcomes: lower LR, partial freezing, LVIS | Done: LR grid, backbone/head freezing, LVIS + COCO + ODinW, WiSE-FT, seeds, bootstrap |
| Xavier 9.4 / Orin Nano 12.5 FPS | Add Thor: TensorRT FP16 **56–57 FPS** at 1280, lossless |
| Descriptive prompts catch unprompted obstructions | Holds for objects **on the pad** (flagged as an obstructed pad region), not for loose objects elsewhere; the driveway prompt fails for every model, base included |

## Claims you can make (with the evidence)

1. **Fine-tuning significantly degrades open-vocabulary detection, even with CLIP frozen.**
   5-epoch novel OVB 0.222 → 0.129 (paired CI excludes 0); LVIS 19% / COCO 18% of base.
2. **Long training at the abstract's LR makes it a closed-set detector.** 40 ep: novel 0.007,
   LVIS 1%.
3. **Learning rate is the control knob.** 10-epoch LR grid: smooth VisDrone↔OV tradeoff up
   to 7.14e-4, total OV loss with no VisDrone gain above it. At lr 1e-4, 40 epochs reach VisDrone
   0.337 ± 0.001 with novel 0.165 ± 0.005 (3 seeds), 24× the novel mAP of the paper-LR 40-epoch run.
4. **Freezing only delays the collapse.** A frozen backbone at 40 epochs still collapses
   (novel 0.001); head-only keeps OV but reaches only VisDrone 0.227.
5. **Augmentation is not the confound.** Mosaic on for all 5 epochs: novel 0.110 ± 0.015 vs.
   0.125 ± 0.021 without (3 seeds each). Small and within noise, nowhere near the 40-epoch collapse.
6. **WiSE-FT recovers most of the loss and stacks with low LR.** Recommended model (lr 1e-4,
   10 ep, head biases restored, WiSE α=0.8 picked on VisDrone only): VisDrone 0.298 ± 0.000,
   novel OVB 0.253 ± 0.013 (3 seeds), **significantly above base** (paired CI [+0.004, +0.046]),
   LVIS 77% / COCO 78% of base.
7. **Benchmark choice matters.** OVB retention overstates general retention (5-epoch: 58% on OVB
   novel vs. 18–19% on LVIS/COCO). Report both.
8. **Domain transfer.** Short/low-LR VisDrone fine-tuning improves zero-shot detection of unseen
   *aerial* classes (ODinW AerialMaritimeDrone novel 0.034 → ~0.06; small set, directional).
9. **Descriptive prompts are the most fragile capability.** Per-class AP of "foreign object on
   pad" drops for every fine-tuned model (base 0.477; recommended 0.261; WiSE α0.7 keeps 0.451)
   while concrete nouns (backpack, dog, pad) recover or improve.
10. **Task-level: the recommended model matches base at deciding "is the pad obstructed?"**
    AUROC 0.861 vs. 0.866 when object prompts inside the detected pad are combined with the
    descriptive prompt, with a 3% vs. 11% false-alarm rate at the same 54% miss rate (conf 0.25).
    The abstract's 5-epoch model: 0.696. *Disclose that this combined score was defined after
    seeing the descriptive-only result.*
11. **Real-time on Thor.** TensorRT FP16, 1280 px: 56 FPS for the recommended model, accuracy
    identical to PyTorch; INT8 adds only 2–4% FPS and costs accuracy.

## Caveats that must appear in the paper

- OVB is small (195 images, 8 classes). "car" has one box, so lead with **novel-only** mAP.
  Bootstrap CIs and 3 seeds mitigate this.
- Absolute LVIS numbers sit below YOLO-World's published ones (ultralytics port, one prompt
  string per class). Quote **% of base**. The ranking holds under the fixed-AP protocol too.
- ODinW aerial set is 74 images: directional evidence only.
- Single seed for LVIS/COCO/ODinW and for the LR grid; 3 seeds for the headline recipes.
- Obstruction-decision analysis: a derived image-level label (pad with vs. without a
  foreign-object box), 91 vs. 101 images; the combined score is post hoc.
- High miss rates at conf 0.25 mean a deployment would use a lower threshold. AUROC is the
  threshold-free comparison.
- Not done (move to future work or drop): YOLOv8 / RT-DETR closed-set comparison with Vu et al.,
  LoRA / replay, VLM prompt generation.

## Checklist

- [ ] Fix 1: replace all OVB numbers (Table 3, abstract, results, discussion, conclusion)
- [ ] Fix 2: hyperparameter table + pinned-optimizer paragraph (+ mosaic row)
- [ ] Fix 3: "CLIP frozen; backbone, neck, head trained" (abstract, methods, conclusion)
- [ ] Fix 4: delete Expected Outcomes
- [ ] Update contributions (intro) and abstract to the new story
- [ ] Methods: evaluation protocol (novel-only OVB, LVIS/COCO/ODinW, bootstrap, seeds), WiSE-FT
- [ ] Results: LR × epoch, freezing, mosaic control, retention methods, per-class OVB, obstruction decision, Thor
- [ ] Figures: Pareto plot, 4-model qualitative frame (keep the two existing comparison figures)
- [ ] Discussion + limitations + conclusion rewritten
- [ ] Bib: WiSE-FT, LVIS, ODinW/GLIP, COCO, bootstrap (entries in `02_LATEX_EDITS.md`)
- [ ] Check whether the abstract's 40/80-epoch figure captions or text still mention 80 epochs
