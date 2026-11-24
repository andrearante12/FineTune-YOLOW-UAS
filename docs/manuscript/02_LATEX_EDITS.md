# LaTeX edits for the final manuscript

Section by section, in paper order. Each block is either a **replacement** (says what it
replaces) or an **insertion** (says where it goes). Numbers match `01_RESULTS.md`. You can cut
tables to fit the page budget; the minimum set is Tables 1–3, R7 (retention methods), R11
(obstruction) and Thor.

New `\label`s used: `tab:lrgrid`, `tab:methods`, `tab:perclassovb`, `tab:obstruction`,
`fig:pareto`, `fig:qual4`, `tab:thor` (replaces `tab:jetson`).

---

## Title (optional)

The current title still fits. If you want the retention result in it:

```latex
\title{Open-Vocabulary Obstruction Detection for UAS Landing: Fine-Tuning YOLO-World Without Losing the Open Vocabulary}
```

## Abstract (replace entirely)

```latex
\begin{abstract}
Safe autonomous landing remains a critical challenge for unmanned aerial systems (UAS)
performing package delivery, where undetected landing zone obstructions pose significant
dangers. Closed-set object detectors cannot recognize obstacles outside their training
categories, while open-vocabulary detectors such as YOLO-World can, but must be adapted to
top-down aerial imagery. This paper characterizes the tradeoff between aerial accuracy and
open-vocabulary retention when fine-tuning YOLO-World on the VisDrone dataset with its CLIP
text encoder frozen. We show that standard fine-tuning significantly degrades detection of
unseen categories within 5 epochs and turns the model into an effectively closed-set
detector by 40 epochs, and that this loss is governed by the learning rate rather than the
training duration or layer freezing. Combining a low learning rate with weight-space
ensembling (WiSE-FT) yields a model that improves VisDrone mAP@0.5 by 137\% over the
pretrained model while retaining 77--78\% of its zero-shot accuracy on LVIS and COCO and
slightly exceeding it on novel UAS obstruction categories. In an image-level landing-pad
obstruction test, this model matches the pretrained model's discrimination (AUROC 0.86)
with a 3.6$\times$ lower false-alarm rate. TensorRT FP16 inference runs at 56 FPS at
1280$\times$1280 on an NVIDIA Jetson AGX Thor with no accuracy loss, alongside real-time
results on Jetson Xavier and Orin Nano.
\end{abstract}
```

## Introduction: replace the contributions paragraph

```latex
This paper makes the following contributions. First, we characterize how fine-tuning
YOLO-World on the VisDrone aerial dataset, with the CLIP text encoder frozen, trades domain
accuracy for open-vocabulary retention, using controlled experiments over training duration,
learning rate, and layer freezing, evaluated on a custom UAS Open-Vocabulary Benchmark (OVB)
and on the standard LVIS, COCO, and ODinW benchmarks. We find that the learning rate, not the
number of epochs, governs forgetting. Second, we show that a low learning rate combined with
weight-space ensembling (WiSE-FT) recovers, and on novel UAS categories slightly exceeds, the
pretrained model's open-vocabulary accuracy at more than twice its VisDrone accuracy. Third, we
evaluate whether descriptive noun-phrase prompts such as ``foreign object on orange circle
landing pad'' can flag obstructions that are not explicitly prompted, at both the object level
and as an image-level landing decision. Fourth, we benchmark real-time inference on NVIDIA
Jetson Xavier, Orin Nano, and AGX Thor, including TensorRT deployment.
```

In the paragraph before it, the sentence starting "It has been observed that fine-tuning the CLIP
encoder…" can stay. Optionally add after it:
```latex
We find that even with the text encoder frozen, fine-tuning the visual components alone is
enough to erode open-vocabulary generalization.
```

## Related Works: add a WiSE-FT sentence

At the end of "Fine-Tuning Open-Vocabulary Models for Aerial Domains", replace the last
sentence ("This work instead prevents encoder degradation by freezing…") with:

```latex
This work keeps the {CLIP} text encoder frozen throughout fine-tuning and, for the visual
components, adopts weight-space ensembling (WiSE-FT)~\cite{wortsman2022robust}, which
linearly interpolates between pretrained and fine-tuned weights and was shown to preserve the
robustness of zero-shot CLIP models under fine-tuning without additional training cost.
```

---

## Methodology

### Methods intro paragraph (replace)

```latex
This section describes the experimental methodology for fine-tuning YOLO-World on the
VisDrone aerial dataset while preserving open-vocabulary generalization. The {CLIP} text
encoder is kept frozen throughout training, while the visual backbone, neck, and detection
head are updated. We vary training duration, learning rate, and the set of trainable layers,
and evaluate weight-space ensembling as a post-hoc retention method. Every model is evaluated
on both VisDrone and a custom Open-Vocabulary Benchmark (OVB), as well as on standard
zero-shot benchmarks, to characterize the specialization-retention tradeoff.
```

### CLIP Encoder Freezing Strategy (replace subsection) — Fix 3

```latex
\subsection{CLIP Encoder Freezing Strategy}

The CLIP text encoder is kept frozen throughout fine-tuning: class prompts are encoded once
by the pretrained encoder and the resulting text embeddings are held fixed, while the image
backbone, RepVL-PAN neck, and detection head are updated. This preserves the pretrained
language embedding space shared by all prompts, seen and unseen, while allowing the visual
components to adapt to aerial imagery.
```

### OVB subsection (replace) — Fix 1

```latex
\subsection{Open-Vocabulary Benchmark (OVB) Evaluation Dataset}

To measure how well fine-tuning preserves the model's open-vocabulary capabilities, we
construct an Open-Vocabulary Benchmark (OVB). OVB consists of 195 images captured with our own
UAS over a residential driveway and manually annotated with eight categories: basic
obstructions and landing-site objects (backpack, dog, soccer ball, orange circle landing pad),
descriptive noun phrases (foreign object on orange circle landing pad, foreign object on
driveway), and two categories shared with VisDrone (car, pedestrian). The six categories absent
from VisDrone form the novel subset, whose mAP is our primary open-vocabulary metric; the two
shared categories improve with fine-tuning and would otherwise mask the loss on unseen ones.
```

### New subsection: Standard Benchmarks and Statistical Protocol (insert after OVB)

```latex
\subsection{Standard Zero-Shot Benchmarks and Statistical Protocol}

Because OVB is small and specific to our landing scenario, we also evaluate every model
zero-shot on three standard benchmarks at 640$\times$640: LVIS v1 minival (4,809 images,
1,203 categories)~\cite{gupta2019lvis}, the benchmark used to evaluate YOLO-World; COCO
val2017~\cite{lin2014coco}; and the AerialMaritimeDrone set from ODinW~\cite{li2022glip},
an aerial dataset whose categories (boat, dock, jet ski, lift) do not appear in VisDrone. Our
absolute LVIS scores are lower than those published for YOLO-World because of evaluation
differences (single prompt string per category, standard rather than fixed AP), so we report
each model relative to the pretrained model under an identical protocol.

OVB results are reported with 95\% bootstrap confidence intervals over images (1,000
resamples)~\cite{efron1994bootstrap}. Differences between a fine-tuned model and the
pretrained model are tested with a paired bootstrap, scoring both models on the same image
resamples. The main recipes are repeated with three random seeds and reported as mean
$\pm$ standard deviation.
```

### Training Durations (replace subsection) — Fix 2

```latex
\subsection{Training Duration, Learning Rate, and Layer Freezing}

All runs share the hyperparameters in Table~\ref{tab:hyperparams} unless stated otherwise,
using the AdamW optimizer with an initial learning rate of $7.14\times10^{-4}$. Using this
configuration, we train for 5 and 40 epochs to
monitor how open-vocabulary degradation progresses with training duration. We then train
for 10 epochs at learning rates from $2.5\times10^{-5}$ to $1.5\times10^{-3}$, train for 40
epochs at $10^{-4}$, and train with the backbone frozen (neck and head trainable) or with
only the detection head trainable.
Performance is assessed using a dual-dataset evaluation framework, measuring both VisDrone
mAP on the VisDrone validation set and OVB mAP on the OVB dataset simultaneously, an approach
analogous to the dual-task evaluation employed by a fine-tuning
study~\cite{gandhi2025finetuningforgettingadaptationyolov8} and the base/novel class
evaluation protocol established in open-vocabulary detection
literature~\cite{zareian2021ovrcnn}.
```

### Hyperparameter table (replace) — Fix 2

```latex
\begin{table}[hbt!]
\caption{Fine-tuning hyperparameters shared by all runs unless stated otherwise.}
\label{tab:hyperparams}
\centering
\begin{tabular}{ll}
\hline
\textbf{Parameter} & \textbf{Value} \\
\hline
Base model                       & YOLOv8m-WorldV2 \\
Optimizer                        & AdamW ($\beta_1 = 0.9$) \\
Learning rate (lr0)              & $7.14 \times 10^{-4}$ \\
Final LR factor (lrf), schedule  & 0.01, linear decay \\
Weight decay                     & 0.05 \\
Batch size                       & 24 \\
Image size                       & $1280 \times 1280$ \\
Warmup epochs                    & 3 \\
Mosaic augmentation              & disabled for final 10 epochs \\
Mixed precision                  & FP16 (AMP) \\
\hline
\end{tabular}
\end{table}
```

### New subsection: Weight-Space Ensembling (insert after Training Durations)

```latex
\subsection{Weight-Space Ensembling (WiSE-FT)}

As a post-hoc retention method, we interpolate between the pretrained weights
$\theta_{\text{base}}$ and the fine-tuned weights $\theta_{\text{ft}}$,
\begin{equation}
\theta(\alpha) = (1-\alpha)\,\theta_{\text{base}} + \alpha\,\theta_{\text{ft}},
\end{equation}
following WiSE-FT~\cite{wortsman2022robust}. The interpolated model has the same architecture
and inference cost as either endpoint. To keep OVB a held-out test set, $\alpha$ is selected on
VisDrone alone: we choose the smallest $\alpha$ whose VisDrone mAP is within 0.01 of the fully
fine-tuned model ($\alpha = 1$). This rule was fixed before the multi-seed experiments.
```

---

## Results

### Main table (replace `tab:results`) — Fix 1

```latex
\begin{table}[hbt!]
\caption{Performance across fine-tuning durations on the VisDrone validation set and the
         Open-Vocabulary Benchmark (OVB). All mAP values are COCO mAP@[.5:.95] unless noted.
         OVB novel excludes the two OVB classes that also appear in VisDrone (car,
         pedestrian). Both fine-tuned models use the hyperparameters in
         Table~\ref{tab:hyperparams}.}
\label{tab:results}
\centering
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{VisDrone mAP} & \textbf{VisDrone mAP@50}
& \textbf{OVB mAP} & \textbf{OVB novel mAP} \\
\hline
Base (YOLOv8m-WorldV2) & 0.131 & 0.198 & 0.263 & 0.222 \\
Fine-tuned (5 epoch)   & 0.308 & 0.492 & 0.196 & 0.129 \\
Fine-tuned (40 epoch)  & 0.341 & 0.538 & 0.037 & 0.007 \\
\hline
\end{tabular}
\end{table}
```

### Quantitative results text (replace the two paragraphs after the table) — Fix 1

```latex
Fine-tuning on VisDrone produces substantial gains in domain-specific detection accuracy.
The 5-epoch checkpoint achieves a VisDrone mAP of 0.308, a +135\% improvement over the base
model, and a mAP@50 of 0.492, a +149\% improvement. Extending training to 40 epochs under
identical hyperparameters raises VisDrone mAP further to 0.341. Per-class results in
Table~\ref{tab:perclass} show consistent improvement across all 10 VisDrone categories, with
the most notable gain on the motor class, which improves from 0.000 in the base model to 0.303
after fine-tuning, indicating that zero-shot detection of certain aerial categories fails
entirely without domain adaptation.

The OVB results reveal open-vocabulary capability loss that begins early and grows with
training duration. After only 5 epochs, novel-category OVB mAP falls from 0.222 to 0.129, a
42\% reduction; a paired bootstrap places the 95\% confidence interval of this drop at
$[-0.124, -0.070]$, confirming the loss is significant despite the small benchmark. Three
repetitions of the 5-epoch run with different seeds give $0.125 \pm 0.021$. On the standard
benchmarks the loss is larger still: the 5-epoch model retains only 19\% of the base model's
LVIS AP and 18\% of its COCO AP. After 40 epochs, novel OVB mAP collapses to 0.007 and LVIS AP
to 1\% of base, effectively reducing the model to a closed-set VisDrone detector.
```

### Per-class VisDrone table (`tab:perclass`): keep as-is

### New subsection: Learning Rate, Duration, and Freezing (insert after per-class table)

```latex
\subsection{Learning Rate, Training Duration, and Layer Freezing}

Table~\ref{tab:lrgrid} varies the learning rate at a fixed 10-epoch budget. Learning rate acts
as a smooth dial between VisDrone accuracy and open-vocabulary retention up to the
$7.14\times10^{-4}$ used above; beyond it, open-vocabulary detection collapses entirely with no
further VisDrone gain. Novel-category mAP tracks the current learning rate rather than the
epoch count: in the 5-epoch run it drops in a single step as the warmup reaches its peak
learning rate (0.229 after epoch 1, 0.132 after epoch 2), then stays flat while VisDrone
accuracy continues to rise.

A low learning rate avoids the long-training collapse rather than delaying it. At
$10^{-4}$, 40 epochs reach a VisDrone mAP of $0.337 \pm 0.001$, matching the 40-epoch run at
the higher learning rate (0.341), while retaining a novel OVB mAP of $0.165 \pm 0.005$ (three
seeds), 24 times higher. Layer freezing is less effective: with the backbone frozen, novel mAP
after 10 epochs is still 0.125 (vs.\ 0.016 unfrozen), but it falls to 0.001 by epoch 40 while
VisDrone accuracy is 0.025 lower. Training only the detection head preserves the most general
ability (76\% of base COCO AP) but limits VisDrone mAP to 0.227.

\begin{table}[hbt!]
\caption{Effect of learning rate at a fixed 10-epoch budget (AdamW, single seed).}
\label{tab:lrgrid}
\centering
\begin{tabular}{lcccc}
\hline
\textbf{Learning rate} & \textbf{VisDrone mAP} & \textbf{VisDrone mAP@50} & \textbf{OVB novel mAP} & \textbf{OVB novel mAP@50} \\
\hline
$2.5\times10^{-5}$ & 0.271 & 0.431 & 0.181 & 0.374 \\
$1\times10^{-4}$   & 0.304 & 0.484 & 0.202 & 0.371 \\
$2\times10^{-4}$   & 0.319 & 0.504 & 0.172 & 0.288 \\
$7.14\times10^{-4}$ & 0.327 & 0.519 & 0.131 & 0.201 \\
$1.5\times10^{-3}$ & 0.323 & 0.510 & 0.000 & 0.000 \\
\hline
\end{tabular}
\end{table}
```

### New subsection: Retention Methods (insert next)

```latex
\subsection{Retention Methods}

Table~\ref{tab:methods} compares retention strategies. Weight-space ensembling recovers most
of the open-vocabulary loss at no training cost: interpolating the 5-epoch model with the base
model at $\alpha = 0.7$ (selected on VisDrone only, and identical across all seeds) costs 0.003
VisDrone mAP and raises novel OVB mAP from 0.125 to 0.209, with LVIS and COCO retention rising
from 18--19\% to 55--59\% of base. The two strategies stack. A model trained for 10 epochs at
$10^{-4}$ and then interpolated at $\alpha = 0.8$ reaches a novel OVB mAP of
$0.253 \pm 0.013$, slightly above the pretrained model (paired bootstrap 95\% CI of the
difference $[+0.004, +0.046]$), while retaining 77\% of base LVIS AP and 78\% of base COCO AP
and more than doubling VisDrone mAP (0.298, mAP@50 0.469). We adopt this model as our
recommended configuration. Fig.~\ref{fig:pareto} shows all trained and interpolated models in
the VisDrone/open-vocabulary plane.

On the aerial ODinW categories absent from VisDrone, short or low-learning-rate fine-tuning
improves zero-shot detection (novel AP 0.034 for the base model vs.\ 0.057--0.064), whereas
40-epoch training at the higher learning rate loses this gain (0.014), suggesting that
moderate VisDrone fine-tuning learns aerial features that transfer to unseen aerial objects.
This set contains only 74 images, so we treat the result as directional.

\begin{table}[hbt!]
\caption{Retention methods. VisDrone and OVB novel mAP@[.5:.95]; $\pm$ is the standard
         deviation over three seeds. LVIS and COCO are zero-shot AP, with percent of the base
         model in parentheses (single seed). ODinW is novel-category AP on AerialMaritimeDrone.}
\label{tab:methods}
\centering
\begin{tabular}{lccccc}
\hline
\textbf{Method} & \textbf{VisDrone} & \textbf{OVB novel} & \textbf{LVIS} & \textbf{COCO} & \textbf{ODinW} \\
\hline
Base (no fine-tuning)         & 0.131 & 0.222 & 0.162 (100\%) & 0.419 (100\%) & 0.034 \\
Fine-tune, 5 ep               & 0.302 $\pm$ 0.004 & 0.125 $\pm$ 0.021 & 0.031 (19\%) & 0.074 (18\%) & 0.061 \\
Fine-tune, 40 ep              & 0.341 & 0.007 & 0.001 (1\%) & 0.007 (2\%) & 0.014 \\
Head only, 5 ep               & 0.227 & 0.209 & 0.094 (58\%) & 0.320 (76\%) & 0.034 \\
Low LR, 10 ep                 & 0.304 $\pm$ 0.001 & 0.215 $\pm$ 0.007 & 0.080 (49\%) & 0.246 (59\%) & 0.057 \\
Low LR, 40 ep                 & 0.337 $\pm$ 0.001 & 0.165 $\pm$ 0.005 & 0.039 (24\%) & 0.124 (30\%) & 0.047 \\
5 ep + WiSE-FT ($\alpha$=0.7)  & 0.299 $\pm$ 0.002 & 0.209 $\pm$ 0.031 & 0.088 (55\%) & 0.248 (59\%) & 0.064 \\
\textbf{Low LR 10 ep + WiSE-FT ($\alpha$=0.8)} & \textbf{0.298 $\pm$ 0.000} & \textbf{0.253 $\pm$ 0.013} & \textbf{0.125 (77\%)} & \textbf{0.325 (78\%)} & 0.060 \\
\hline
\end{tabular}
\end{table}

\begin{figure}[hbt!]
\centering
\includegraphics[width=0.75\textwidth]{pareto_novel.png}
\caption{VisDrone mAP versus novel-category OVB mAP for all fine-tuned and weight-interpolated
         models. Lower learning rates and weight-space ensembling move the frontier toward the
         upper right.}
\label{fig:pareto}
\end{figure}
```

### New subsection: Descriptive Prompts and the Landing Decision (insert next)

```latex
\subsection{Descriptive Prompts and the Landing Decision}

Table~\ref{tab:perclassovb} breaks OVB down by category. Concrete object prompts recover or
improve under the recommended recipe (backpack 0.076 to 0.174, landing pad 0.398 to 0.577),
but the descriptive prompt ``foreign object on orange circle landing pad'' is the most fragile
category, falling from 0.477 to 0.261. The prompt ``foreign object on driveway'' is near zero
for every model, including the base model.

To test whether descriptive prompts catch objects that are not explicitly prompted, we
removed the backpack, dog, and soccer ball prompts and measured how many of those 200 objects
were found by a ``foreign object'' detection at confidence 0.1. Such detections rarely
localize the object itself (at most 11\% at IoU $\geq$ 0.5 for any model); instead, they flag the
obstructed pad region. 84\% of backpacks, which lie on the pad, are covered by a base-model
``foreign object on orange circle landing pad'' detection, whereas dogs and soccer balls
elsewhere in the scene are missed. Descriptive prompts therefore act as a region-level
obstruction signal, and specific prompts remain necessary for localization; the recommended
model has the highest specific-prompt recall (59\% vs.\ 53\% for the base model).

Because the landing decision is ultimately binary, we also evaluate an image-level task:
distinguishing the 91 OVB images with an obstructed pad from the 101 with a clear pad
(Table~\ref{tab:obstruction}). Scored by the descriptive prompt alone, the base model
discriminates best (AUROC 0.835) and every fine-tuned model is worse, mirroring the per-class
results. When object prompts detected inside the landing pad are also counted, the
recommended model matches the base model (AUROC 0.861 vs.\ 0.866) at the same miss rate and
with 3\% instead of 11\% false alarms at confidence 0.25, while the 5-epoch model reaches only
0.696. This combined score was defined after observing the descriptive-only results and should
be read as exploratory.

\begin{table}[hbt!]
\caption{Per-category OVB AP@[.5:.95]. FO = foreign object. Car (one ground-truth box) is omitted.}
\label{tab:perclassovb}
\centering
\begin{tabular}{lccccccc}
\hline
\textbf{Model} & \textbf{backpack} & \textbf{dog} & \textbf{ball} & \textbf{pad} & \textbf{FO on pad} & \textbf{FO on driveway} & \textbf{pedestrian} \\
\hline
Base             & 0.076 & 0.360 & 0.016 & 0.398 & 0.477 & 0.005 & 0.275 \\
Fine-tune, 5 ep  & 0.014 & 0.011 & 0.013 & 0.424 & 0.310 & 0.001 & 0.347 \\
Fine-tune, 40 ep & 0.000 & 0.000 & 0.000 & 0.042 & 0.000 & 0.000 & 0.216 \\
5 ep + WiSE-FT   & 0.050 & 0.178 & 0.025 & 0.547 & 0.451 & 0.001 & 0.379 \\
Recommended      & 0.174 & 0.423 & 0.017 & 0.577 & 0.261 & 0.026 & 0.377 \\
\hline
\end{tabular}
\end{table}

\begin{table}[hbt!]
\caption{Image-level landing-pad obstruction detection on OVB (91 obstructed, 101 clear).
         AUROC with 95\% bootstrap CI. Descriptive: maximum confidence of the ``foreign object
         on orange circle landing pad'' prompt. Combined: additionally counts object prompts
         detected inside the landing pad. Miss and false-alarm rates for the combined score at
         confidence 0.25.}
\label{tab:obstruction}
\centering
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{AUROC (descriptive)} & \textbf{AUROC (combined)} & \textbf{Miss} & \textbf{False alarm} \\
\hline
Base             & 0.835 [0.774, 0.893] & 0.866 [0.813, 0.920] & 0.54 & 0.11 \\
Fine-tune, 5 ep  & 0.668 [0.590, 0.746] & 0.696 [0.620, 0.773] & 0.81 & 0.02 \\
Fine-tune, 40 ep & 0.421 [0.354, 0.491] & 0.421 [0.354, 0.491] & 1.00 & 0.00 \\
5 ep + WiSE-FT   & 0.759 [0.687, 0.826] & 0.818 [0.755, 0.874] & 0.76 & 0.01 \\
Recommended      & 0.682 [0.602, 0.756] & 0.861 [0.806, 0.908] & 0.54 & 0.03 \\
\hline
\end{tabular}
\end{table}
```

### Embedded Deployment (replace subsection text and `tab:jetson`)

```latex
\subsection{Embedded Deployment}

The 5-epoch fine-tuned checkpoint was first evaluated for real-time inference on the Jetson
Xavier and the Jetson Orin Nano using the PyTorch runtime at 1280$\times$1280, reaching 9.4 and
12.5 FPS respectively. Embedded inference on {NVIDIA} Jetson platforms has been demonstrated
across a range of real-time computer vision tasks, with the Jetson ecosystem providing a
practical deployment target for {UAS} onboard computing~\cite{mittal2019survey}. We further
benchmarked the models on an NVIDIA Jetson AGX Thor (120~W mode) with PyTorch and TensorRT,
measuring end-to-end latency (preprocessing, inference, and non-maximum suppression) at batch
size 1 over 300--500 frames held in memory, and re-scoring accuracy with each runtime
(Table~\ref{tab:thor}). TensorRT FP16 runs at 56--57 FPS with accuracy identical to PyTorch.
INT8 quantization adds only 2--4\% throughput, because CPU-side preprocessing and
non-maximum suppression dominate the frame time, and it reduces accuracy slightly, so FP16 is
the preferred configuration. The TensorRT engine embeds the prompt vocabulary and must be
rebuilt when the prompt list changes (about 2.5 minutes for FP16).

\begin{table}[hbt!]
\caption{Real-time inference at 1280$\times$1280. Xavier and Orin Nano: 5-epoch model,
         PyTorch. Thor: end-to-end latency, batch 1, 120~W mode.}
\label{tab:thor}
\centering
\begin{tabular}{llcccc}
\hline
\textbf{Platform / model} & \textbf{Runtime} & \textbf{FPS} & \textbf{p99 latency (ms)} & \textbf{VisDrone mAP} & \textbf{OVB novel} \\
\hline
Xavier, 5 ep        & PyTorch        & 9.4  & -- & -- & -- \\
Orin Nano, 5 ep     & PyTorch        & 12.5 & -- & -- & -- \\
Thor, 5 ep          & PyTorch FP32   & 35 & 32.1 & 0.308 & 0.126 \\
Thor, 5 ep          & TensorRT FP16  & 57 & 22.1 & 0.308 & 0.128 \\
Thor, 5 ep          & TensorRT INT8  & 59 & 20.8 & 0.300 & 0.119 \\
Thor, recommended   & TensorRT FP16  & 56 & 22.0 & 0.297 & 0.247 \\
\hline
\end{tabular}
\end{table}
```

Check before submitting: if the Xavier/Orin FPS included reading images from disk, add to the
caption "Xavier and Orin Nano timings include image loading; measured the same way, Thor
PyTorch FP32 reaches 21 FPS." (`01_RESULTS.md` R12.)

Replace the sentence "Both platforms sustain real-time inference speeds… such as NVIDIA Jetson
Thor." with:
```latex
All three platforms sustain real-time inference suitable for onboard UAS deployment, and the
Thor's throughput leaves headroom for running additional onboard models in parallel.
```

### Qualitative figures

Keep the two existing comparison figures. Add:

```latex
\begin{figure}[hbt!]
\centering
\includegraphics[width=0.95\textwidth]{dji0162_frame2076_4models.jpg}
\caption{The same UAS frame processed by (a) the base model, (b) the 5-epoch fine-tuned model,
         (c) the 5-epoch model with WiSE-FT, and (d) the low-learning-rate model. The low-LR
         model detects the landing pad, pedestrian, and soccer ball and flags the bag as a
         foreign object, while the 5-epoch model misses the ball, labels the bag ``car,'' and
         produces spurious ``dog'' detections on shadows.}
\label{fig:qual4}
\end{figure}
```
Check the panel order and labels against the image before using this caption.

---

## Discussion (replace entirely) — Fix 1 + additions

```latex
\section{Discussion}

The results reveal a clear tradeoff between domain specialization and open-vocabulary
retention, but one that depends far more on how the model is fine-tuned than on how long.
Even with the CLIP text encoder frozen, updating the visual backbone, neck, and detection head
at the learning rate selected by default degrades detection of unseen categories within a few
hundred iterations and removes it almost entirely by 40 epochs. Fig.~\ref{fig:backpack_comparison}
illustrates this: detection confidence on the prompted OVB class ``backpack'' decreases after
fine-tuning, while Fig.~\ref{fig:pedestrian_comparison} shows the dramatic improvement on
VisDrone classes such as ``pedestrian.'' Freezing the text encoder is therefore necessary but
not sufficient. Our results point to the size of the update in weight space as the governing
factor: a lower learning rate keeps the visual components close to the pretrained solution
over long training, and interpolating back toward the pretrained weights recovers most of what
is lost. Layer freezing, by contrast, restricts which weights move but not how far, and only
delays the collapse.

The benchmark used to measure retention matters. OVB retention overstated general retention:
the 5-epoch model kept 58\% of base novel OVB mAP but only 18--19\% of base LVIS and COCO AP. The
recommended model retains 77--78\% on LVIS and COCO and slightly exceeds the base model on
novel OVB categories, so the gain is not specific to our benchmark. Descriptive noun phrases
are the most fragile capability. Concrete object prompts recover under the recommended recipe,
but ``foreign object on orange circle landing pad'' does not, and ``foreign object on
driveway'' fails for every model. In practice the descriptive prompt behaves as a region-level
signal that the pad is occupied rather than as a detector of arbitrary objects, and it is most
useful when combined with an exhaustive list of specific hazard prompts. Under that
combination the recommended model matches the base model at deciding whether the pad is
obstructed while raising far fewer false alarms.

Several limitations bound these findings. The 195-image OVB set is small and comes from a
single site; we mitigate this with bootstrap confidence intervals, three-seed repetitions,
novel-only reporting, and standard benchmarks, but absolute OVB values should be interpreted
with caution. Standard-benchmark and learning-rate-grid results are single-seed. The
image-level obstruction labels are derived from the OVB annotations, and the combined
obstruction score was defined after inspecting the results; at confidence 0.25 all models
miss more than half of obstructed pads, so an operational system would require a lower,
calibrated threshold validated on substantially more data. We did not compare against
closed-set detectors trained on the same data, and the text encoder could not be partially
unfrozen in our training framework.

A practical limitation of this approach is that target classes must be prompted before
inference, requiring prior knowledge of potential obstructions. This can be partially
addressed using an exhaustive prompt list of common hazard categories combined with
descriptive prompts for the landing pad region. Future work will explore using
vision-language models, either cloud-based or compact onboard models such as
LiteVLM~\cite{huang2025litevlmlowlatencyvisionlanguagemodel}, to dynamically generate prompts
from a scene prior to YOLO-World inference, as well as parameter-efficient fine-tuning and
replay of pretraining data as further retention strategies.
```

## Expected Outcomes — Fix 4

Delete the whole `\section{Expected Outcomes}` (both paragraphs).

## Conclusion (replace)

```latex
\section{Conclusion}

This paper presented a fine-tuning methodology for adapting YOLO-World to UAS landing zone
obstruction detection and characterized the tradeoff between VisDrone accuracy and
open-vocabulary retention. Freezing the CLIP text encoder alone is not sufficient: standard
fine-tuning significantly degrades detection of unseen categories within 5 epochs and
eliminates it by 40. The learning rate, rather than training duration or layer freezing,
governs this loss. A low learning rate combined with weight-space ensembling yields a model
that more than doubles VisDrone accuracy (+137\% mAP@0.5) while retaining 77--78\% of the
pretrained model's zero-shot accuracy on LVIS and COCO and slightly exceeding it on novel UAS
obstruction categories. Combined with specific hazard prompts, its descriptive landing-pad
prompt identifies obstructed landing pads as well as the pretrained model with a
3.6$\times$ lower false-alarm rate, and TensorRT deployment reaches 56 FPS at full resolution
on an NVIDIA Jetson AGX Thor. These results establish a practical foundation for
open-vocabulary obstruction detection in autonomous UAS landing and motivate future work on
dynamic prompt generation via vision-language model integration.
```

---

## New bibliography entries (add to `sample.bib`)

```bibtex
@inproceedings{wortsman2022robust,
  title     = {Robust Fine-Tuning of Zero-Shot Models},
  author    = {Wortsman, Mitchell and Ilharco, Gabriel and Kim, Jong Wook and Li, Mike and Kornblith, Simon and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Hajishirzi, Hannaneh and Farhadi, Ali and Namkoong, Hongseok and Schmidt, Ludwig},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {7959--7971},
  year      = {2022}
}

@inproceedings{gupta2019lvis,
  title     = {{LVIS}: A Dataset for Large Vocabulary Instance Segmentation},
  author    = {Gupta, Agrim and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {5356--5364},
  year      = {2019}
}

@inproceedings{lin2014coco,
  title     = {Microsoft {COCO}: Common Objects in Context},
  author    = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C. Lawrence},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {740--755},
  year      = {2014}
}

@inproceedings{li2022glip,
  title     = {Grounded Language-Image Pre-training},
  author    = {Li, Liunian Harold and Zhang, Pengchuan and Zhang, Haotian and Yang, Jianwei and Li, Chunyuan and Zhong, Yiwu and Wang, Lijuan and Yuan, Lu and Zhang, Lei and Hwang, Jenq-Neng and Chang, Kai-Wei and Gao, Jianfeng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10965--10975},
  year      = {2022}
}

@book{efron1994bootstrap,
  title     = {An Introduction to the Bootstrap},
  author    = {Efron, Bradley and Tibshirani, Robert J.},
  publisher = {Chapman and Hall/CRC},
  year      = {1994}
}
```

Double-check these entries against the publishers' pages before submitting. They are from
memory.
