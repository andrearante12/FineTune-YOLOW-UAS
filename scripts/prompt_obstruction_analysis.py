"""Two manuscript analyses on the OVB set that need no training.

1. Generic-prompt recall: run each model with only the generic vocabulary (landing pad,
   pedestrian, car, and the two "foreign object ..." prompts; no backpack / dog / soccer
   ball prompt) and measure how many backpack / dog / soccer-ball boxes are still found.
   Reference: recall of the same boxes when their specific prompt *is* given.
2. Image-level "is the landing pad obstructed?": positive = image has a "foreign object on
   orange circle landing pad" box, negative = image has a clear landing pad box and no
   foreign-object-on-pad box. Score = max confidence of the "foreign object on orange
   circle landing pad" prompt. AUROC (bootstrap CI), TPR at fixed FPR, miss / false-alarm
   rates at fixed confidence thresholds.

Inference runs once per (model, vocabulary) and is cached as COCO-style JSON, so the
metrics can be recomputed without re-running models. CPU by default (training owns the GPU).

Usage:
  python scripts/prompt_obstruction_analysis.py --device cpu --threads 6
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

PAD = "orange circle landing pad"
FO_PAD = "foreign object on orange circle landing pad"
FO_DRIVE = "foreign object on driveway"
HELD_OUT = ["backpack", "dog", "soccer ball"]
FULL_VOCAB = ["backpack", "car", "dog", FO_DRIVE, FO_PAD, PAD, "pedestrian", "soccer ball"]
GENERIC_VOCAB = [PAD, "pedestrian", "car", FO_PAD, FO_DRIVE]
FO_CLASSES = {FO_PAD, FO_DRIVE}

MODELS = {
    "base": "yolov8m-worldv2.pt",
    "ft5_paper": "5_epoch.pt",
    "paper_e40": "runs/sweep/paper_e40/weights/last.pt",
    "wise5_a0.7": "checkpoints/wise5_a0.7.pt",
    "lr1e-4_e10_bias": "checkpoints/lr1e-4_e10_bias.pt",
    "wise_lr1e-4_e10_bias": "checkpoints/wise_lr1e-4_e10_bias.pt",
    "lr1e-4_e40_bias": "checkpoints/lr1e-4_e40_bias.pt",
}
CONFS = [0.1, 0.25]


def xywh(b):
    return [float(v) for v in b]


def iou_and_cover(gt, det):
    """IoU, and the fraction of the gt box covered by det."""
    gx, gy, gw, gh = gt
    dx, dy, dw, dh = det
    ix = max(0.0, min(gx + gw, dx + dw) - max(gx, dx))
    iy = max(0.0, min(gy + gh, dy + dh) - max(gy, dy))
    inter = ix * iy
    return inter / (gw * gh + dw * dh - inter + 1e-9), inter / (gw * gh + 1e-9)


def run_inference(weights, vocab, gt, img_dir, args):
    from ultralytics import YOLOWorld
    model = YOLOWorld(weights)
    model.set_classes(vocab)
    preds = []
    for k, img in enumerate(gt["images"]):
        r = model.predict(os.path.join(img_dir, img["file_name"]), imgsz=args.imgsz,
                          conf=0.001, iou=0.7, max_det=300, verbose=False, device=args.device)[0]
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                          r.boxes.conf.tolist()):
            preds.append({"image_id": img["id"], "label": vocab[int(c)],
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
        if (k + 1) % 65 == 0:
            print(f"    {k + 1}/{len(gt['images'])}", flush=True)
    return preds


def recall_analysis(gt_by_img, preds_by_img, img_area, vocab_name):
    out = {}
    for conf in CONFS:
        hits = defaultdict(lambda: defaultdict(int))
        total = defaultdict(int)
        for img_id, gts in gt_by_img.items():
            dets = [p for p in preds_by_img.get(img_id, []) if p["score"] >= conf]
            for g in gts:
                if g["label"] not in HELD_OUT:
                    continue
                total[g["label"]] += 1
                gb = g["bbox"]
                ious = [(p, *iou_and_cover(gb, p["bbox"])) for p in dets]
                if vocab_name == "full":
                    if any(p["label"] == g["label"] and i >= 0.5 for p, i, _ in ious):
                        hits["specific_prompt_iou50"][g["label"]] += 1
                else:
                    fo_tight = any(p["label"] in FO_CLASSES and i >= 0.5 for p, i, _ in ious)
                    fo_cover = any(p["label"] in FO_CLASSES and cov >= 0.5
                                   and p["bbox"][2] * p["bbox"][3] <= 0.25 * img_area[img_id]
                                   for p, _, cov in ious)
                    any_tight = any(i >= 0.5 for p, i, _ in ious)
                    hits["generic_fo_iou50"][g["label"]] += fo_tight
                    hits["generic_fo_tight_or_covering"][g["label"]] += fo_tight or fo_cover
                    hits["any_label_iou50"][g["label"]] += any_tight
        crits = (["specific_prompt_iou50"] if vocab_name == "full" else
                 ["generic_fo_iou50", "generic_fo_tight_or_covering", "any_label_iou50"])
        res = {}
        for crit in crits:
            per = hits[crit]
            res[crit] = {
                "per_class": {c: round(per[c] / total[c], 3) for c in HELD_OUT},
                "all": round(sum(per.values()) / sum(total.values()), 3),
            }
        res["n_boxes"] = dict(total)
        out[f"conf{conf}"] = res
    return out


def auroc(pos, neg):
    """Mann-Whitney AUROC, ties count 1/2."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def obstruction_analysis(labels, preds_by_img, n_boot=1000, seed=0):
    ids = sorted(labels)
    score = {i: max([p["score"] for p in preds_by_img.get(i, []) if p["label"] == FO_PAD],
                    default=0.0) for i in ids}
    pos = [score[i] for i in ids if labels[i]]
    neg = [score[i] for i in ids if not labels[i]]
    res = {"n_pos": len(pos), "n_neg": len(neg), "auroc": round(auroc(pos, neg), 3)}
    rng = np.random.default_rng(seed)
    pos_a, neg_a = np.array(pos), np.array(neg)
    boots = [auroc(rng.choice(pos_a, len(pos_a)), rng.choice(neg_a, len(neg_a)))
             for _ in range(n_boot)]
    res["auroc_ci95"] = [round(float(np.percentile(boots, 2.5)), 3),
                         round(float(np.percentile(boots, 97.5)), 3)]
    neg_sorted = np.sort(neg_a)[::-1]
    for fpr in (0.05, 0.10):
        k = int(np.floor(fpr * len(neg_a)))  # allow k false alarms
        thr = neg_sorted[k] if k < len(neg_sorted) else 0.0
        res[f"tpr_at_fpr{int(fpr * 100)}"] = round(float((pos_a > thr).mean()), 3)
    for conf in CONFS:
        res[f"conf{conf}"] = {"miss_rate": round(float((pos_a < conf).mean()), 3),
                              "false_alarm_rate": round(float((neg_a >= conf).mean()), 3)}
    return res


def false_alarms_on_clear(labels, preds_by_img):
    """Mean number of foreign-object detections per clear-pad image (generic vocab)."""
    clear = [i for i, v in labels.items() if not v]
    return {f"conf{c}": round(float(np.mean([
        sum(p["label"] in FO_CLASSES and p["score"] >= c for p in preds_by_img.get(i, []))
        for i in clear])), 2) for c in CONFS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="datasets/OpenVocab/annotations/val.json")
    ap.add_argument("--images", default="datasets/OpenVocab/images")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--output", default="eval_results/prompt_analysis")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    gt = json.load(open(args.ann))
    names = {c["id"]: c["name"] for c in gt["categories"]}
    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]].append({"label": names[a["category_id"]], "bbox": xywh(a["bbox"])})
    img_area = {i["id"]: i["width"] * i["height"] for i in gt["images"]}
    labels = {}
    for img in gt["images"]:
        present = {g["label"] for g in gt_by_img[img["id"]]}
        if FO_PAD in present:
            labels[img["id"]] = True
        elif PAD in present:
            labels[img["id"]] = False

    out_dir = Path(args.output)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    results = {"definitions": {
        "held_out_classes": HELD_OUT, "generic_vocab": GENERIC_VOCAB, "full_vocab": FULL_VOCAB,
        "generic_fo_iou50": "held-out box matched (IoU>=0.5) by a foreign-object detection, generic vocab",
        "generic_fo_tight_or_covering": "as above, or >=50% of the box inside a foreign-object "
                                        "detection no larger than 25% of the image",
        "any_label_iou50": "held-out box matched (IoU>=0.5) by a detection of any generic label",
        "specific_prompt_iou50": "held-out box matched by its own class prompt, full vocab (reference)",
        "obstruction_positive": "image has a 'foreign object on orange circle landing pad' box",
        "obstruction_negative": "image has an 'orange circle landing pad' box and no foreign-object-on-pad box",
        "obstruction_score": "max confidence of the 'foreign object on orange circle landing pad' prompt",
    }, "models": {}}

    for name in args.models:
        weights = MODELS.get(name, name)
        results["models"][name] = {"weights": weights}
        preds = {}
        for vocab_name, vocab in (("full", FULL_VOCAB), ("generic", GENERIC_VOCAB)):
            cache = out_dir / "preds" / f"{name}__{vocab_name}.json"
            if cache.exists():
                preds[vocab_name] = json.load(open(cache))
            else:
                print(f"[infer] {name} / {vocab_name}", flush=True)
                preds[vocab_name] = run_inference(weights, vocab, gt, args.images, args)
                json.dump(preds[vocab_name], open(cache, "w"))
        by = {v: defaultdict(list) for v in preds}
        for v, ps in preds.items():
            for p in ps:
                by[v][p["image_id"]].append(p)
        m = results["models"][name]
        m["recall_full"] = recall_analysis(gt_by_img, by["full"], img_area, "full")
        m["recall_generic"] = recall_analysis(gt_by_img, by["generic"], img_area, "generic")
        m["fo_detections_per_clear_image_generic"] = false_alarms_on_clear(labels, by["generic"])
        m["obstruction_full"] = obstruction_analysis(labels, by["full"])
        m["obstruction_generic"] = obstruction_analysis(labels, by["generic"])
        json.dump(results, open(out_dir / "results.json", "w"), indent=2)
        r = m["recall_generic"]["conf0.1"]
        o = m["obstruction_full"]
        print(f"[done] {name}: generic FO recall@0.1 {r['generic_fo_iou50']['all']} "
              f"(tight|cover {r['generic_fo_tight_or_covering']['all']}), specific "
              f"{m['recall_full']['conf0.1']['specific_prompt_iou50']['all']}; "
              f"obstruction AUROC {o['auroc']} {o['auroc_ci95']}", flush=True)


if __name__ == "__main__":
    main()
