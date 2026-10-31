import argparse
import contextlib
import io
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLOWorld

DEFAULT_MODELS = OrderedDict([
    ("Base (yolov8m-worldv2)", "yolov8m-worldv2.pt"),
    ("Fine-tuned (5 epoch)", "5_epoch.pt"),
    ("Fine-tuned (40 epoch)", "runs/detect/runs/VisDrone_NewPath/yolo_world_balanced/weights/best.pt"),
    ("Fine-tuned (preserved)", "runs/detect/runs/VisDrone_Research/yolo_world_preserved/weights/best.pt"),
])


def get_classes_from_coco(coco_gt):
    """Extract class names from COCO annotation file, sorted by category_id."""
    cat_ids = sorted(coco_gt.getCatIds())
    return [coco_gt.loadCats([cid])[0]["name"] for cid in cat_ids]


def load_model(weights_path, classes):
    model = YOLOWorld(weights_path)
    model.set_classes(classes)
    return model


def run_inference(model, img_dir, coco_gt, args):
    """Run model.predict() on every val image, return COCO-format results and FPS."""
    predictions = []
    total_time = 0.0
    img_ids = list(coco_gt.imgs.keys())
    num_images = len(img_ids)
    # Model class index i is the i-th sorted category (get_classes_from_coco); map it
    # back to the annotation's category id, which need not be 0-indexed (Roboflow
    # exports start at 1).
    cat_ids = sorted(coco_gt.getCatIds())

    for idx, img_id in enumerate(img_ids):
        img_info = coco_gt.imgs[img_id]
        img_path = os.path.join(img_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            print(f"  Warning: image not found: {img_path}, skipping")
            continue

        start = time.time()
        results = model.predict(
            img_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou_thresh,
            max_det=args.max_det,
            verbose=False,
            device=args.device,
            quantize=16 if getattr(args, "half", False) else None,
        )
        elapsed = time.time() - start
        total_time += elapsed

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box_xyxy, conf, cls in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu()):
                x1, y1, x2, y2 = box_xyxy.tolist()
                predictions.append({
                    "image_id": img_id,
                    "category_id": cat_ids[int(cls)],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(conf),
                })

        if (idx + 1) % 50 == 0 or (idx + 1) == num_images:
            print(f"  [{idx + 1}/{num_images}]")

    avg_fps = num_images / total_time if total_time > 0 else 0.0
    return predictions, avg_fps


def evaluate_coco(coco_gt, predictions, class_names):
    """Run COCOeval and return overall + per-class metrics."""
    if len(predictions) == 0:
        print("  No predictions — returning zero metrics.")
        zero_metrics = {
            "mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0,
            "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
        }
        per_class = {name: 0.0 for name in class_names}
        return zero_metrics, per_class

    coco_dt = coco_gt.loadRes(predictions)

    # Overall metrics
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    metrics = {
        "mAP": stats[0],
        "mAP_50": stats[1],
        "mAP_75": stats[2],
        "mAP_small": stats[3],
        "mAP_medium": stats[4],
        "mAP_large": stats[5],
    }

    # Per-class AP
    cat_ids = sorted(coco_gt.getCatIds())
    id_to_name = {cat["id"]: cat["name"] for cat in coco_gt.loadCats(cat_ids)}
    per_class = {}
    for cat_id in cat_ids:
        cat_eval = COCOeval(coco_gt, coco_dt, "bbox")
        cat_eval.params.catIds = [cat_id]
        cat_eval.evaluate()
        cat_eval.accumulate()
        cat_eval.summarize()
        name = id_to_name.get(cat_id, str(cat_id))
        per_class[name] = cat_eval.stats[0]

    return metrics, per_class


VISDRONE_CLASSES = {"pedestrian", "people", "bicycle", "car", "van",
                    "truck", "tricycle", "awning-tricycle", "bus", "motor"}


def evaluate_novel(coco_gt, predictions):
    """mAP / mAP@50 over only the categories the model never saw in VisDrone training."""
    novel_ids = [c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())
                 if c["name"] not in VISDRONE_CLASSES]
    preds = [p for p in predictions if p["category_id"] in novel_ids]
    if not preds:
        return {"mAP_novel": 0.0, "mAP_50_novel": 0.0}
    coco_eval = COCOeval(coco_gt, coco_gt.loadRes(preds), "bbox")
    coco_eval.params.catIds = novel_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {"mAP_novel": coco_eval.stats[0], "mAP_50_novel": coco_eval.stats[1]}


def _bootstrap_sample(coco_gt, preds_by_img, img_ids, cat_ids):
    """COCOeval stats[:2] on one resample. Images drawn twice get fresh ids, since
    COCOeval de-duplicates imgIds and would silently drop the repeats."""
    images, anns, dets = [], [], []
    for new_id, old_id in enumerate(img_ids):
        images.append(dict(coco_gt.imgs[old_id], id=new_id))
        for a in coco_gt.imgToAnns[old_id]:
            anns.append(dict(a, id=len(anns) + 1, image_id=new_id))
        dets.extend(dict(p, image_id=new_id) for p in preds_by_img.get(old_id, []))
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": coco_gt.dataset["categories"]}
    gt.createIndex()
    if not dets:
        return 0.0, 0.0
    coco_eval = COCOeval(gt, gt.loadRes(dets), "bbox")
    coco_eval.params.catIds = cat_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0], coco_eval.stats[1]


def bootstrap_ci(coco_gt, predictions, n=1000, seed=0, novel_only=False):
    """95% percentile CI on mAP and mAP@50 by resampling images with replacement."""
    rng = np.random.default_rng(seed)
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_ids = [c["id"] for c in cats
               if not (novel_only and c["name"] in VISDRONE_CLASSES)]
    preds_by_img = {}
    for p in predictions:
        if p["category_id"] in cat_ids:
            preds_by_img.setdefault(p["image_id"], []).append(p)
    all_ids = np.array(sorted(coco_gt.getImgIds()))

    stats = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n):
            sample = rng.choice(all_ids, size=len(all_ids), replace=True)
            stats.append(_bootstrap_sample(coco_gt, preds_by_img, sample, cat_ids))
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    suffix = "_novel" if novel_only else ""
    return {f"mAP{suffix}_ci95": [lo[0], hi[0]], f"mAP_50{suffix}_ci95": [lo[1], hi[1]]}


def print_comparison(all_results):
    """Print a side-by-side comparison table."""
    model_names = list(all_results.keys())
    base_name = model_names[0]

    metric_labels = [
        ("mAP", "mAP@[.5:.95]"),
        ("mAP_50", "mAP@.50"),
        ("mAP_75", "mAP@.75"),
        ("mAP_small", "mAP Small"),
        ("mAP_medium", "mAP Medium"),
        ("mAP_large", "mAP Large"),
        ("mAP_novel", "mAP Novel-only"),
        ("mAP_50_novel", "mAP@.50 Novel-only"),
        ("fps", "FPS"),
    ]

    print("\n" + "=" * 80)
    print("Open-Vocab Evaluation Results")
    print("=" * 80)

    # Header
    header = f"{'Metric':<20}"
    for name in model_names:
        header += f" | {name:>20}"
    if len(model_names) > 1:
        header += f" | {'Delta':>10}"
    print(header)
    print("-" * len(header))

    # Overall metrics
    for key, label in metric_labels:
        row = f"{label:<20}"
        values = []
        for name in model_names:
            val = all_results[name]["metrics"].get(key, all_results[name].get(key, 0.0))
            values.append(val)
            row += f" | {val:>20.3f}"
        if len(model_names) > 1:
            delta = values[-1] - values[0]
            sign = "+" if delta >= 0 else ""
            row += f" | {sign}{delta:>9.3f}"
        print(row)

    # Per-class AP
    if any("per_class" in all_results[n] for n in model_names):
        print()
        print(f"{'Per-Class AP@[.5:.95]'}")
        print("-" * len(header))
        class_header = f"{'Class':<20}"
        for name in model_names:
            class_header += f" | {name:>20}"
        if len(model_names) > 1:
            class_header += f" | {'Delta':>10}"
        print(class_header)
        print("-" * len(header))

        class_names = list(all_results[base_name]["per_class"].keys())
        for cls_name in class_names:
            row = f"{cls_name:<20}"
            values = []
            for name in model_names:
                val = all_results[name]["per_class"].get(cls_name, 0.0)
                values.append(val)
                row += f" | {val:>20.3f}"
            if len(model_names) > 1:
                delta = values[-1] - values[0]
                sign = "+" if delta >= 0 else ""
                row += f" | {sign}{delta:>9.3f}"
            print(row)

    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO-World open-vocab capability on a custom dataset"
    )
    parser.add_argument("--val-images", default="datasets/OpenVocab/images",
                        help="Path to validation images directory")
    parser.add_argument("--val-ann", default="datasets/OpenVocab/annotations/val.json",
                        help="Path to COCO-format ground-truth annotations")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Override class prompts for set_classes() "
                             "(one per category, in ascending category_id order)")
    parser.add_argument("--base-weights", default="yolov8m-worldv2.pt",
                        help="Base model weights")
    parser.add_argument("--finetuned-weights", nargs="+", default=None,
                        help="Fine-tuned model weight paths (auto-discovers if omitted)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold (low for full mAP curve)")
    parser.add_argument("--iou-thresh", type=float, default=0.7,
                        help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Max detections per image")
    parser.add_argument("--device", default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Image-level bootstrap resamples for 95%% CIs (0 = off; ~1000)")
    parser.add_argument("--output", default="eval_results",
                        help="Directory to save results JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate paths
    if not os.path.isdir(args.val_images):
        print(f"Error: validation images directory not found: {args.val_images}")
        return
    if not os.path.isfile(args.val_ann):
        print(f"Error: annotation file not found: {args.val_ann}")
        return

    # Load ground-truth
    print(f"Loading ground-truth from {args.val_ann}")
    coco_gt = COCO(args.val_ann)
    print(f"  {len(coco_gt.imgs)} images, {len(coco_gt.anns)} annotations")

    # Determine classes
    if args.classes:
        classes = args.classes
        print(f"\nUsing custom class prompts ({len(classes)} classes):")
    else:
        classes = get_classes_from_coco(coco_gt)
        print(f"\nUsing classes from annotation file ({len(classes)} classes):")

    # Print class-to-id mapping for verification
    cat_ids = sorted(coco_gt.getCatIds())
    ann_names = [coco_gt.loadCats([cid])[0]["name"] for cid in cat_ids]
    for i, (cls, ann_name, cat_id) in enumerate(zip(classes, ann_names, cat_ids)):
        if cls != ann_name:
            print(f"  [{i}] prompt: \"{cls}\" -> annotation category: \"{ann_name}\" (id={cat_id})")
        else:
            print(f"  [{i}] \"{cls}\" (id={cat_id})")

    # Build list of models to evaluate
    models_to_eval = OrderedDict()
    models_to_eval["Base (yolov8m-worldv2)"] = args.base_weights

    if args.finetuned_weights:
        for path in args.finetuned_weights:
            if os.path.isfile(path):
                label = f"Fine-tuned ({Path(path).stem})"
                models_to_eval[label] = path
            else:
                print(f"Warning: weights not found, skipping: {path}")
    else:
        for label, path in DEFAULT_MODELS.items():
            if label.startswith("Base"):
                continue
            if os.path.isfile(path):
                models_to_eval[label] = path
                print(f"Found: {label} -> {path}")
            else:
                print(f"Not found, skipping: {label} -> {path}")

    if len(models_to_eval) < 2:
        print("Warning: only one model found — no comparison will be shown.")

    # Evaluate each model
    all_results = OrderedDict()

    for label, weights_path in models_to_eval.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {label} ({weights_path})")
        print(f"{'=' * 60}")

        model = load_model(weights_path, classes)
        predictions, fps = run_inference(model, args.val_images, coco_gt, args)
        print(f"  {len(predictions)} detections, {fps:.1f} FPS")

        metrics, per_class = evaluate_coco(coco_gt, predictions, classes)
        metrics.update(evaluate_novel(coco_gt, predictions))
        if args.bootstrap:
            print(f"  Bootstrapping {args.bootstrap} resamples for 95% CIs...")
            metrics.update(bootstrap_ci(coco_gt, predictions, n=args.bootstrap))
            metrics.update(bootstrap_ci(coco_gt, predictions, n=args.bootstrap,
                                        novel_only=True))
            print("  " + "  ".join(f"{k}=[{v[0]:.3f}, {v[1]:.3f}]"
                                   for k, v in metrics.items() if k.endswith("_ci95")))

        all_results[label] = {
            "metrics": metrics,
            "per_class": per_class,
            "fps": fps,
            "num_predictions": len(predictions),
            "weights": weights_path,
            "classes": classes,
        }

    # Print comparison
    print_comparison(all_results)

    # Save to JSON
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "open_vocab_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
