import argparse
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLOWorld

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

DEFAULT_MODELS = OrderedDict([
    ("Base (yolov8m-worldv2)", "yolov8m-worldv2.pt"),
    ("Fine-tuned (5 epoch)", "5_epoch.pt"),
    ("Fine-tuned (40 epoch)", "runs/detect/runs/VisDrone_NewPath/yolo_world_balanced/weights/best.pt"),
    ("Fine-tuned (preserved)", "runs/detect/runs/VisDrone_Research/yolo_world_preserved/weights/best.pt"),
])


def load_model(weights_path, classes, device):
    model = YOLOWorld(weights_path)
    model.set_classes(classes)
    return model


def run_inference(model, img_dir, coco_gt, args):
    """Run model.predict() on every val image, return COCO-format results and FPS."""
    predictions = []
    total_time = 0.0
    img_ids = list(coco_gt.imgs.keys())
    num_images = len(img_ids)

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
        )
        elapsed = time.time() - start
        total_time += elapsed

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box_xyxy, conf, cls in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu()):
                x1, y1, x2, y2 = box_xyxy.tolist()
                predictions.append({
                    "image_id": img_id,
                    "category_id": int(cls),
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
        ("fps", "FPS"),
    ]

    print("\n" + "=" * 80)
    print("YOLO-World VisDrone Evaluation Results")
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
        description="Evaluate YOLO-World models on VisDrone validation set"
    )
    parser.add_argument("--val-images", default="datasets/VisDrone/images/val",
                        help="Path to validation images directory")
    parser.add_argument("--val-ann", default="datasets/VisDrone/annotations/val.json",
                        help="Path to COCO-format ground-truth annotations")
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

    # Build list of models to evaluate
    models_to_eval = OrderedDict()
    models_to_eval["Base (yolov8m-worldv2)"] = args.base_weights

    if args.finetuned_weights:
        for i, path in enumerate(args.finetuned_weights):
            if os.path.isfile(path):
                label = f"Fine-tuned ({Path(path).stem})"
                models_to_eval[label] = path
            else:
                print(f"Warning: weights not found, skipping: {path}")
    else:
        # Auto-discover known fine-tuned models
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

    # Load ground-truth
    print(f"\nLoading ground-truth from {args.val_ann}")
    coco_gt = COCO(args.val_ann)
    print(f"  {len(coco_gt.imgs)} images, {len(coco_gt.anns)} annotations")

    # Evaluate each model
    all_results = OrderedDict()

    for label, weights_path in models_to_eval.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {label} ({weights_path})")
        print(f"{'=' * 60}")

        model = load_model(weights_path, VISDRONE_CLASSES, args.device)
        predictions, fps = run_inference(model, args.val_images, coco_gt, args)
        print(f"  {len(predictions)} detections, {fps:.1f} FPS")

        metrics, per_class = evaluate_coco(coco_gt, predictions, VISDRONE_CLASSES)

        all_results[label] = {
            "metrics": metrics,
            "per_class": per_class,
            "fps": fps,
            "num_predictions": len(predictions),
            "weights": weights_path,
        }

    # Print comparison
    print_comparison(all_results)

    # Save to JSON
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
