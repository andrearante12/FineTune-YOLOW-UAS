import argparse
import glob
import json
import os
import re
import sys
from collections import OrderedDict
from types import SimpleNamespace

from pycocotools.coco import COCO
from ultralytics import YOLOWorld

# Import shared evaluation functions
from evaluate import (
    VISDRONE_CLASSES,
    run_inference as visdrone_run_inference,
    evaluate_coco as visdrone_evaluate_coco,
)
from evaluate_open_vocab import (
    get_classes_from_coco,
    run_inference as openvocab_run_inference,
    evaluate_coco as openvocab_evaluate_coco,
)

# Set for fast lookup — VISDRONE_CLASSES from evaluate.py is a list
VISDRONE_SET = set(VISDRONE_CLASSES)


def find_checkpoints(weights_dir):
    """Find all epoch*.pt files in a directory, sorted by epoch number."""
    pattern = os.path.join(weights_dir, "epoch*.pt")
    files = glob.glob(pattern)

    def epoch_num(path):
        match = re.search(r'epoch(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else 0

    return sorted(files, key=epoch_num)


def parse_run_info(weights_dir):
    """Extract learning rate from directory name like 'yolo_world_lr0.0001'."""
    parent = os.path.basename(os.path.dirname(weights_dir.rstrip('/\\')))
    match = re.search(r'lr([\d.e-]+)', parent)
    lr_str = match.group(1) if match else "unknown"
    return parent, lr_str


def compute_confidence_metrics(predictions, class_names, conf_threshold):
    """Compute per-class detection counts and mean confidence above threshold.

    Args:
        predictions: list of COCO-format dicts with 'category_id' and 'score'
        class_names: ordered list of class names (index = category_id)
        conf_threshold: only count detections at or above this confidence
    """
    per_class = {}
    for cls_idx, cls_name in enumerate(class_names):
        cls_preds = [p for p in predictions
                     if p["category_id"] == cls_idx and p["score"] >= conf_threshold]
        count = len(cls_preds)
        mean_conf = (sum(p["score"] for p in cls_preds) / count) if count > 0 else 0.0
        per_class[cls_name] = {"count": count, "mean_conf": mean_conf}
    return per_class


def compute_confidence_retention(base_conf, ckpt_conf, class_names):
    """Compute weighted confidence retention score, excluding VisDrone classes.

    Returns:
        overall: single retention score in [0, 1] (1.0 = perfect preservation)
        per_class: dict of per-class retention details
    """
    total_weight = 0
    weighted_sum = 0.0
    per_class_ret = {}

    for cls_name in class_names:
        if cls_name in VISDRONE_SET:
            continue
        base = base_conf[cls_name]
        ckpt = ckpt_conf[cls_name]
        if base["count"] == 0:
            continue  # No base detections — can't measure retention

        count_ratio = ckpt["count"] / base["count"]
        conf_ratio = (ckpt["mean_conf"] / base["mean_conf"]) if base["mean_conf"] > 0 else 0.0
        # Cap count_ratio at 1 — we only penalize losing detections, not gaining
        retention = min(count_ratio, 1.0) * min(conf_ratio, 1.0)

        per_class_ret[cls_name] = {
            "count_base": base["count"],
            "count_ckpt": ckpt["count"],
            "count_ratio": round(count_ratio, 4),
            "mean_conf_base": round(base["mean_conf"], 4),
            "mean_conf_ckpt": round(ckpt["mean_conf"], 4),
            "conf_ratio": round(conf_ratio, 4),
            "retention": round(retention, 4),
        }
        total_weight += base["count"]
        weighted_sum += retention * base["count"]

    overall = weighted_sum / total_weight if total_weight > 0 else 0.0
    return overall, per_class_ret


def evaluate_checkpoint(weights_path, visdrone_gt, visdrone_img_dir,
                        openvocab_gt, openvocab_img_dir, openvocab_classes,
                        eval_args):
    """Evaluate a single checkpoint on both datasets."""
    # VisDrone evaluation
    model = YOLOWorld(weights_path)
    model.set_classes(VISDRONE_CLASSES)
    vd_preds, vd_fps = visdrone_run_inference(model, visdrone_img_dir, visdrone_gt, eval_args)
    vd_metrics, vd_per_class = visdrone_evaluate_coco(visdrone_gt, vd_preds, VISDRONE_CLASSES)

    # Open-vocab evaluation (reload model with different classes)
    model = YOLOWorld(weights_path)
    model.set_classes(openvocab_classes)
    ov_preds, ov_fps = openvocab_run_inference(model, openvocab_img_dir, openvocab_gt, eval_args)
    ov_metrics, ov_per_class = openvocab_evaluate_coco(openvocab_gt, ov_preds, openvocab_classes)

    return {
        "visdrone": {"metrics": vd_metrics, "per_class": vd_per_class, "fps": vd_fps},
        "openvocab": {"metrics": ov_metrics, "per_class": ov_per_class, "fps": ov_fps,
                      "predictions": ov_preds},
    }


def compute_combined_score(vd_mAP, base_vd_mAP, confidence_retention, vd_weight, ov_weight):
    """Compute combined score: VisDrone improvement + OpenVocab confidence retention."""
    vd_ratio = vd_mAP / base_vd_mAP if base_vd_mAP > 0 else 0.0
    return vd_weight * vd_ratio + ov_weight * confidence_retention


def main():
    parser = argparse.ArgumentParser(
        description="Sweep epoch checkpoints across training runs to find optimal (lr, epoch)"
    )
    parser.add_argument("--weights-dir", nargs="+", default=[],
                        help="One or more weights directories containing epoch*.pt files")
    parser.add_argument("--checkpoint", nargs="+", default=[],
                        help="One or more individual .pt files to evaluate")
    parser.add_argument("--base-weights", default="yolov8m-worldv2.pt",
                        help="Base model weights for reference scores")
    parser.add_argument("--visdrone-images", default="datasets/VisDrone/images/val",
                        help="VisDrone validation images directory")
    parser.add_argument("--visdrone-ann", default="datasets/VisDrone/annotations/val.json",
                        help="VisDrone COCO annotation file")
    parser.add_argument("--openvocab-images", default="datasets/OpenVocab/images",
                        help="Open-vocab validation images directory")
    parser.add_argument("--openvocab-ann", default="datasets/OpenVocab/annotations/val.json",
                        help="Open-vocab COCO annotation file")
    parser.add_argument("--visdrone-weight", type=float, default=1.0,
                        help="Weight for VisDrone score in combined metric (default: 1.0)")
    parser.add_argument("--openvocab-weight", type=float, default=1.0,
                        help="Weight for OpenVocab confidence retention in combined metric (default: 1.0)")
    parser.add_argument("--conf-threshold", type=float, default=0.1,
                        help="Confidence threshold for retention metrics (default: 0.1)")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou-thresh", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="eval_results",
                        help="Directory to save results JSON")
    args = parser.parse_args()

    if not args.weights_dir and not args.checkpoint:
        print("Error: provide --weights-dir and/or --checkpoint")
        sys.exit(1)

    # Validate paths
    for path, label in [(args.visdrone_images, "VisDrone images"),
                        (args.visdrone_ann, "VisDrone annotations"),
                        (args.openvocab_images, "OpenVocab images"),
                        (args.openvocab_ann, "OpenVocab annotations")]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}")
            sys.exit(1)

    # Create eval args namespace for run_inference calls
    eval_args = SimpleNamespace(
        imgsz=args.imgsz, conf=args.conf, iou_thresh=args.iou_thresh,
        max_det=args.max_det, device=args.device,
    )

    # Load ground-truth datasets
    print("Loading VisDrone ground-truth...")
    visdrone_gt = COCO(args.visdrone_ann)
    print(f"  {len(visdrone_gt.imgs)} images, {len(visdrone_gt.anns)} annotations")

    print("Loading OpenVocab ground-truth...")
    openvocab_gt = COCO(args.openvocab_ann)
    openvocab_classes = get_classes_from_coco(openvocab_gt)
    print(f"  {len(openvocab_gt.imgs)} images, {len(openvocab_gt.anns)} annotations")
    print(f"  Classes: {openvocab_classes}")

    # Evaluate base model first for reference scores
    print(f"\n{'=' * 70}")
    print(f"Evaluating BASE model: {args.base_weights}")
    print(f"{'=' * 70}")
    base_results = evaluate_checkpoint(
        args.base_weights, visdrone_gt, args.visdrone_images,
        openvocab_gt, args.openvocab_images, openvocab_classes, eval_args
    )
    base_vd_mAP = base_results["visdrone"]["metrics"]["mAP"]
    base_ov_mAP = base_results["openvocab"]["metrics"]["mAP"]

    # Compute base confidence metrics for retention comparison
    base_conf_metrics = compute_confidence_metrics(
        base_results["openvocab"]["predictions"], openvocab_classes, args.conf_threshold
    )
    base_combined = compute_combined_score(
        base_vd_mAP, base_vd_mAP, 1.0,
        args.visdrone_weight, args.openvocab_weight
    )

    # Show base confidence profile
    print(f"\nBase scores: VisDrone mAP={base_vd_mAP:.4f}, OpenVocab mAP={base_ov_mAP:.4f}")
    print(f"Base confidence profile (conf>={args.conf_threshold}):")
    for cls_name in openvocab_classes:
        if cls_name in VISDRONE_SET:
            continue
        info = base_conf_metrics[cls_name]
        if info["count"] > 0:
            print(f"  {cls_name}: {info['count']} detections, mean_conf={info['mean_conf']:.3f}")
        else:
            print(f"  {cls_name}: 0 detections (excluded from retention)")
    print(f"Combined score: {base_combined:.3f} (reference)")

    # Sweep all runs
    all_sweep_results = OrderedDict()
    global_best_score = -1
    global_best_checkpoint = None
    global_best_run = None

    for weights_dir in args.weights_dir:
        run_name, lr_str = parse_run_info(weights_dir)
        checkpoints = find_checkpoints(weights_dir)

        if not checkpoints:
            print(f"\nNo epoch*.pt files found in {weights_dir}, skipping.")
            continue

        print(f"\n{'=' * 70}")
        print(f"Run: {run_name} (lr={lr_str}) — {len(checkpoints)} checkpoints")
        print(f"{'=' * 70}")

        run_results = []

        for ckpt_path in checkpoints:
            epoch_match = re.search(r'epoch(\d+)', os.path.basename(ckpt_path))
            epoch = int(epoch_match.group(1)) if epoch_match else 0

            print(f"\n--- Epoch {epoch} ({os.path.basename(ckpt_path)}) ---")
            results = evaluate_checkpoint(
                ckpt_path, visdrone_gt, args.visdrone_images,
                openvocab_gt, args.openvocab_images, openvocab_classes, eval_args
            )

            vd_mAP = results["visdrone"]["metrics"]["mAP"]
            ov_mAP = results["openvocab"]["metrics"]["mAP"]

            # Confidence retention metrics
            ckpt_conf_metrics = compute_confidence_metrics(
                results["openvocab"]["predictions"], openvocab_classes, args.conf_threshold
            )
            retention, per_class_ret = compute_confidence_retention(
                base_conf_metrics, ckpt_conf_metrics, openvocab_classes
            )

            combined = compute_combined_score(
                vd_mAP, base_vd_mAP, retention,
                args.visdrone_weight, args.openvocab_weight
            )

            # Don't save raw predictions to JSON (too large)
            ov_results_no_preds = {k: v for k, v in results["openvocab"].items() if k != "predictions"}

            entry = {
                "epoch": epoch,
                "checkpoint": ckpt_path,
                "visdrone_mAP": vd_mAP,
                "openvocab_mAP": ov_mAP,
                "confidence_retention": round(retention, 4),
                "per_class_retention": per_class_ret,
                "combined_score": combined,
                "visdrone": results["visdrone"],
                "openvocab": ov_results_no_preds,
            }
            run_results.append(entry)

            # Print summary with per-class retention
            ret_parts = [f"{c}:{r['retention']:.2f}" for c, r in per_class_ret.items()]
            ret_str = " ".join(ret_parts) if ret_parts else "n/a"
            print(f"  VisDrone mAP={vd_mAP:.4f}, OV Retention={retention:.3f}, Combined={combined:.3f}")
            print(f"    Per-class: {ret_str}")

            if combined > global_best_score:
                global_best_score = combined
                global_best_checkpoint = ckpt_path
                global_best_run = run_name

        all_sweep_results[run_name] = {
            "lr": lr_str,
            "weights_dir": weights_dir,
            "checkpoints": run_results,
        }

    # Evaluate individual checkpoints
    if args.checkpoint:
        from pathlib import Path
        run_results = []
        for ckpt_path in args.checkpoint:
            if not os.path.isfile(ckpt_path):
                print(f"\nCheckpoint not found: {ckpt_path}, skipping.")
                continue

            label = Path(ckpt_path).stem
            print(f"\n{'=' * 70}")
            print(f"Evaluating: {label} ({ckpt_path})")
            print(f"{'=' * 70}")

            results = evaluate_checkpoint(
                ckpt_path, visdrone_gt, args.visdrone_images,
                openvocab_gt, args.openvocab_images, openvocab_classes, eval_args
            )

            vd_mAP = results["visdrone"]["metrics"]["mAP"]
            ov_mAP = results["openvocab"]["metrics"]["mAP"]

            ckpt_conf_metrics = compute_confidence_metrics(
                results["openvocab"]["predictions"], openvocab_classes, args.conf_threshold
            )
            retention, per_class_ret = compute_confidence_retention(
                base_conf_metrics, ckpt_conf_metrics, openvocab_classes
            )

            combined = compute_combined_score(
                vd_mAP, base_vd_mAP, retention,
                args.visdrone_weight, args.openvocab_weight
            )

            ov_results_no_preds = {k: v for k, v in results["openvocab"].items() if k != "predictions"}

            entry = {
                "label": label,
                "checkpoint": ckpt_path,
                "visdrone_mAP": vd_mAP,
                "openvocab_mAP": ov_mAP,
                "confidence_retention": round(retention, 4),
                "per_class_retention": per_class_ret,
                "combined_score": combined,
                "visdrone": results["visdrone"],
                "openvocab": ov_results_no_preds,
            }
            run_results.append(entry)

            ret_parts = [f"{c}:{r['retention']:.2f}" for c, r in per_class_ret.items()]
            ret_str = " ".join(ret_parts) if ret_parts else "n/a"
            print(f"  VisDrone mAP={vd_mAP:.4f}, OV Retention={retention:.3f}, Combined={combined:.3f}")
            print(f"    Per-class: {ret_str}")

            if combined > global_best_score:
                global_best_score = combined
                global_best_checkpoint = ckpt_path
                global_best_run = label

        if run_results:
            all_sweep_results["individual"] = {
                "lr": "n/a",
                "weights_dir": "individual",
                "checkpoints": run_results,
            }

    # Print summary table
    print(f"\n\n{'=' * 90}")
    print(f"SWEEP SUMMARY  (vd_weight={args.visdrone_weight}, ov_weight={args.openvocab_weight}, "
          f"conf_thresh={args.conf_threshold})")
    print(f"{'=' * 90}")
    print()

    for run_name, run_data in all_sweep_results.items():
        print(f"Run: {run_name} (lr={run_data['lr']})")
        print(f"{'Name':>12} | {'VisDrone mAP':>13} | {'OV mAP':>8} | {'OV Retention':>13} | {'Combined':>10} | Per-Class Retention")
        print(f"{'-'*12}-+-{'-'*13}-+-{'-'*8}-+-{'-'*13}-+-{'-'*10}-+-{'-'*30}")

        for entry in run_data["checkpoints"]:
            marker = " *" if entry["checkpoint"] == global_best_checkpoint else ""
            ret_parts = [f"{c[:12]}:{r['retention']:.2f}" for c, r in entry.get("per_class_retention", {}).items()]
            ret_str = " ".join(ret_parts) if ret_parts else ""
            row_label = entry.get("label", f"epoch {entry.get('epoch', '?')}")
            print(f"{row_label:>12} | {entry['visdrone_mAP']:>13.4f} | {entry['openvocab_mAP']:>8.4f} | "
                  f"{entry['confidence_retention']:>13.3f} | {entry['combined_score']:>10.3f}{marker:2s} | {ret_str}")
        print()

    print(f"{'Base':>6} | {base_vd_mAP:>13.4f} | {base_ov_mAP:>8.4f} | {'1.000':>13s} | {base_combined:>10.3f}   | (reference)")
    print()

    if global_best_checkpoint:
        best_entry = None
        for run_data in all_sweep_results.values():
            for e in run_data["checkpoints"]:
                if e["checkpoint"] == global_best_checkpoint:
                    best_entry = e
                    break

        print(f"OPTIMAL: {global_best_checkpoint}")
        print(f"  Run: {global_best_run}, Combined={global_best_score:.3f}")
        if best_entry:
            vd_gain = (best_entry["visdrone_mAP"] / base_vd_mAP - 1) * 100 if base_vd_mAP > 0 else 0
            print(f"  VisDrone mAP: {best_entry['visdrone_mAP']:.4f} ({vd_gain:+.1f}% vs base)")
            print(f"  OV Confidence Retention: {best_entry['confidence_retention']:.3f}")
            if best_entry.get("per_class_retention"):
                for cls, r in best_entry["per_class_retention"].items():
                    print(f"    {cls}: {r['count_ckpt']}/{r['count_base']} detections "
                          f"(count_ratio={r['count_ratio']:.2f}), "
                          f"conf {r['mean_conf_base']:.3f}->{r['mean_conf_ckpt']:.3f} "
                          f"(retention={r['retention']:.3f})")

    # Save full results
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "sweep_results.json")

    # Strip predictions from base results for JSON
    base_ov_no_preds = {k: v for k, v in base_results["openvocab"].items() if k != "predictions"}
    base_conf_serializable = {cls: {"count": m["count"], "mean_conf": round(m["mean_conf"], 4)}
                              for cls, m in base_conf_metrics.items()}

    save_data = {
        "config": {
            "visdrone_weight": args.visdrone_weight,
            "openvocab_weight": args.openvocab_weight,
            "conf_threshold": args.conf_threshold,
            "conf_for_mAP": args.conf,
        },
        "base": {
            "weights": args.base_weights,
            "visdrone_mAP": base_vd_mAP,
            "openvocab_mAP": base_ov_mAP,
            "confidence_profile": base_conf_serializable,
            "combined_score": base_combined,
            "visdrone": base_results["visdrone"],
            "openvocab": base_ov_no_preds,
        },
        "runs": all_sweep_results,
        "optimal": {
            "checkpoint": global_best_checkpoint,
            "run": global_best_run,
            "combined_score": global_best_score,
        },
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
