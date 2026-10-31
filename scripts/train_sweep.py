"""Single fine-tuning run for sweeps on Jetson Thor (one GPU).

Generalizes scripts/finetune.py: every hyperparameter is a flag, the optimizer is
explicit (ultralytics' optimizer='auto' silently ignores lr0), every epoch is
checkpointed, and after each epoch the run is scored on the open-vocab benchmark
and logged in the same CSV format as eval_results/training_log_lr*.csv.

Example:
    scripts/thor_run.sh python3 scripts/train_sweep.py --epochs 10 --lr0 1e-4 \
        --name m_lr1e-4_e10
"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).parent))

from pycocotools.coco import COCO
from ultralytics import YOLOWorld

from evaluate_open_vocab import (
    evaluate_coco as ov_evaluate_coco,
    evaluate_novel,
    get_classes_from_coco,
    load_model as ov_load_model,
    run_inference as ov_run_inference,
)

VISDRONE_CLASSES = ["pedestrian", "people", "bicycle", "car", "van",
                    "truck", "tricycle", "awning-tricycle", "bus", "motor"]
CSV_FIELDS = ["epoch", "lr", "visdrone_mAP50", "visdrone_mAP50-95",
              "openvocab_mAP", "openvocab_mAP50", "openvocab_mAP_novel",
              "ov_ratio_to_base", "ov_best", "ov_patience",
              "box_loss", "cls_loss", "dfl_loss", "status"]


def score_openvocab(weights, ov_gt, classes, img_dir, eval_args):
    """Return (mAP, mAP50, novel-only mAP) on the open-vocab benchmark."""
    with contextlib.redirect_stdout(io.StringIO()):
        model = ov_load_model(str(weights), classes)
        preds, _ = ov_run_inference(model, img_dir, ov_gt, eval_args)
        metrics, _ = ov_evaluate_coco(ov_gt, preds, classes)
        novel = evaluate_novel(ov_gt, preds)
    return metrics["mAP"], metrics["mAP_50"], novel["mAP_novel"]


def make_ov_callback(args, log_path, ov_gt, classes, eval_args, base_map):
    """on_fit_epoch_end hook: score last.pt on OVB, log, optionally early-stop."""
    state = {"best": base_map, "since_best": 0}

    def on_fit_epoch_end(trainer):
        ov_map, ov_map50, ov_novel = score_openvocab(
            trainer.last, ov_gt, classes, args.ov_images, eval_args)
        ratio = ov_map / base_map if base_map > 0 else 0.0

        if ov_map > state["best"]:
            state["best"], state["since_best"] = ov_map, 0
        else:
            state["since_best"] += 1

        status = "training"
        if args.ov_min_ratio and ratio < args.ov_min_ratio:
            status = f"stop: ov_ratio<{args.ov_min_ratio}"
        elif args.ov_patience and state["since_best"] >= args.ov_patience:
            status = f"stop: no ov improvement for {args.ov_patience} epochs"
        if status != "training":
            trainer.stop = True

        losses = trainer.tloss.tolist() if trainer.tloss is not None else [None] * 3
        row = {
            "epoch": trainer.epoch + 1,
            "lr": trainer.optimizer.param_groups[0]["lr"],
            "visdrone_mAP50": trainer.metrics.get("metrics/mAP50(B)"),
            "visdrone_mAP50-95": trainer.metrics.get("metrics/mAP50-95(B)"),
            "openvocab_mAP": round(ov_map, 4),
            "openvocab_mAP50": round(ov_map50, 4),
            "openvocab_mAP_novel": round(ov_novel, 4),
            "ov_ratio_to_base": round(ratio, 4),
            "ov_best": round(state["best"], 4),
            "ov_patience": state["since_best"],
            "box_loss": losses[0], "cls_loss": losses[1], "dfl_loss": losses[2],
            "status": status,
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, CSV_FIELDS).writerow(row)
        print(f"[ov] epoch {row['epoch']}: OV mAP {ov_map:.4f} (novel {ov_novel:.4f}, "
              f"ratio {ratio:.3f})  VisDrone mAP50 {row['visdrone_mAP50']}  {status}")

    return on_fit_epoch_end


def make_restore_bias_callback(weights):
    """on_pretrain_routine_end hook: copy the pretrained contrastive-head biases in.

    ultralytics' weight transfer skips them (0-d in yolov8*-worldv2.pt, shape (1,)
    in the freshly built model), so they restart at the -10 default init instead
    of the pretrained per-scale values (-11.96 / -10.45 / -8.97 for v8m).
    """
    import torch

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    pretrained = [h.bias.detach().float().reshape(-1)
                  for h in (ckpt.get("ema") or ckpt["model"]).model[-1].cv4]

    def on_pretrain_routine_end(trainer):
        targets = [trainer.model] + ([trainer.ema.ema] if trainer.ema else [])
        for m in targets:
            for head, b in zip(m.model[-1].cv4, pretrained):
                head.bias.data.copy_(b.to(head.bias).reshape(head.bias.shape))
        print("[bias] restored pretrained cv4 biases:", [round(float(b), 3) for b in pretrained])

    return on_pretrain_routine_end


def parse_args():
    p = argparse.ArgumentParser(description="One YOLO-World VisDrone fine-tuning run")
    p.add_argument("--weights", default="yolov8m-worldv2.pt",
                   help="Starting checkpoint (s/m/l/x-worldv2 for the model-size study)")
    p.add_argument("--data", default="datasets/VisDrone/data.yaml")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--optimizer", default="AdamW",
                   help="Never 'auto': it ignores lr0 and switches to SGD lr=0.01 "
                        "past 10k iterations")
    p.add_argument("--lr0", type=float, default=2e-4)
    p.add_argument("--momentum", type=float, default=0.937, help="SGD momentum / Adam beta1")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--warmup-bias-lr", type=float, default=0.1)
    p.add_argument("--freeze", type=int, default=0,
                   help="Freeze the first N model layers (10 = whole backbone)")
    p.add_argument("--restore-head-bias", action="store_true",
                   help="Reload pretrained contrastive-head biases that ultralytics' "
                        "weight transfer drops (ablation; off = stock behaviour)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/sweep")
    p.add_argument("--name", required=True)
    p.add_argument("--ov-images", default="datasets/OpenVocab/images")
    p.add_argument("--ov-ann", default="datasets/OpenVocab/annotations/val.json")
    p.add_argument("--ov-min-ratio", type=float, default=0.0,
                   help="Stop when OV mAP / base OV mAP falls below this (0 = off)")
    p.add_argument("--ov-patience", type=int, default=0,
                   help="Stop after N epochs without OV mAP improvement (0 = off)")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "ov_log.csv"

    with contextlib.redirect_stdout(io.StringIO()):
        ov_gt = COCO(args.ov_ann)
    ov_classes = get_classes_from_coco(ov_gt)
    eval_args = SimpleNamespace(imgsz=args.imgsz, conf=0.001, iou_thresh=0.7,
                                max_det=300, device="0")

    base_map, base_map50, base_novel = score_openvocab(
        args.weights, ov_gt, ov_classes, args.ov_images, eval_args)
    print(f"[ov] base {args.weights}: OV mAP {base_map:.4f} (novel {base_novel:.4f})")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, CSV_FIELDS)
        w.writeheader()
        w.writerow({"epoch": 0, "lr": args.lr0, "openvocab_mAP": round(base_map, 4),
                    "openvocab_mAP50": round(base_map50, 4),
                    "openvocab_mAP_novel": round(base_novel, 4),
                    "ov_ratio_to_base": 1.0, "ov_best": round(base_map, 4),
                    "ov_patience": 0, "status": "base_reference"})
    (run_dir / "sweep_args.json").write_text(json.dumps(vars(args), indent=2))

    model = YOLOWorld(args.weights)
    model.set_classes(VISDRONE_CLASSES)
    model.add_callback("on_fit_epoch_end",
                       make_ov_callback(args, log_path, ov_gt, ov_classes, eval_args, base_map))
    if args.restore_head_bias:
        model.add_callback("on_pretrain_routine_end", make_restore_bias_callback(args.weights))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_bias_lr=args.warmup_bias_lr,
        freeze=args.freeze or None,
        seed=args.seed,
        workers=args.workers,
        amp=True,
        device=0,
        save_period=1,
        # absolute, or ultralytics nests a relative project under runs/detect/
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
