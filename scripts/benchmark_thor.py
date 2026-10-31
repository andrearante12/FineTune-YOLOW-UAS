"""Jetson Thor inference benchmark: PyTorch vs TensorRT, latency / power / accuracy.

Runs natively (TensorRT engines need JetPack's TensorRT 10.x, not the container):
    source ~/venvs/yoloworld-trt/bin/activate
    python scripts/benchmark_thor.py --weights yolov8m-worldv2.pt 5_epoch.pt

For each weights x runtime x imgsz it measures batch-1 latency on real VisDrone
frames (p50/p95/p99, preprocess/inference/postprocess split), samples board power
with tegrastats, and optionally (--accuracy) re-scores VisDrone and open-vocab mAP
through the same evaluators as the paper so export-induced drops are visible.
The power mode (nvpmodel) is recorded but not changed; set it before running.

TensorRT engines bake the vocabulary in at export time, so accuracy runs build one
engine per vocabulary (VisDrone classes, OVB classes).
"""
import argparse
import contextlib
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO, YOLOWorld

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as vd  # noqa: E402
import evaluate_open_vocab as ov  # noqa: E402

RUNTIMES = ["torch-fp32", "torch-fp16", "trt-fp16", "trt-int8"]
RAILS = ["VIN", "VDD_GPU", "VDD_CPU_SOC_MSS"]


class PowerSampler:
    """Background tegrastats reader; mean instantaneous mW per rail."""

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.samples = {r: [] for r in RAILS}

    def __enter__(self):
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                     stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            for rail in RAILS:
                m = re.search(rf"\b{rail} (\d+)mW", line)
                if m:
                    self.samples[rail].append(int(m.group(1)))

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()

    def mean_w(self):
        return {r: (np.mean(v) / 1000 if v else None) for r, v in self.samples.items()}


def power_mode():
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout
        return out.splitlines()[0].replace("NV Power Mode:", "").strip()
    except (OSError, IndexError):
        return "unknown"


def engine_path(weights, runtime, imgsz, vocab_name, cache_dir):
    return Path(cache_dir) / f"{Path(weights).stem}_{vocab_name}_{imgsz}_{runtime}.engine"


def build_model(weights, runtime, imgsz, classes, vocab_name, args):
    """Return (predict-able model, half flag, export seconds or None)."""
    if runtime.startswith("torch"):
        model = YOLOWorld(weights)
        model.set_classes(classes)
        return model, runtime == "torch-fp16", None

    target = engine_path(weights, runtime, imgsz, vocab_name, args.engine_dir)
    export_s = None
    if not target.exists():
        model = YOLOWorld(weights)
        model.set_classes(classes)
        t0 = time.perf_counter()
        out = model.export(format="engine", imgsz=imgsz, batch=1, device=0,
                           quantize=8 if runtime == "trt-int8" else 16,
                           data=args.calib_data if runtime == "trt-int8" else None,
                           fraction=args.calib_fraction, workspace=8, verbose=False)
        export_s = time.perf_counter() - t0
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(out, target)
    return YOLO(str(target), task="detect"), False, export_s


def time_latency(model, images, imgsz, half, args):
    """Warm up, then time `args.iters` batch-1 predictions cycling through in-memory frames."""
    kw = dict(imgsz=imgsz, conf=args.conf, quantize=16 if half else None, device=0,
              verbose=False)
    for i in range(args.warmup):
        model.predict(images[i % len(images)], **kw)
    torch.cuda.synchronize()

    e2e, pre, inf, post = [], [], [], []
    with PowerSampler() as power:
        for i in range(args.iters):
            t0 = time.perf_counter()
            r = model.predict(images[i % len(images)], **kw)[0]
            torch.cuda.synchronize()
            e2e.append((time.perf_counter() - t0) * 1000)
            pre.append(r.speed["preprocess"])
            inf.append(r.speed["inference"])
            post.append(r.speed["postprocess"])
    e2e = np.array(e2e)
    watts = power.mean_w()
    fps = 1000 / e2e.mean()
    return {
        "fps": fps,
        "lat_mean_ms": e2e.mean(),
        "lat_p50_ms": np.percentile(e2e, 50),
        "lat_p95_ms": np.percentile(e2e, 95),
        "lat_p99_ms": np.percentile(e2e, 99),
        "preprocess_ms": np.mean(pre),
        "inference_ms": np.mean(inf),
        "postprocess_ms": np.mean(post),
        "power_vin_w": watts["VIN"],
        "power_gpu_w": watts["VDD_GPU"],
        "power_cpu_soc_w": watts["VDD_CPU_SOC_MSS"],
        "fps_per_w": fps / watts["VIN"] if watts["VIN"] else None,
    }


def score_accuracy(weights, runtime, imgsz, args):
    """Paper evaluators on VisDrone val and OVB, using this runtime's model."""
    eval_args = SimpleNamespace(imgsz=imgsz, conf=0.001, iou_thresh=0.7, max_det=300,
                                device="0")
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        vd_gt = COCO(args.visdrone_ann)
        model, half, _ = build_model(weights, runtime, imgsz, vd.VISDRONE_CLASSES,
                                     "visdrone", args)
        eval_args.half = half
        preds, _ = vd.run_inference(model, args.visdrone_images, vd_gt, eval_args)
        m, _ = vd.evaluate_coco(vd_gt, preds, vd.VISDRONE_CLASSES)
        out.update(visdrone_mAP=m["mAP"], visdrone_mAP50=m["mAP_50"])

        ov_gt = COCO(args.ov_ann)
        ov_classes = ov.get_classes_from_coco(ov_gt)
        model, half, _ = build_model(weights, runtime, imgsz, ov_classes, "ovb", args)
        eval_args.half = half
        preds, _ = ov.run_inference(model, args.ov_images, ov_gt, eval_args)
        m, _ = ov.evaluate_coco(ov_gt, preds, ov_classes)
        out.update(ov_mAP=m["mAP"], ov_mAP50=m["mAP_50"],
                   ov_mAP_novel=ov.evaluate_novel(ov_gt, preds)["mAP_novel"])
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weights", nargs="+", default=["yolov8m-worldv2.pt", "5_epoch.pt"])
    p.add_argument("--runtimes", nargs="+", default=RUNTIMES, choices=RUNTIMES)
    p.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--conf", type=float, default=0.05,
                   help="Deployment-like threshold for timing (accuracy uses 0.001)")
    p.add_argument("--accuracy", action="store_true",
                   help="Also re-score VisDrone + OVB mAP for every configuration")
    p.add_argument("--bench-images", default="datasets/VisDrone/images/val")
    p.add_argument("--num-bench-images", type=int, default=50)
    p.add_argument("--visdrone-images", default="datasets/VisDrone/images/val")
    p.add_argument("--visdrone-ann", default="datasets/VisDrone/annotations/val.json")
    p.add_argument("--ov-images", default="datasets/OpenVocab/images")
    p.add_argument("--ov-ann", default="datasets/OpenVocab/annotations/val.json")
    p.add_argument("--calib-data", default="datasets/VisDrone/data.yaml",
                   help="INT8 calibration images (val split of this dataset yaml)")
    p.add_argument("--calib-fraction", type=float, default=1.0)
    p.add_argument("--engine-dir", default="runs/engines")
    p.add_argument("--output", default="eval_results/thor_benchmark")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = power_mode()
    tag = re.sub(r"\W+", "", mode) or "unknown"
    csv_path = out_dir / f"benchmark_{tag}.csv"

    images = sorted(Path(args.bench_images).glob("*.jpg"))[:args.num_bench_images]
    if not images:
        raise SystemExit(f"No benchmark images in {args.bench_images}")
    # Decode up front: a camera pipeline hands the model frames already in memory
    images = [cv2.imread(str(p)) for p in images]
    print(f"Power mode: {mode} | {len(images)} benchmark images | -> {csv_path}")

    rows = []
    for weights in args.weights:
        for imgsz in args.imgsz:
            for runtime in args.runtimes:
                label = f"{Path(weights).stem} {runtime} @{imgsz}"
                print(f"\n=== {label}")
                try:
                    model, half, export_s = build_model(
                        weights, runtime, imgsz, vd.VISDRONE_CLASSES, "visdrone", args)
                    row = {"power_mode": mode, "weights": weights, "runtime": runtime,
                           "imgsz": imgsz, "export_s": export_s}
                    row.update(time_latency(model, images, imgsz, half, args))
                    del model
                    torch.cuda.empty_cache()
                    if args.accuracy:
                        row.update(score_accuracy(weights, runtime, imgsz, args))
                except Exception as e:  # keep the sweep going; record the failure
                    print(f"  FAILED: {e!r}")
                    row = {"power_mode": mode, "weights": weights, "runtime": runtime,
                           "imgsz": imgsz, "error": repr(e)}
                rows.append(row)
                print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in row.items() if k not in ("weights",)))

                fields = sorted({k for r in rows for k in r},
                                key=lambda k: (k not in rows[0], k))
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fields)
                    w.writeheader()
                    w.writerows(rows)
    (out_dir / f"benchmark_{tag}.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
