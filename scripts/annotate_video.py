"""Annotated videos of YOLO-World models on drone footage, for qualitative comparison.

Writes, per model, a full-resolution annotated MP4, and (with 2-4 models) a 2x2
side-by-side comparison video where every model sees the same frame.

    python scripts/annotate_video.py --video ~/vlm_on_jetson/data/videos/DJI_0162.MP4 \
        --models base=yolov8m-worldv2.pt wise=runs/wise_ft/wise_5_epoch_a0.70.pt \
        --output deliverables/videos
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOWorld

# Landing-zone prompts from scripts/run_inference.py (plus dog, which OVB annotates)
DEFAULT_PROMPTS = [
    "orange circle landing pad",
    "backpack",
    "dog",
    "pedestrian",
    "car",
    "soccer ball",
    "foreign object on driveway",
    "foreign object on orange circle landing pad",
]


def label_frame(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True)
    p.add_argument("--models", nargs="+", required=True, help="label=weights pairs")
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.10, help="as in run_inference.py")
    p.add_argument("--device", default="0")
    p.add_argument("--stride", type=int, default=1, help="Use every Nth frame")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--no-full", action="store_true", help="Only write the comparison grid")
    p.add_argument("--grid-name", default="comparison",
                   help="Grid video is <video stem>_<grid-name>.mp4; use distinct names "
                        "for separate runs into the same --output (or 'none' to skip it)")
    p.add_argument("--output", default="deliverables/videos")
    args = p.parse_args()

    models = []
    for spec in args.models:
        label, _, path = spec.partition("=")
        if not Path(path).is_file():
            print(f"[skip] {label}: {path} not found")
            continue
        m = YOLOWorld(path)
        m.set_classes(args.prompts)
        models.append((label, m))
    if not models:
        raise SystemExit("no models")

    cap = cv2.VideoCapture(str(Path(args.video).expanduser()))
    fps = cap.get(cv2.CAP_PROP_FPS) / args.stride
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writers = {} if args.no_full else {
        label: cv2.VideoWriter(str(out_dir / f"{stem}_{label}.mp4"), fourcc, fps, (w, h))
        for label, _ in models}
    grid = None
    if 2 <= len(models) <= 4 and args.grid_name != "none":
        grid = cv2.VideoWriter(str(out_dir / f"{stem}_{args.grid_name}.mp4"), fourcc, fps, (w, h))

    i = written = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and written >= args.max_frames):
            break
        i += 1
        if (i - 1) % args.stride:
            continue
        tiles = []
        for label, m in models:
            r = m.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device,
                          verbose=False)[0]
            annotated = label_frame(r.plot(), f"{label}  ({len(r.boxes)} dets)")
            if label in writers:
                writers[label].write(annotated)
            tiles.append(cv2.resize(annotated, (w // 2, h // 2)))
        if grid is not None:
            while len(tiles) < 4:
                tiles.append(np.zeros_like(tiles[0]))
            grid.write(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
        written += 1
        if written % 300 == 0:
            print(f"  {written} frames", flush=True)

    cap.release()
    for v in list(writers.values()) + ([grid] if grid is not None else []):
        v.release()
    print(f"Wrote {written} frames per video to {out_dir}")


if __name__ == "__main__":
    main()
