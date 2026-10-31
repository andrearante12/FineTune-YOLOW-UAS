"""Zero-shot evaluation on standard open-vocabulary benchmarks (the ones YOLO-World
reports), to measure how much general open-vocab ability fine-tuning keeps.

Datasets (expected layout under datasets/):
  lvis   LVIS v1 minival (YOLO-World / GLIP split), official LVIS API: AP, APr/APc/APf
         lvis/lvis_v1_minival_inserted_image_name.json + coco/val2017/
  coco   COCO val2017, 80 classes, pycocotools
         coco/annotations/instances_val2017.json + coco/val2017/
  odinw_aerial  ODinW AerialMaritimeDrone (large, test split), pycocotools; aerial
         imagery, reported for all classes and for classes not in VisDrone
         odinw/AerialMaritimeDrone/large/test/

Prompts are the category names with "_" replaced by spaces. Model class index i is
the i-th category in ascending id order, mapped back to that id for scoring (category
ids are not contiguous in COCO). Native venv (needs the `lvis` package):

    python scripts/evaluate_standard_ov.py --models base=yolov8m-worldv2.pt \
        ft=5_epoch.pt --datasets lvis coco odinw_aerial
"""
import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLOWorld

# The official lvis API (0.5.3, unmaintained) still uses np.float, removed in NumPy 2;
# it was an alias of the builtin, so restoring the alias changes nothing.
if not hasattr(np, "float"):
    np.float = float

VISDRONE_CLASSES = {"pedestrian", "people", "bicycle", "car", "van",
                    "truck", "tricycle", "awning-tricycle", "bus", "motor"}

DATASETS = {
    "lvis": {"ann": "datasets/lvis/lvis_v1_minival_inserted_image_name.json",
             "images": "datasets/coco/val2017"},
    "coco": {"ann": "datasets/coco/annotations/instances_val2017.json",
             "images": "datasets/coco/val2017"},
    "odinw_aerial": {"ann": "datasets/odinw/AerialMaritimeDrone/large/test/annotations_without_background.json",
                     "images": "datasets/odinw/AerialMaritimeDrone/large/test"},
    # The official test split is only 7 images / 78 boxes. We never train on ODinW, so
    # every split is unseen: all_zero_shot.json pools train+valid+test (74 images,
    # 1151 boxes) for a less noisy zero-shot number. Built by the merge snippet logged
    # in docs/THOR_PROGRESS.md; file_name carries the split subdirectory.
    "odinw_aerial_all": {"ann": "datasets/odinw/AerialMaritimeDrone/large/all_zero_shot.json",
                         "images": "datasets/odinw/AerialMaritimeDrone/large"},
}


def prompt(name):
    return name.replace("_", " ").strip()


def load_dataset(name, limit):
    """Return (gt api object, categories sorted by id, image records)."""
    cfg = DATASETS[name]
    with contextlib.redirect_stdout(io.StringIO()):
        if name == "lvis":
            from lvis import LVIS
            gt = LVIS(cfg["ann"])
            cats = sorted(gt.load_cats(gt.get_cat_ids()), key=lambda c: c["id"])
            imgs = gt.load_imgs(sorted(gt.get_img_ids()))
        else:
            gt = COCO(cfg["ann"])
            used = {a["category_id"] for a in gt.dataset["annotations"]}
            # Roboflow exports carry a placeholder supercategory with no boxes; skip it
            cats = sorted((c for c in gt.loadCats(gt.getCatIds()) if c["id"] in used),
                          key=lambda c: c["id"])
            imgs = gt.loadImgs(sorted(gt.getImgIds()))
    if limit:
        imgs = imgs[:limit]
    return gt, cats, imgs


def image_path(name, img):
    if name == "lvis":
        # minival image records carry COCO file names (or only a coco_url)
        fname = img.get("file_name") or img["coco_url"].rsplit("/", 1)[-1]
        return Path(DATASETS[name]["images"]) / Path(fname).name
    return Path(DATASETS[name]["images"]) / img["file_name"]


def predict(model, name, imgs, cat_ids, args):
    dets, t0 = [], time.time()
    for i, img in enumerate(imgs):
        r = model.predict(str(image_path(name, img)), imgsz=args.imgsz, conf=args.conf,
                          iou=args.iou, max_det=args.max_det, device=args.device,
                          verbose=False)[0]
        if r.boxes is not None:
            for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                              r.boxes.cls.tolist()):
                dets.append({"image_id": img["id"], "category_id": cat_ids[int(c)],
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(imgs)} images ({time.time() - t0:.0f}s)", flush=True)
    return dets


def score(name, gt, cats, imgs, dets):
    img_ids = [im["id"] for im in imgs]
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        if name == "lvis":
            from lvis import LVISEval, LVISResults
            ev = LVISEval(gt, LVISResults(gt, dets, max_dets=300), "bbox")
            ev.params.img_ids = img_ids
            ev.run()
            out = {k: float(v) for k, v in ev.get_results().items()}
        else:
            def coco_stats(cat_ids):
                if not dets:
                    return [0.0] * 12
                ev = COCOeval(gt, gt.loadRes(dets), "bbox")
                ev.params.imgIds, ev.params.catIds = img_ids, cat_ids
                ev.evaluate(); ev.accumulate(); ev.summarize()
                return ev.stats
            s = coco_stats([c["id"] for c in cats])
            out = {"AP": s[0], "AP50": s[1], "AP75": s[2], "APs": s[3], "APm": s[4],
                   "APl": s[5]}
            if name.startswith("odinw_aerial"):
                novel = [c["id"] for c in cats if prompt(c["name"]) not in VISDRONE_CLASSES]
                s = coco_stats(novel)
                out.update(AP_novel=s[0], AP50_novel=s[1])
    return {k: float(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", nargs="+", required=True,
                   help="label=weights pairs; missing weights are skipped")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--imgsz", type=int, default=640, help="640 = YOLO-World's protocol")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="0")
    p.add_argument("--limit", type=int, default=0, help="First N images only (smoke test)")
    p.add_argument("--output", default="eval_results/standard_ov")
    args = p.parse_args()

    models = []
    for spec in args.models:
        label, _, path = spec.partition("=")
        if Path(path).is_file():
            models.append((label, path))
        else:
            print(f"[skip] {label}: {path} not found")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for ds in args.datasets:
        gt, cats, imgs = load_dataset(ds, args.limit)
        cat_ids = [c["id"] for c in cats]
        prompts = [prompt(c["name"]) for c in cats]
        print(f"\n=== {ds}: {len(imgs)} images, {len(cats)} classes", flush=True)
        for label, path in models:
            key = f"{ds}/{label}"
            if key in results and not args.limit:
                print(f"  {label}: cached"); continue
            t0 = time.time()
            model = YOLOWorld(path)
            model.set_classes(prompts)
            dets = predict(model, ds, imgs, cat_ids, args)
            metrics = score(ds, gt, cats, imgs, dets)
            metrics.update(weights=path, n_images=len(imgs), n_dets=len(dets),
                           imgsz=args.imgsz, seconds=round(time.time() - t0))
            results[key] = metrics
            results_path.write_text(json.dumps(results, indent=2))
            print(f"  {label:<18} " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                                                if isinstance(v, float)), flush=True)
            del model


if __name__ == "__main__":
    main()
