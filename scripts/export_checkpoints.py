"""Copy selected run checkpoints to checkpoints/ as clean, deployable weights.

Each label=weights pair is saved as checkpoints/<label>.pt via ultralytics'
strip_optimizer (EMA weights in FP16, no optimizer state: ~50 MB instead of ~280 MB),
and checkpoints/index.json records source, size and sha256 so a file can be traced
back to the run (and metrics) it came from.

    python scripts/export_checkpoints.py wise5_a0.7=runs/wise_ft/wise_5_epoch_a0.70.pt
"""
import argparse
import hashlib
import json
from pathlib import Path

from ultralytics.utils.torch_utils import strip_optimizer


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("models", nargs="+", help="label=weights pairs")
    p.add_argument("--output", default="checkpoints")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    for spec in args.models:
        label, _, src = spec.partition("=")
        if not Path(src).is_file():
            print(f"[skip] {label}: {src} not found")
            continue
        dst = out_dir / f"{label}.pt"
        strip_optimizer(src, s=str(dst))
        index[label] = {"source": src, "file": dst.name,
                        "size_mb": round(dst.stat().st_size / 2**20, 1),
                        "sha256": sha256(dst)}
        print(f"{label}: {src} -> {dst} ({index[label]['size_mb']} MB)")
    index_path.write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
