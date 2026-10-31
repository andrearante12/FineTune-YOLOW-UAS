"""Plot the VisDrone-vs-open-vocab tradeoff from one or more sweep_results.json files.

Each run (epoch sweep, WiSE-FT alphas, ...) is drawn as a connected series; the
base model is a star and the Pareto frontier over all points is shaded.

    python scripts/plot_pareto.py eval_results/sweep_*/sweep_results.json \
        --ov-metric mAP_novel --output eval_results/pareto.png
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def series_key(results_dir, label):
    """Group points into lines: 'e5' -> the run (results dir), 'wise_5_epoch_a0.50' ->
    'wise_5_epoch', anything else stands alone."""
    stem = re.sub(r"_?(a\d+\.\d+|e\d+)$", "", label)
    name = results_dir.removeprefix("sweep_")
    if not stem or stem == "wise_last":  # generic labels: the results dir names the run
        return name
    return stem


def load_points(paths, ov_metric):
    """Return base point and {series label: [(vd, ov, point label), ...]}."""
    base, series = None, {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        b = data["base"]
        base = base or (b["visdrone_mAP"], b["openvocab"]["metrics"][ov_metric])
        for run in data["runs"].values():
            for c in run["checkpoints"]:
                label = c.get("label") or f"e{c['epoch']}"
                key = series_key(Path(path).parent.name, label)
                series.setdefault(key, []).append(
                    (c["visdrone_mAP"], c["openvocab"]["metrics"][ov_metric], label))
    return base, series


def pareto_front(points):
    """Points not dominated on both axes (higher is better for both)."""
    front = [p for p in points
             if not any(q[0] >= p[0] and q[1] >= p[1] and q[:2] != p[:2] for q in points)]
    return sorted(front)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", nargs="+", help="sweep_results.json files")
    parser.add_argument("--ov-metric", default="mAP_novel",
                        help="Key in openvocab.metrics: mAP, mAP_50, mAP_novel, mAP_50_novel")
    parser.add_argument("--annotate", action="store_true", help="Label every point")
    parser.add_argument("--output", default="eval_results/pareto.png")
    args = parser.parse_args()

    base, series = load_points(args.results, args.ov_metric)
    fig, ax = plt.subplots(figsize=(7, 5))

    all_pts = [base + ("base",)]
    for name, pts in series.items():
        xs, ys = [p[1] for p in pts], [p[0] for p in pts]
        if len(pts) == 1:
            ax.scatter(xs, ys, marker="D", s=40, zorder=4, label=name)
        else:
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.2, label=name)
        if args.annotate:
            for vd, ov, lab in pts:
                short = lab if len(pts) == 1 else re.sub(r".*_(a\d+\.\d+|e\d+)$", r"\1", lab)
                ax.annotate(short, (ov, vd), fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
        all_pts += pts

    front = pareto_front(all_pts)
    ax.plot([p[1] for p in front], [p[0] for p in front], color="0.8", linewidth=6,
            solid_capstyle="round", zorder=0, label="Pareto frontier")
    ax.scatter([base[1]], [base[0]], marker="*", s=180, color="black", zorder=5,
               label="base (yolov8m-worldv2)")

    ax.set_xlabel(f"Open-vocab {args.ov_metric} (OVB)")
    ax.set_ylabel("VisDrone mAP@[.5:.95]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")
    print("Frontier:", ", ".join(f"{p[2]} (vd {p[0]:.3f}, ov {p[1]:.3f})" for p in front))


if __name__ == "__main__":
    main()
