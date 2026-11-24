"""Paper version of the VisDrone vs. novel-OVB tradeoff figure (fig:pareto).

One point per row of tab:methods (seed mean, error bars = seed std where there are 3 seeds),
the WiSE-FT alpha
sweeps for the 5-epoch and low-LR 10-epoch models averaged over three seeds.

    python scripts/plot_pareto_paper.py --output eval_results/figures/pareto_novel_paper.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#222222", "#6b6b66"

# tab:methods rows: (label, novel, novel_std, visdrone, visdrone_std, color, marker)
ROWS = [
    ("Fine-tune, 5 ep", 0.125, 0.021, 0.302, 0.004, ORANGE, "o"),
    ("Fine-tune, 40 ep", 0.007, 0, 0.341, 0, ORANGE, "o"),
    ("Head only, 5 ep", 0.209, 0, 0.227, 0, AQUA, "s"),
    ("Low LR, 10 ep", 0.215, 0.007, 0.304, 0.001, BLUE, "o"),
    ("Low LR, 40 ep", 0.165, 0.005, 0.337, 0.001, BLUE, "o"),
    ("5 ep + WiSE-FT (α=0.7)", 0.209, 0.031, 0.299, 0.002, ORANGE, "D"),
    ("Recommended: low LR 10 ep\n+ WiSE-FT (α=0.8)", 0.253, 0.013, 0.298, 0.000, BLUE, "D"),
]
BASE = (0.222, 0.131)
# tab:lrgrid: (lr label, novel, visdrone)
LR_GRID = [("2.5e-5", 0.181, 0.271), ("1e-4", 0.202, 0.304), ("2e-4", 0.172, 0.319),
           ("7.14e-4", 0.131, 0.327), ("1.5e-3", 0.000, 0.323)]
WISE = {
    "5 ep": ("sweep_wise_paper_e5", ORANGE),
    "Low LR, 10 ep": ("sweep_wise_lr1e-4_e10_bias", BLUE),
}
# hand-placed label positions (data coords) so nothing collides; leader line to the point
LABEL_AT = {
    "Fine-tune, 5 ep": (0.125, 0.278, "center"),
    "Fine-tune, 40 ep": (0.020, 0.341, "left"),
    "Head only, 5 ep": (0.200, 0.227, "right"),
    "Low LR, 10 ep": (0.150, 0.322, "center"),
    "Low LR, 40 ep": (0.178, 0.341, "left"),
    "5 ep + WiSE-FT (α=0.7)": (0.140, 0.262, "center"),
    "Recommended: low LR 10 ep\n+ WiSE-FT (α=0.8)": (0.270, 0.258, "right"),
}


def wise_curve(results_dir, root):
    curves = []
    for suffix in ("", "_s1", "_s2"):
        data = json.loads((root / f"{results_dir}{suffix}" / "sweep_results.json").read_text())
        pts = sorted((float(c["label"].rsplit("_a", 1)[1]), c["openvocab"]["metrics"]["mAP_novel"],
                      c["visdrone_mAP"]) for r in data["runs"].values() for c in r["checkpoints"])
        curves.append(pts)
    arr = np.array(curves)  # seeds x alphas x (alpha, novel, visdrone)
    return arr.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval_results")
    ap.add_argument("--output", default="eval_results/figures/pareto_novel_paper.png")
    args = ap.parse_args()
    root = Path(args.results)

    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=300)
    ax.grid(True, color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # WiSE-FT alpha sweeps (seed mean), thin dashed
    for name, (d, col) in WISE.items():
        c = wise_curve(d, root)
        ax.plot(c[:, 1], c[:, 2], ls="--", lw=1.0, color=col, alpha=0.7, zorder=2)
        ax.scatter(c[1:-1, 1], c[1:-1, 2], s=7, color=col, alpha=0.7, zorder=2, lw=0)
    ax.annotate("WiSE-FT α sweeps\n(α = 0 → 1, dashed)", xy=(0.2255, 0.215), xytext=(0.150, 0.195),
                fontsize=7.5, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))

    # tab:methods rows
    for label, x, xs, y, ys, col, mk in ROWS:
        big = label.startswith("Recommended")
        if xs or ys:
            ax.errorbar(x, y, xerr=xs or None, yerr=ys or None, fmt="none", ecolor=col,
                        elinewidth=1.0, capsize=2, zorder=4)
        ax.scatter(x, y, s=95 if big else 42, marker=mk, color=col, edgecolor="white",
                   linewidth=1.2, zorder=5)
        tx, ty, ha = LABEL_AT[label]
        near = abs(tx - x) < 0.015 and abs(ty - y) < 0.006
        ax.annotate(label, (x, y), xytext=(tx, ty), fontsize=7.5, color=INK, ha=ha, va="center",
                    fontweight="bold" if big else "normal",
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85),
                    arrowprops=None if near else dict(arrowstyle="-", color=MUTED, lw=0.6,
                                                      shrinkA=2, shrinkB=5))

    ax.scatter(*BASE, s=150, marker="*", color=INK, edgecolor="white", linewidth=1.0, zorder=6)
    ax.annotate("Base (pretrained)", BASE, xytext=(-10, 0), textcoords="offset points",
                fontsize=7.5, color=INK, ha="right", va="center")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=ORANGE, marker="o", ls="none", label="lr 7.14e-4"),
        Line2D([], [], color=BLUE, marker="o", ls="none", label="lr 1e-4 (low LR)"),
        Line2D([], [], color=AQUA, marker="s", ls="none", label="Head only"),
        Line2D([], [], color=MUTED, marker="D", ls="none", label="WiSE-FT (selected α)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=False)
    ax.set_xlabel("OVB novel-category mAP@[.5:.95]  (open-vocabulary retention →)")
    ax.set_ylabel("VisDrone mAP@[.5:.95]  (aerial accuracy →)")
    ax.set_xlim(-0.008, 0.275)
    ax.set_ylim(0.115, 0.36)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out, out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
