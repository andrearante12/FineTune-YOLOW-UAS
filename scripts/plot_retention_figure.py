"""Paper figure: VisDrone accuracy vs. open-vocabulary retention (OVB novel and LVIS).

Final models only (the rows of the paper's retention-methods table), mean +/- std over seeds,
plus the two WiSE-FT interpolation sweeps (alpha 0 -> 1, mean over three seeds) on the OVB panel.

    python scripts/plot_retention_figure.py . paper/retention_tradeoff
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1])
EV = ROOT / "eval_results"
OUT = Path(sys.argv[2])

STANDARD, LOWLR, HEAD = "#eb6834", "#2a78d6", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# name, color, marker, VisDrone (mean, std), OVB novel (mean, std), LVIS key in standard_ov
# VisDrone / OVB values are the paper's retention-methods table (verified against eval_results).
MODELS = [
    ("Standard FT, 5 ep",    STANDARD, "o", (0.302, 0.004), (0.125, 0.021), "ft5_paper"),
    ("Standard FT, 40 ep",   STANDARD, "^", (0.341, 0.0),   (0.007, 0.0),   "paper_e40"),
    ("Standard FT, 5 ep + WiSE-FT", STANDARD, "s", (0.299, 0.002), (0.209, 0.031), "wise5_a0.7"),
    ("Head only, 5 ep",      HEAD,     "D", (0.227, 0.0),   (0.209, 0.0),   "freeze22_e5_bias"),
    ("Low LR, 10 ep",        LOWLR,    "o", (0.304, 0.001), (0.215, 0.007), "lr1e-4_e10_bias"),
    ("Low LR, 40 ep",        LOWLR,    "^", (0.337, 0.001), (0.165, 0.005), "lr1e-4_e40_bias"),
    ("Recommended", LOWLR, "s", (0.298, 0.0),   (0.253, 0.013), "wise_lr1e-4_e10_bias"),
]
BASE_VD, BASE_OVB = 0.131, 0.222



def wise_curve(prefix):
    """Mean (OVB novel, VisDrone) over the three seeds of a WiSE sweep, alpha = 0.0 ... 1.0."""
    runs = []
    for sfx in ["", "_s1", "_s2"]:
        d = json.load(open(EV / f"sweep_wise_{prefix}{sfx}" / "sweep_results.json"))
        c = d["runs"]["individual"]["checkpoints"]
        pts = c.values() if isinstance(c, dict) else c
        runs.append([(p["openvocab"]["metrics"]["mAP_novel"], p["visdrone"]["metrics"]["mAP"]) for p in pts])
    return np.array(runs).mean(axis=0)


def style(ax, xlabel):
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def draw_points(ax, xs, label_rec):
    for (name, col, mk, vd, _, _), x in zip(MODELS, xs):
        rec = name.startswith("Recommended")
        ax.errorbar(x[0], vd[0], fmt=mk, color=col,
                    ms=11 if rec else 7.5, mec=INK if rec else "white", mew=1.2 if rec else 0.8,
                    ecolor=col, elinewidth=1, capsize=2, zorder=4, label=name)
        if rec:
            ax.annotate("Recommended", (x[0], vd[0]), xytext=label_rec, textcoords="offset points",
                        ha="center", fontsize=10, color=INK, fontweight="bold")


def draw_base(ax, x):
    ax.plot(x, BASE_VD, "*", color=INK, ms=14, mec="white", mew=0.8, zorder=5)
    ax.annotate("Base", (x, BASE_VD), xytext=(-8, -3), textcoords="offset points",
                ha="right", fontsize=10, color=INK)


def main():
    lvis = json.load(open(EV / "standard_ov" / "results.json"))
    lv = lambda k: lvis[f"lvis/{k}"]["AP"]

    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 11.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 4.0), sharey=True,
                               gridspec_kw={"width_ratios": [1.35, 1]})

    for prefix, col in [("paper_e5", STANDARD), ("lr1e-4_e10_bias", LOWLR)]:
        cv = wise_curve(prefix)
        a.plot(cv[:, 0], cv[:, 1], "-", color=col, lw=1, alpha=0.5, zorder=2)
        a.plot(cv[:, 0], cv[:, 1], ".", color=col, ms=3, alpha=0.5, zorder=2)
    draw_points(a, [m[4] for m in MODELS], (0, 12))
    draw_base(a, BASE_OVB)
    a.annotate("WiSE-FT sweeps,\n$\\alpha$ = 0 $\\to$ 1", (0.244, 0.183), fontsize=9.5, color=INK2)
    style(a, "OVB novel mAP")
    a.set_ylabel("VisDrone mAP", color=INK)
    a.set_title("(a) Custom UAS benchmark (OVB)", fontsize=11, color=INK, loc="left")
    a.set_xlim(-0.02, 0.29)
    a.set_ylim(0.11, 0.36)

    draw_points(b, [(lv(m[5]), 0.0) for m in MODELS], (0, -16))
    draw_base(b, lv("base"))
    style(b, "LVIS zero-shot AP")
    b.set_title("(b) LVIS minival", fontsize=11, color=INK, loc="left")
    b.set_xlim(-0.01, 0.20)

    h, l = a.get_legend_handles_labels()
    h = [x[0] if isinstance(x, tuple) else x for x in h]
    base_h = plt.Line2D([], [], marker="*", color=INK, ls="", ms=10, mec="white")
    fig.legend([base_h] + h, ["Base (no fine-tuning)"] + l, loc="lower center", ncol=3,
               fontsize=10, frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(w_pad=1.5, rect=(0, 0.16, 1, 1))
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    for m in MODELS:
        print(f"  {m[0]:<22} LVIS {lv(m[5]):.3f}")
    print("  base LVIS", round(lv("base"), 3))


if __name__ == "__main__":
    main()
