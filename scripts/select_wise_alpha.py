"""Pick a WiSE-FT alpha without looking at the open-vocab benchmark, then report OVB.

Rule (fixed before seeing OVB results for the seed runs): alpha* is the smallest
alpha whose VisDrone val mAP is within --tolerance of the fully fine-tuned model
(alpha = 1). Smaller alpha = closer to the base model = more open-vocab ability,
so this keeps as much of the base model as the VisDrone budget allows. OVB is
only read *after* alpha* is chosen, so it stays a held-out result.

With several sweep files (one per fine-tuning seed) it also reports mean +- std.

    python scripts/select_wise_alpha.py eval_results/sweep_wise_paper_e5*/sweep_results.json
"""
import argparse
import json
import re
import statistics
from pathlib import Path


def load_alphas(path):
    """{alpha: checkpoint entry} from a sweep_checkpoints.py run over wise_ft outputs."""
    data = json.loads(Path(path).read_text())
    out = {}
    for c in data["runs"]["individual"]["checkpoints"]:
        m = re.search(r"_a(\d+\.\d+)$", c["label"])
        if m:
            out[float(m.group(1))] = c
    return data["base"], out


def select(alphas, tolerance):
    ft_vd = alphas[1.0]["visdrone_mAP"]
    ok = [a for a, c in alphas.items() if c["visdrone_mAP"] >= ft_vd - tolerance]
    return min(ok)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("results", nargs="+", help="sweep_results.json files over wise_ft checkpoints")
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="Allowed VisDrone mAP drop vs alpha=1 (absolute)")
    p.add_argument("--output", default=None, help="Optional JSON summary path")
    args = p.parse_args()

    keys = [("visdrone_mAP", None), ("mAP", "openvocab"), ("mAP_novel", "openvocab")]
    rows = []
    for path in args.results:
        base, alphas = load_alphas(path)
        if 1.0 not in alphas:
            print(f"{path}: no alpha=1.00 checkpoint, skipping")
            continue
        a = select(alphas, args.tolerance)
        picked, ft = alphas[a], alphas[1.0]
        row = {"source": path, "alpha": a}
        for name, model in (("selected", picked), ("finetuned", ft)):
            row[f"{name}_visdrone_mAP"] = model["visdrone_mAP"]
            row[f"{name}_ov_mAP"] = model["openvocab"]["metrics"]["mAP"]
            row[f"{name}_ov_mAP_novel"] = model["openvocab"]["metrics"]["mAP_novel"]
        row["base_ov_mAP_novel"] = base["openvocab"]["metrics"].get("mAP_novel")
        rows.append(row)
        print(f"{path}\n  alpha*={a:.2f}  VisDrone {row['selected_visdrone_mAP']:.3f} "
              f"(ft {row['finetuned_visdrone_mAP']:.3f})  OV {row['selected_ov_mAP']:.3f} "
              f"(ft {row['finetuned_ov_mAP']:.3f})  OV novel {row['selected_ov_mAP_novel']:.3f} "
              f"(ft {row['finetuned_ov_mAP_novel']:.3f}, base {row['base_ov_mAP_novel']:.3f})")

    if len(rows) > 1:
        print(f"\nAcross {len(rows)} seeds (mean ± std):")
        summary = {}
        for k in [k for k in rows[0] if k not in ("source",) and rows[0][k] is not None]:
            vals = [r[k] for r in rows]
            summary[k] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals)}
            print(f"  {k:<28} {summary[k]['mean']:.3f} ± {summary[k]['std']:.3f}")
        rows.append({"summary": summary})
    if args.output:
        Path(args.output).write_text(json.dumps({"tolerance": args.tolerance, "rows": rows},
                                                indent=2))


if __name__ == "__main__":
    main()
