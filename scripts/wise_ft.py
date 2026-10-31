"""WiSE-FT: weight-space interpolation between the base and a fine-tuned YOLO-World.

theta(alpha) = (1 - alpha) * theta_base + alpha * theta_ft

alpha=0 reproduces the base model and alpha=1 the fine-tuned one. Intermediate
alphas often recover open-vocab ability at little VisDrone cost (Wortsman et al.,
"Robust fine-tuning of zero-shot models", CVPR 2022). Outputs one ultralytics-
loadable checkpoint per alpha, ready for scripts/sweep_checkpoints.py --checkpoint.
"""
import argparse
import copy
from pathlib import Path

import torch


def load_model(path):
    """Return (checkpoint dict, nn.Module) the way ultralytics picks weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt.get("ema") or ckpt["model"]
    return ckpt, model.float()


def interpolate(base_sd, ft_sd, alpha):
    """Interpolate float tensors; take everything else (e.g. BN counters) from ft."""
    out = {}
    for k, v_ft in ft_sd.items():
        # Contrastive-head biases are 0-d in older checkpoints and shape (1,) in
        # newer ones; same values, so align shapes.
        v_base = base_sd[k].reshape(v_ft.shape)
        if v_ft.is_floating_point():
            out[k] = torch.lerp(v_base, v_ft, alpha)
        else:
            out[k] = v_ft.clone()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-weights", default="yolov8m-worldv2.pt")
    parser.add_argument("--finetuned-weights", default="5_epoch.pt")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--output", default="runs/wise_ft",
                        help="Directory for wise_<ft-stem>_a<alpha>.pt checkpoints")
    args = parser.parse_args()

    _, base = load_model(args.base_weights)
    ft_ckpt, ft = load_model(args.finetuned_weights)
    base_sd, ft_sd = base.state_dict(), ft.state_dict()

    missing = set(ft_sd) ^ set(base_sd)
    if missing:
        raise SystemExit(f"State dict keys differ between models: {sorted(missing)[:10]}")
    mismatched = [k for k in ft_sd if ft_sd[k].numel() != base_sd[k].numel()]
    if mismatched:
        raise SystemExit(f"Shape mismatch (different architectures?): {mismatched[:10]}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.finetuned_weights).stem

    for alpha in args.alphas:
        model = copy.deepcopy(ft)
        model.load_state_dict(interpolate(base_sd, ft_sd, alpha))
        # Keep the fine-tuned checkpoint's metadata (train_args, names, ...) but
        # drop training state so ultralytics loads `model` directly.
        ckpt = {k: v for k, v in ft_ckpt.items()
                if k not in ("ema", "optimizer", "updates", "best_fitness")}
        ckpt.update(model=model.half(), ema=None, optimizer=None, epoch=-1,
                    wise_ft={"alpha": alpha, "base": args.base_weights,
                             "finetuned": args.finetuned_weights})
        path = out_dir / f"wise_{stem}_a{alpha:.2f}.pt"
        torch.save(ckpt, path)
        print(f"alpha={alpha:.2f} -> {path}")


if __name__ == "__main__":
    main()
