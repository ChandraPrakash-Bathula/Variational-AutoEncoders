"""Aggregate AE / VAE runs into a results table, trend plots and a side-by-side reconstruction figure.

    python -m analysis.compare_models                                   # table + plots for all runs
    python -m analysis.compare_models --ae-run outputs/ae/z128 --vae-run outputs/vae/z128_beta0.1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMNS = ["model", "latent_dim", "beta", "mse_mean", "psnr_mean", "masked_psnr_mean", "ssim_mean",
           "fid_recon", "fid_gauss", "fid_prior", "fid_prior_t07", "fid_real_floor",
           "diversity_gauss", "diversity_prior", "diversity_real",
           "kl_mean_nats", "active_units", "epoch", "n_params"]


def collect(output_dir: Path, split: str) -> pd.DataFrame:
    rows = []
    for f in sorted(output_dir.glob(f"*/*/metrics_{split}.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    if not rows:
        raise SystemExit(f"No metrics_{split}.json found under {output_dir}/*/*/. Run the evaluate scripts first.")
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[COLUMNS].sort_values(["model", "beta", "latent_dim"]).reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:.4g}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def plot_trends(df: pd.DataFrame, path: Path, title: str = "") -> None:
    """Reconstruction (top) and generation (bottom) metrics vs latent size, one line per model."""
    recon = [("psnr_mean", "PSNR (dB) \u2191"), ("masked_psnr_mean", "foreground PSNR (dB) \u2191"),
             ("ssim_mean", "SSIM \u2191")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    groups = list(df.groupby(["model", df["beta"].fillna(-1)]))
    color = {key: f"C{i}" for i, (key, _) in enumerate(groups)}   # one colour per model, in every panel

    def name(model, beta):
        return "AE" if model == "ae" else f"VAE \u03b2={beta:g}"

    for (key, t), ax in zip(recon, axes[0]):
        for (model, beta), g in groups:
            g = g.sort_values("latent_dim")
            ax.plot(g["latent_dim"], g[key], "o-", color=color[(model, beta)], label=name(model, beta))
        ax.set_title(t)
    ax = axes[1, 0]
    for (model, beta), g in groups:
        g = g.sort_values("latent_dim")
        ax.plot(g["latent_dim"], g["fid_recon"], "o-", color=color[(model, beta)], label=name(model, beta))
    ax.set_title("FID of reconstructions \u2193")
    ax = axes[1, 1]
    for (model, beta), g in groups:
        g = g.sort_values("latent_dim")
        ax.plot(g["latent_dim"], g["fid_gauss"], "s--", color=color[(model, beta)],
                label=f"{name(model, beta)}: Gaussian-fit samples")
        if model == "vae":
            ax.plot(g["latent_dim"], g["fid_prior"], "o-", color=color[(model, beta)],
                    label=f"{name(model, beta)}: prior samples")
    ax.set_title("FID of generated samples \u2193  (solid = prior, dashed = Gaussian fit)"); ax.set_yscale("log")
    ax = axes[1, 2]
    for (model, beta), g in groups:
        g = g.sort_values("latent_dim")
        key = "diversity_prior" if model == "vae" else "diversity_gauss"
        ax.plot(g["latent_dim"], g[key], "o-", color=color[(model, beta)],
                label=f"{name(model, beta)} ({'prior' if model == 'vae' else 'Gaussian fit'})")
    if df["diversity_real"].notna().any():
        ax.axhline(df["diversity_real"].mean(), color="k", ls=":", label="real data")
    ax.set_title("sample diversity (1 - mean pairwise SSIM)")
    for ax in axes.flat:
        ax.set_xscale("log", base=2); ax.set_xlabel("latent dimension"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    floor = df["fid_real_floor"].dropna()
    if len(floor):
        for a in axes[1, :2]:
            a.axhline(floor.mean(), color="k", ls=":", lw=1)
            a.text(0.01, 0.02, f"real-vs-real FID {floor.mean():.1f}", transform=a.transAxes, fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def side_by_side(ae_run: Path, vae_run: Path, split: str, n: int, path: Path) -> None:
    import torch
    from autoencoder.model import Autoencoder
    from common.dataset import SliceDataset
    from common.evaluation import pick_images
    from common.utils import error_map, save_comparison_grid
    from vae.model import VAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, ae_ckpt = Autoencoder.from_checkpoint(ae_run / "checkpoints" / "best.pt", device)
    vae, vae_ckpt = VAE.from_checkpoint(vae_run / "checkpoints" / "best.pt", device)
    ds = SliceDataset(ae_ckpt["train_config"]["data_dir"], split)
    x = pick_images(ds, n, device)
    with torch.no_grad():
        ra, rv = ae.reconstruct(x), vae.reconstruct(x)
    save_comparison_grid(
        [x, ra, rv, error_map(x, ra), error_map(x, rv)],
        ["input", f"AE z={ae.cfg.latent_dim}", f"VAE z={vae.cfg.latent_dim}", "|AE err| x4", "|VAE err| x4"], path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/analysis"))
    p.add_argument("--split", default="test")
    p.add_argument("--ae-run", type=Path, default=None)
    p.add_argument("--vae-run", type=Path, default=None)
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--title", default="", help="figure title, e.g. the dataset name")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = collect(args.output_dir, args.split)
    df.to_csv(args.out_dir / "results_table.csv", index=False)
    (args.out_dir / "results_table.md").write_text(to_markdown(df) + "\n")
    plot_trends(df, args.out_dir / "metrics_vs_latent_dim.png", args.title)
    print(to_markdown(df))
    if args.ae_run and args.vae_run:
        side_by_side(args.ae_run, args.vae_run, args.split, args.num_images,
                     args.out_dir / "ae_vs_vae_reconstructions.png")
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
