"""Evaluate a trained VAE: metrics, generation, figures and latent-code export.

    python -m vae.evaluate --run-dir outputs/vae/z128_beta0.1

Extra over the AE evaluation: KL per image, number of active latent units, random samples from
the prior, a sample-diversity score and per-dimension latent traversals.
Latents stored for visualisation are the posterior means (plus log-variances).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common.dataset import SliceDataset
from common.evaluation import (GaussianLatentSampler, encode_dataset, evaluate_split, interpolation_figure,
                               perturbation_figure, realism_metrics, reconstruction_figure, save_latents,
                               summarize)
from common.utils import (count_parameters, get_device, save_comparison_grid, save_image_grid,
                          save_json, seed_everything)
from vae.model import VAE


@torch.no_grad()
def traversal_figure(model: VAE, mu: np.ndarray, device, path: Path, n_dims: int = 6,
                     values=(-3, -2, -1, 0, 1, 2, 3)) -> None:
    """Decode z = 0 with one latent dimension swept; dims chosen as the most active (highest Var[mu])."""
    k = min(n_dims, mu.shape[1])
    dims = np.argsort(-mu.var(0))[:k]
    rows = []
    for d in dims:
        z = torch.zeros(len(values), mu.shape[1], device=device)
        z[:, d] = torch.tensor(values, dtype=torch.float32, device=device)
        rows.append(model.decode(z))
    save_comparison_grid(rows, [f"z[{d}]" for d in dims], path, col_titles=[f"{v:+d}" for v in values])


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the VAE")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--latent-splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    p.add_argument("--max-latents", type=int, default=6000)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--n-fid", type=int, default=10000, help="max real/generated images for FID")
    p.add_argument("--no-fid", action="store_true", help="skip FID / generation metrics (debug)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, ckpt = VAE.from_checkpoint(args.run_dir / "checkpoints" / args.checkpoint, device)
    cfg = ckpt["train_config"]
    seed_everything(cfg["seed"])
    data_dir = args.data_dir or cfg["data_dir"]
    fig_dir, lat_dir = args.run_dir / "figures", args.run_dir / "latents"
    fig_dir.mkdir(exist_ok=True); lat_dir.mkdir(exist_ok=True)

    def forward_fn(x):
        mu, logvar = model.encode(x)
        return model.decode(mu), mu, logvar

    results = {}
    for split in sorted(set([args.split] + args.latent_splits)):
        ds = SliceDataset(data_dir, split)
        per, mu, logvar = evaluate_split(ds, forward_fn, device, args.batch_size, args.num_workers)
        results[split] = (ds, per, mu, logvar)
        if split in args.latent_splits:
            save_latents(lat_dir / f"{split}.npz", ds, mu, per, logvar=logvar, max_points=args.max_latents)

    ds, per, mu, logvar = results[args.split]
    kl = -0.5 * (1 + logvar - mu ** 2 - np.exp(logvar)).sum(1)
    gauss = GaussianLatentSampler(encode_dataset(SliceDataset(data_dir, "train"), lambda x: model.encode(x)[0], device))
    samplers = {"prior": lambda k: model.sample(k, device),                       # z ~ N(0, I)
                "prior_t07": lambda k: model.sample(k, device, temperature=0.7),  # z ~ N(0, 0.7^2 I)
                "gauss": lambda k: model.decode(gauss(k))}                        # same sampler as the AE
    torch.manual_seed(0)
    samples = model.sample(args.n_samples, device)
    metrics = {"model": "vae", "run_name": args.run_dir.name, "latent_dim": cfg["latent_dim"],
               "beta": float(ckpt["beta"]), "split": args.split, "checkpoint": args.checkpoint,
               "epoch": ckpt["epoch"], "n_images": len(ds),
               "n_subjects": int(len(set(ds.subject_ids.tolist()))), "n_params": count_parameters(model),
               **summarize(per),
               "kl_mean_nats": float(kl.mean()),
               "active_units": int((mu.var(0) > 0.01).sum()),   # Burda et al. 2016 criterion
               }
    if not args.no_fid:
        metrics.update(realism_metrics(ds, data_dir, args.split, device, model.reconstruct, samplers,
                                       n_fid=args.n_fid))
        metrics["sample_diversity"] = metrics["diversity_prior"]
    save_json(metrics, args.run_dir / f"metrics_{args.split}.json")

    reconstruction_figure(lambda x: model.reconstruct(x), ds, device,
                          fig_dir / f"reconstructions_{args.split}.png", "VAE recon")
    perturbation_figure(lambda x: model.encode(x)[0], model.decode, ds, mu, device,
                        fig_dir / f"latent_perturbation_{args.split}.png")
    interpolation_figure(model.interpolate, ds, device, fig_dir / f"latent_interpolation_{args.split}.png")
    save_image_grid(samples, fig_dir / "generated_samples.png", n_cols=8,
                    title=f"VAE samples, z ~ N(0, I)  (z={cfg['latent_dim']}, beta={ckpt['beta']:g})")
    traversal_figure(model, mu, device, fig_dir / "latent_traversal.png")
    temps, rows = (0.5, 0.7, 1.0), []
    for t in temps:
        torch.manual_seed(0)  # same noise per row -> rows differ only by temperature
        rows.append(model.sample(10, device, temperature=t))
    save_comparison_grid(rows, [f"T={t:g}" for t in temps], fig_dir / "samples_temperature.png")
    torch.manual_seed(0)
    save_image_grid(samplers["gauss"](64), fig_dir / "generated_samples_gaussfit.png", n_cols=8,
                    title=f"VAE samples, z ~ Gaussian fit to posterior means  (z={cfg['latent_dim']}, beta={ckpt['beta']:g})")

    print(f"[VAE z={cfg['latent_dim']} beta={ckpt['beta']:g}] {args.split}: MSE {metrics['mse_mean']:.5f} | "
          f"PSNR {metrics['psnr_mean']:.2f} dB | SSIM {metrics['ssim_mean']:.4f} | "
          f"KL {metrics['kl_mean_nats']:.1f} nats | active units {metrics['active_units']}/{cfg['latent_dim']} "
          + (f"| FID recon {metrics['fid_recon']:.1f} | FID prior {metrics['fid_prior']:.1f} "
             f"| FID prior T0.7 {metrics['fid_prior_t07']:.1f} | FID gauss {metrics['fid_gauss']:.1f} "
             f"(real floor {metrics['fid_real_floor']:.1f}) | diversity {metrics['diversity_prior']:.3f} "
             f"(real {metrics['diversity_real']:.3f})" if "fid_recon" in metrics else ""))


if __name__ == "__main__":
    main()
