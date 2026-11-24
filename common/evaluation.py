"""Evaluation helpers shared by autoencoder/evaluate.py and vae/evaluate.py."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.dataset import SliceDataset
from common.metrics import reconstruction_metrics
from common.utils import error_map, save_comparison_grid, save_image_grid
from common.metrics import ssim_per_image

METRIC_KEYS = ("mse", "masked_mse", "psnr", "masked_psnr", "ssim")


@torch.no_grad()
def evaluate_split(dataset: SliceDataset, forward_fn: Callable, device: torch.device,
                   batch_size: int = 128, num_workers: int = 2
                   ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Run a model over a dataset in fp32.

    forward_fn(x) must return (reconstruction, latent_vector, logvar_or_None), where the latent
    is the deterministic code (AE) or the posterior mean (VAE).
    Returns per-image metric arrays, latents (N x D) and logvars (N x D or None).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    per = {k: [] for k in METRIC_KEYS}
    zs, lvs = [], []
    for x in loader:
        x = x.to(device)
        recon, z, logvar = forward_fn(x)
        m = reconstruction_metrics(x, recon.float())
        for k in METRIC_KEYS:
            per[k].append(m[k].cpu())
        zs.append(z.float().cpu())
        if logvar is not None:
            lvs.append(logvar.float().cpu())
    per_np = {k: torch.cat(v).numpy() for k, v in per.items()}
    return per_np, torch.cat(zs).numpy(), (torch.cat(lvs).numpy() if lvs else None)


def summarize(per_image: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in per_image.items():
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std())
    return out


def save_latents(path: Path, dataset: SliceDataset, latents: np.ndarray,
                 per_image: Dict[str, np.ndarray], logvar: Optional[np.ndarray] = None,
                 max_points: Optional[int] = None, seed: int = 0) -> None:
    """Persist latents + labels used for visualisation (random subsample if max_points is set)."""
    idx = np.arange(len(latents))
    if max_points and len(idx) > max_points:
        idx = np.sort(np.random.RandomState(seed).choice(idx, max_points, replace=False))
    payload = dict(latents=latents[idx], subject_ids=dataset.subject_ids[idx],
                   mse=per_image["mse"][idx], ssim=per_image["ssim"][idx], index=idx)
    payload.update({k: v[idx] for k, v in dataset.labels.items()})   # slice_pos / attr_* / foreground ...
    if logvar is not None:
        payload["logvar"] = logvar[idx]
    np.savez_compressed(path, **payload)


def pick_images(dataset: SliceDataset, n: int, device: torch.device) -> torch.Tensor:
    idx = np.linspace(0, len(dataset) - 1, n).astype(int)
    return torch.stack([dataset[int(i)] for i in idx]).to(device)


@torch.no_grad()
def reconstruction_figure(encode_decode: Callable, dataset: SliceDataset, device, path: Path,
                          label: str, n: int = 10) -> None:
    x = pick_images(dataset, n, device)
    recon = encode_decode(x)
    save_comparison_grid([x, recon, error_map(x, recon)], ["input", label, "|error| x4"], path)


@torch.no_grad()
def perturbation_figure(encode_fn: Callable, decode_fn: Callable, dataset: SliceDataset,
                        latents: np.ndarray, device, path: Path,
                        scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
                        n_images: int = 4, seed: int = 0) -> None:
    """Add Gaussian noise (scaled by the per-dimension std of the latent codes) to z and decode.

    Shows how fragile / smooth the decoder is around a data point.
    """
    x = pick_images(dataset, n_images, device)
    z = encode_fn(x)
    std = torch.from_numpy(latents.std(0)).to(device)
    gen = torch.Generator().manual_seed(seed)
    decoded = [decode_fn(z)]
    for s in scales:
        eps = torch.randn(z.shape, generator=gen).to(device)
        decoded.append(decode_fn(z + s * std * eps))
    rows = [torch.cat([x[i:i + 1]] + [d[i:i + 1] for d in decoded]) for i in range(n_images)]
    cols = ["input", "decode(z)"] + [f"+{s:g}\u03c3 noise" for s in scales]
    save_comparison_grid(rows, [f"slice {i + 1}" for i in range(n_images)], path, col_titles=cols)


@torch.no_grad()
def interpolation_figure(interp_fn: Callable, dataset: SliceDataset, device, path: Path,
                         n_pairs: int = 3, steps: int = 8) -> None:
    """Latent interpolation between pairs of test slices (picked far apart in the dataset)."""
    imgs = pick_images(dataset, 2 * n_pairs, device)
    x1, x2 = imgs[:n_pairs], imgs[n_pairs:]
    out = interp_fn(x1, x2, steps)  # (steps, B, C, H, W)
    rows = [out[:, b] for b in range(n_pairs)]
    cols = [f"\u03b1={a:.2f}" for a in np.linspace(0, 1, steps)]
    save_comparison_grid(rows, [f"pair {b + 1}" for b in range(n_pairs)], path, col_titles=cols)


# --------------------------------------------------------------------------- #
# Generation / realism (shared by AE and VAE evaluation)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def diversity(images: torch.Tensor, n_pairs: int = 2000, seed: int = 0) -> float:
    """1 - mean SSIM over random pairs (0 = identical images). Compare against the same score on real data."""
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, len(images), (n_pairs,), generator=g)
    j = torch.randint(0, len(images), (n_pairs,), generator=g)
    keep = i != j
    out = []
    for a in range(0, int(keep.sum()), 500):
        ii, jj = i[keep][a:a + 500], j[keep][a:a + 500]
        out.append(ssim_per_image(images[ii].float(), images[jj].float()))
    return float(1.0 - torch.cat(out).mean())


@torch.no_grad()
def encode_dataset(dataset: SliceDataset, encode_fn: Callable, device, n_max: int = 20000,
                   batch_size: int = 256, seed: int = 0) -> torch.Tensor:
    idx = np.arange(len(dataset))
    if len(idx) > n_max:
        idx = np.sort(np.random.RandomState(seed).choice(idx, n_max, replace=False))
    zs = []
    for a in range(0, len(idx), batch_size):
        x = torch.stack([dataset[int(k)] for k in idx[a:a + batch_size]]).to(device)
        zs.append(encode_fn(x).float())
    return torch.cat(zs)


class GaussianLatentSampler:
    """Full-covariance Gaussian fitted to encoded TRAINING codes (AE codes / VAE posterior means).

    Gives the deterministic AE a generative mode (Ghosh et al., 2020, "From variational to
    deterministic autoencoders") and lets AE and VAE be compared with the same sampler.
    """

    def __init__(self, z: torch.Tensor, seed: int = 0):
        z = z.double()
        self.mu = z.mean(0)
        cov = torch.cov(z.T) if z.shape[1] > 1 else z.var().reshape(1, 1)
        cov = cov + 1e-6 * cov.diagonal().mean() * torch.eye(len(cov), dtype=cov.dtype, device=cov.device)
        self.L = torch.linalg.cholesky(cov)
        self.gen = torch.Generator(device=z.device).manual_seed(seed)

    def __call__(self, n: int) -> torch.Tensor:
        eps = torch.randn(n, len(self.mu), generator=self.gen, device=self.mu.device, dtype=self.mu.dtype)
        return (self.mu + eps @ self.L.T).float()


@torch.no_grad()
def realism_metrics(dataset: SliceDataset, data_dir, split: str, device, recon_fn: Callable,
                    samplers: Dict[str, Callable[[int], torch.Tensor]], n_fid: int = 10000,
                    n_div: int = 1000) -> Dict[str, float]:
    """FID of reconstructions and of each sampler's images vs real `split` images + diversity scores.

    samplers: name -> fn(k) returning k generated images. Keys become fid_<name> / diversity_<name>.
    """
    from common import fid as F

    ref, idx, floor = F.real_reference(data_dir, split, n_fid, device)
    out = {"fid_n": int(len(idx)), "fid_real_floor": floor}
    rec = F.inception_features((recon_fn(b.to(device)) for b in F.image_batches(dataset, idx)), device)
    out["fid_recon"] = F.frechet(ref, F.stats(rec))
    real = torch.stack([dataset[int(k)] for k in F.fid_subset(len(dataset), n_div, seed=2)])
    out["diversity_real"] = diversity(real)
    for name, gen in samplers.items():
        feats = F.inception_features(F.generated_batches(gen, len(idx)), device)
        out[f"fid_{name}"] = F.frechet(ref, F.stats(feats))
        out[f"diversity_{name}"] = diversity(torch.cat([gen(min(500, n_div)).cpu() for _ in range((n_div + 499) // 500)])[:n_div])
    return out
