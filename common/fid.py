"""Frechet Inception Distance (FID) for image realism, using the standard pytorch-fid InceptionV3.

Images are [0, 1] tensors (N x C x H x W); grayscale is repeated to 3 channels and everything is
resized to 299x299 inside the network (standard practice for small / non-RGB images - FID values
are only comparable within one dataset, image size and sample count, never across datasets).

Reference = real images of the evaluation split (cached per dataset). We also report a
real-vs-real floor (val vs test images, same count) that says what "indistinguishable" looks like.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch

_MODEL = {}


def _inception(device: torch.device):
    key = str(device)
    if key not in _MODEL:
        from pytorch_fid.inception import InceptionV3
        _MODEL[key] = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True,
                                  normalize_input=True).to(device).eval()
    return _MODEL[key]


@torch.no_grad()
def inception_features(batches: Iterable[torch.Tensor], device: torch.device) -> np.ndarray:
    model, feats = _inception(device), []
    for x in batches:
        x = x.to(device).float().clamp(0, 1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats.append(model(x)[0].flatten(1).double().cpu())
    return torch.cat(feats).numpy()


def stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return feats.mean(0), np.cov(feats, rowvar=False)


def frechet(s1: Tuple[np.ndarray, np.ndarray], s2: Tuple[np.ndarray, np.ndarray]) -> float:
    from pytorch_fid.fid_score import calculate_frechet_distance
    return float(calculate_frechet_distance(s1[0], s1[1], s2[0], s2[1]))


def image_batches(dataset, idx: np.ndarray, batch_size: int = 200):
    for i in range(0, len(idx), batch_size):
        yield torch.stack([dataset[int(j)] for j in idx[i:i + batch_size]])


def generated_batches(gen_fn: Callable[[int], torch.Tensor], n: int, batch_size: int = 200):
    """gen_fn(k) -> k images; yields n images in batches."""
    done = 0
    while done < n:
        k = min(batch_size, n - done)
        yield gen_fn(k)
        done += k


def fid_subset(n_total: int, n_max: int, seed: int = 0) -> np.ndarray:
    idx = np.arange(n_total)
    if n_total > n_max:
        idx = np.sort(np.random.RandomState(seed).choice(n_total, n_max, replace=False))
    return idx


def real_reference(data_dir: str | Path, split: str, n_max: int, device: torch.device,
                   cache_dir: Optional[Path] = None):
    """(stats of real `split` images, subset indices, real-vs-real floor FID). Cached on disk."""
    from common.dataset import SliceDataset

    cache_dir = Path(cache_dir or data_dir)
    path = cache_dir / f"fid_ref_{split}_n{n_max}.npz"
    ds = SliceDataset(data_dir, split)
    idx = fid_subset(len(ds), n_max)
    if path.exists():
        d = np.load(path)
        return (d["mu"], d["sigma"]), idx, float(d["floor"])
    ref = stats(inception_features(image_batches(ds, idx), device))
    other = "val" if split != "val" else "test"
    ods = SliceDataset(data_dir, other)
    oidx = fid_subset(len(ods), len(idx), seed=1)
    floor = frechet(ref, stats(inception_features(image_batches(ods, oidx), device)))
    try:  # atomic: parallel evaluation jobs may build the same cache
        import os
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, mu=ref[0], sigma=ref[1], floor=floor)
        os.replace(tmp, path)
    except OSError:
        pass
    return ref, idx, floor
