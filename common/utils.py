"""Shared utilities: seeding, config handling, plotting helpers."""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# Reproducibility / environment
# --------------------------------------------------------------------------- #
def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Config / IO
# --------------------------------------------------------------------------- #
def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    return cfg


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_history_csv(history: List[Dict[str, Any]], path: Path) -> None:
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


class AverageMeter:
    """Running average weighted by sample count."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


# --------------------------------------------------------------------------- #
# Plot helpers (matplotlib is imported lazily with the Agg backend so these
# work on headless cluster nodes)
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_history(history: List[Dict[str, Any]], keys: Sequence[str], path: Path,
                 title: str = "") -> None:
    """One subplot per key in `keys`, drawing train_<key> and val_<key> if present."""
    plt = _plt()
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for prefix, style in (("train_", "-"), ("val_", "--")):
            name = prefix + key
            if name in history[0]:
                ax.plot(epochs, [h[name] for h in history], style, label=prefix[:-1])
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _show(ax, img: torch.Tensor, cmap: str = "gray", vmin: float = 0.0, vmax: float = 1.0) -> None:
    """imshow a single C x H x W tensor: grayscale with `cmap` (C == 1) or RGB (C == 3)."""
    a = img.detach().float().cpu()
    if a.shape[0] == 1:
        ax.imshow(a[0].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.imshow(((a - vmin) / (vmax - vmin)).clamp(0, 1).permute(1, 2, 0).numpy())


def error_map(x: torch.Tensor, recon: torch.Tensor, gain: float = 4.0) -> torch.Tensor:
    """|x - recon| * gain, averaged over colour channels -> N x 1 x H x W (shown in grayscale)."""
    return (x - recon).abs().mean(1, keepdim=True) * gain


def save_comparison_grid(rows: Sequence[torch.Tensor], row_titles: Sequence[str], path: Path,
                         col_titles: Optional[Sequence[str]] = None, cmap: str = "gray",
                         vmin: float = 0.0, vmax: float = 1.0, dpi: int = 150) -> None:
    """Rows of images (each tensor: N x C x H x W, C in {1, 3}) drawn as a labelled grid."""
    plt = _plt()
    n_rows, n_cols = len(rows), rows[0].shape[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols + 1.2, 1.5 * n_rows + (0.3 if col_titles else 0)),
                             squeeze=False)
    for r, (imgs, title) in enumerate(zip(rows, row_titles)):
        for c in range(n_cols):
            ax = axes[r, c]
            _show(ax, imgs[c], cmap, vmin, vmax)
            ax.axis("off")
            if col_titles and r == 0:
                ax.set_title(col_titles[c], fontsize=8)
        axes[r, 0].text(-0.08, 0.5, title, transform=axes[r, 0].transAxes, ha="right",
                        va="center", fontsize=9)
    fig.subplots_adjust(left=0.14, right=0.995, top=0.93 if col_titles else 0.995, bottom=0.005,
                        wspace=0.02, hspace=0.02)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_image_grid(images: torch.Tensor, path: Path, n_cols: int = 8, title: str = "",
                    dpi: int = 150) -> None:
    """Tile N x C x H x W images (C in {1, 3}) into a single grid figure."""
    plt = _plt()
    n = images.shape[0]
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            _show(ax, images[i])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95 if title else 0.995, bottom=0.005,
                        wspace=0.02, hspace=0.02)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
