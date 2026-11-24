"""Dataset utilities for the preprocessed 2D MRI slices (see data/preprocess_ixi.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SliceDataset(Dataset):
    """In-memory dataset of uint8 images stored in <data_dir>/<split>.npz.

    `images` is (N, H, W) for grayscale (IXI slices) or (N, H, W, C) for colour (CelebA).
    Each item is a float tensor of shape (C, H, W) scaled to [0, 1].
    Metadata aligned with dataset order:
      `subject_ids` (subject / identity id, used for leakage checks and hover labels) and
      `labels`: dict of every 1-D numeric per-image array in the file (e.g. `slice_pos` for IXI,
      `attr_<Name>` for CelebA) plus the computed `foreground` fraction (`brain_area` alias kept
      for grayscale data). These are the colourings available to analysis/visualize_latents.py.
    """

    def __init__(self, data_dir: str | Path, split: str):
        path = Path(data_dir) / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run the preprocessing script first.")
        data = np.load(path, allow_pickle=False)
        self.images = data["images"]                       # (N, H, W) or (N, H, W, C) uint8
        n = len(self.images)
        self.subject_ids = data["subject_ids"]             # (N,) str
        self.labels = {k: data[k].astype(np.float32) for k in data.files
                       if k not in ("images", "subject_ids") and data[k].ndim == 1 and len(data[k]) == n
                       and np.issubdtype(data[k].dtype, np.number)}
        gray = self.images if self.images.ndim == 3 else self.images.mean(-1)
        fg = (gray > 25).reshape(n, -1).mean(1).astype(np.float32)
        self.labels["foreground"] = fg
        if self.images.ndim == 3:
            self.labels["brain_area"] = fg
        self.slice_pos = self.labels.get("slice_pos")       # backward compatibility (IXI)
        self.brain_area = fg
        self.split = split

    @property
    def image_size(self) -> int:
        return int(self.images.shape[2] if self.images.ndim == 4 else self.images.shape[-1])

    @property
    def channels(self) -> int:
        return 1 if self.images.ndim == 3 else int(self.images.shape[-1])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(self.images[idx])).float().div_(255.0)
        return x.unsqueeze(0) if x.ndim == 2 else x.permute(2, 0, 1).contiguous()


def make_loaders(data_dir: str | Path, batch_size: int, num_workers: int = 4,
                 pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SliceDataset(data_dir, "train")
    val_ds = SliceDataset(data_dir, "val")
    test_ds = SliceDataset(data_dir, "test")
    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader
