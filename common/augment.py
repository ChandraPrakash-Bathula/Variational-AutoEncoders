"""Batched, on-device training augmentation for 2D MRI slices in [0, 1].

Applied to TRAINING batches only (never val/test), identically for AE and VAE.
Per image: random left-right flip, small rotation / isotropic scale / translation
(bilinear resampling, zero background) and a random intensity gamma.
The reconstruction target is the augmented image itself.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def augment_batch(x: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    if not cfg.get("augment", False):
        return x
    b, dev = x.shape[0], x.device

    def u(*shape):  # uniform in [-1, 1]
        return torch.rand(*shape, device=dev) * 2 - 1

    rot, scl, trn = (cfg.get("aug_rotate_deg", 10.0), cfg.get("aug_scale", 0.05), cfg.get("aug_translate", 0.05))
    flip_mask = torch.rand(b, device=dev) < cfg.get("aug_flip_p", 0.5)
    if not (rot or scl or trn):                                 # flip only (e.g. aligned faces): exact, no resampling
        x = torch.where(flip_mask[:, None, None, None], x.flip(-1), x)
    else:
        ang = u(b) * math.radians(rot)
        scale = 1 + u(b) * scl
        shift = u(b, 2) * trn                                   # normalised coords ([-1, 1] spans the image)
        flip = torch.where(flip_mask, -1.0, 1.0)
        cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
        theta = torch.stack([torch.stack([cos * flip, -sin, shift[:, 0]], 1),
                             torch.stack([sin * flip, cos, shift[:, 1]], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    gmax = cfg.get("aug_gamma", 1.2)
    if gmax and gmax > 1:
        gamma = torch.exp(u(b, 1, 1, 1) * math.log(gmax))       # log-uniform in [1/gmax, gmax]
        x = x.clamp(0, 1).pow(gamma)
    return x.clamp(0, 1)
