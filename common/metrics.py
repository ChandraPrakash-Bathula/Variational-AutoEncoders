"""Image-quality metrics implemented in plain PyTorch (inputs scaled to [0, 1])."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).flatten(1).mean(1)


def masked_mse_per_image(x: torch.Tensor, y: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """MSE restricted to foreground pixels of the target `x` (background is trivial to fit)."""
    mask = (x > threshold).float()
    err = ((x - y) ** 2 * mask).flatten(1).sum(1)
    return err / mask.flatten(1).sum(1).clamp_min(1.0)


def psnr_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = mse_per_image(x, y).clamp_min(1e-10)
    return 10.0 * torch.log10(data_range ** 2 / mse)


def masked_psnr_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """PSNR over foreground pixels only (the black background otherwise inflates PSNR)."""
    return 10.0 * torch.log10(data_range ** 2 / masked_mse_per_image(x, y).clamp_min(1e-10))


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    return window.expand(channels, 1, size, size).contiguous()


def ssim_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0,
                   window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM (Wang et al., 2004) per image using a Gaussian window."""
    channels = x.shape[1]
    w = _gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    pad = window_size // 2

    def blur(t):
        return F.conv2d(t, w, padding=pad, groups=channels)

    mu_x, mu_y = blur(x), blur(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = blur(x * x) - mu_x2
    sigma_y = blur(y * y) - mu_y2
    sigma_xy = blur(x * y) - mu_xy
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2))
    return ssim_map.flatten(1).mean(1)


@torch.no_grad()
def reconstruction_metrics(x: torch.Tensor, y: torch.Tensor) -> dict:
    """Per-image metric tensors for a batch."""
    return {
        "mse": mse_per_image(x, y),
        "masked_mse": masked_mse_per_image(x, y),
        "psnr": psnr_per_image(x, y),
        "masked_psnr": masked_psnr_per_image(x, y),
        "ssim": ssim_per_image(x, y),
    }
