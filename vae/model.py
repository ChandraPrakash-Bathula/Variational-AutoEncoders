"""Variational Autoencoder (self-contained model file).

    x --encoder--> (mu, logvar) --reparameterise--> z ~ N(mu, sigma^2) --decoder--> x_hat

Architecture (encoder/decoder backbones) intentionally matches the Autoencoder so the
comparison isolates the effect of the probabilistic latent: only the final encoder
layer differs (two heads: mu and logvar instead of one deterministic code).

Objective (beta-VAE, normalised by data dimensionality D = number of pixels):

    L = MSE_mean_per_pixel(x, x_hat) + beta * KL(q(z|x) || N(0, I)) / D

With beta = 1 this is the exact negative ELBO of a Gaussian decoder with variance 0.5
(up to a constant), divided by D. Smaller beta trades sample quality / prior-matching
for sharper reconstructions. beta is therefore an experimental variable to sweep.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    image_size: int = 128
    in_channels: int = 1
    latent_dim: int = 128
    base_channels: int = 32
    channel_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])

    @property
    def n_stages(self) -> int:
        return len(self.channel_mults)

    @property
    def bottleneck_size(self) -> int:
        size = self.image_size // (2 ** self.n_stages)
        if size < 1 or self.image_size % (2 ** self.n_stages) != 0:
            raise ValueError("image_size must be divisible by 2**len(channel_mults)")
        return size


def conv_block(in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, out_c),
        nn.SiLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        chans = [cfg.base_channels * m for m in cfg.channel_mults]
        layers: List[nn.Module] = [conv_block(cfg.in_channels, chans[0])]
        prev = chans[0]
        for c in chans:
            layers += [conv_block(prev, c, stride=2), conv_block(c, c)]
            prev = c
        self.features = nn.Sequential(*layers)
        self.flat_dim = prev * cfg.bottleneck_size ** 2
        self.fc_mu = nn.Linear(self.flat_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        chans = [cfg.base_channels * m for m in cfg.channel_mults][::-1]
        self.bottleneck = (chans[0], cfg.bottleneck_size, cfg.bottleneck_size)
        self.from_latent = nn.Linear(cfg.latent_dim, chans[0] * cfg.bottleneck_size ** 2)
        layers: List[nn.Module] = []
        prev = chans[0]
        for c in chans:
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), conv_block(prev, c), conv_block(c, c)]
            prev = c
        layers.append(nn.Conv2d(prev, cfg.in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(z.shape[0], *self.bottleneck)
        return torch.sigmoid(self.net(h))


class VAE(nn.Module):
    LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    # -- API ----------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        return mu, logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction through the posterior mean."""
        mu, _ = self.encode(x)
        return self.decode(mu)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu", temperature: float = 1.0) -> torch.Tensor:
        """Generate images by decoding z ~ N(0, temperature^2 I)."""
        z = torch.randn(n, self.cfg.latent_dim, device=device) * temperature
        return self.decode(z)

    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 8) -> torch.Tensor:
        """Linear interpolation between posterior means -> (steps, B, C, H, W)."""
        z1, _ = self.encode(x1)
        z2, _ = self.encode(x2)
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        return torch.stack([self.decode((1 - a) * z1 + a * z2) for a in alphas])

    # -- loss ---------------------------------------------------------------
    @staticmethod
    def loss_function(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
                      logvar: torch.Tensor, beta: float):
        """Returns (total, recon_mse, kl_per_image_nats). All scalars (batch means)."""
        recon, x, mu, logvar = recon.float(), x.float(), mu.float(), logvar.float()
        n_pixels = x[0].numel()
        recon_loss = F.mse_loss(recon, x, reduction="none").flatten(1).mean(1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
        total = recon_loss + beta * kl / n_pixels
        return total.mean(), recon_loss.mean(), kl.mean()

    # -- persistence --------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(VAEConfig(**ckpt["model_config"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt


def model_config_dict(cfg: VAEConfig) -> dict:
    return asdict(cfg)
