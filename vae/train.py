"""Train the Variational Autoencoder.

    python -m vae.train                                 # defaults from vae/config.yaml
    python -m vae.train --latent-dim 16 --beta 0.5
    python -m vae.train --resume outputs/vae/z128_beta0.1/checkpoints/last.pt

Outputs -> <output_dir>/vae/z<latent_dim>_beta<beta>/{checkpoints,figures,history.csv,...}
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from common.augment import augment_batch
from common.dataset import make_loaders
from common.metrics import reconstruction_metrics
from common.utils import (AverageMeter, count_parameters, get_device, load_config,
                          plot_history, save_comparison_grid, save_image_grid, save_json,
                          save_yaml, seed_everything, write_history_csv)
from vae.model import VAE, VAEConfig, model_config_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the convolutional VAE")
    p.add_argument("--config", default="vae/config.yaml")
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--latent-dim", dest="latent_dim", type=int)
    p.add_argument("--beta", type=float, help="KL weight")
    p.add_argument("--kl-warmup-epochs", dest="kl_warmup_epochs", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--weight-decay", dest="weight_decay", type=float)
    p.add_argument("--patience", type=int, help="early-stopping patience in epochs (0 = off)")
    p.add_argument("--augment", type=int, choices=[0, 1], help="train-time augmentation on (1) / off (0)")
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", default=None, help="path to last.pt to continue training")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device, use_amp, beta):
    """Validation always uses the FULL beta (no warm-up) and posterior means for metrics."""
    model.eval()
    keys = ("loss", "recon", "kl", "mse", "psnr", "ssim")
    meters = {k: AverageMeter() for k in keys}
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            recon, mu, logvar, _ = model(x)
            det_recon = model.decode(mu)
        total, rec, kl = VAE.loss_function(recon, x, mu, logvar, beta)
        m = reconstruction_metrics(x, det_recon.float())
        n = x.shape[0]
        meters["loss"].update(total.item(), n)
        meters["recon"].update(rec.item(), n)
        meters["kl"].update(kl.item(), n)
        for k in ("mse", "psnr", "ssim"):
            meters[k].update(m[k].mean().item(), n)
    return {k: v.avg for k, v in meters.items()}


def main() -> None:
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "device", "resume")}
    cfg = load_config(args.config, overrides)
    seed_everything(cfg["seed"], cfg.get("deterministic", False))
    device = get_device(args.device)
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    beta = float(cfg["beta"])

    run_dir = Path(cfg["output_dir"]) / "vae" / f"z{cfg['latent_dim']}_beta{beta:g}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    train_loader, val_loader, _ = make_loaders(cfg["data_dir"], cfg["batch_size"], cfg["num_workers"],
                                               pin_memory=device.type == "cuda")
    cfg["image_size"] = train_loader.dataset.image_size
    cfg["in_channels"] = train_loader.dataset.channels
    save_yaml(cfg, run_dir / "config_resolved.yaml")

    model_cfg = VAEConfig(image_size=cfg["image_size"], in_channels=cfg["in_channels"],
                          latent_dim=cfg["latent_dim"], base_channels=cfg["base_channels"],
                          channel_mults=list(cfg["channel_mults"]))
    model = VAE(model_cfg).to(device)
    print(f"VAE | latent_dim={cfg['latent_dim']} | beta={beta:g} | params={count_parameters(model)/1e6:.2f}M "
          f"| device={device} | train/val slices={len(train_loader.dataset)}/{len(val_loader.dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"],
                                                           eta_min=cfg["lr"] * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_val, start_epoch, bad_epochs = [], float("inf"), 1, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history, best_val = ckpt["history"], ckpt["best_val"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    vis_idx = torch.linspace(0, len(val_loader.dataset) - 1, cfg["num_vis"]).long()
    vis_batch = torch.stack([val_loader.dataset[int(i)] for i in vis_idx]).to(device)
    fixed_z = torch.randn(32, cfg["latent_dim"], device=device, generator=torch.Generator(device).manual_seed(0))

    def save_ckpt(path: Path, epoch: int, val_metrics: dict) -> None:
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "model_config": model_config_dict(model_cfg),
                    "train_config": cfg, "epoch": epoch, "best_val": best_val, "beta": beta,
                    "val_metrics": val_metrics, "history": history, "model_type": "vae"}, path)

    warmup = max(int(cfg["kl_warmup_epochs"]), 0)
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()
        beta_t = beta * min(1.0, epoch / warmup) if warmup else beta  # linear ramp, full beta at epoch == warmup
        model.train()
        m_loss, m_rec, m_kl = AverageMeter(), AverageMeter(), AverageMeter()
        for x in tqdm(train_loader, desc=f"epoch {epoch}/{cfg['epochs']}", leave=False):
            x = augment_batch(x.to(device, non_blocking=True), cfg)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                recon, mu, logvar, _ = model(x)
            loss, rec, kl = VAE.loss_function(recon, x, mu, logvar, beta_t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            n = x.shape[0]
            m_loss.update(loss.item(), n); m_rec.update(rec.item(), n); m_kl.update(kl.item(), n)
        scheduler.step()

        val = validate(model, val_loader, device, use_amp, beta)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "beta_t": beta_t,
               "train_loss": m_loss.avg, "train_recon": m_rec.avg, "train_kl": m_kl.avg,
               "val_loss": val["loss"], "val_recon": val["recon"], "val_kl": val["kl"],
               "val_psnr": val["psnr"], "val_ssim": val["ssim"], "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[{epoch:3d}] beta_t {beta_t:.3f} | train {row['train_loss']:.5f} | val {row['val_loss']:.5f} "
              f"| recon {row['val_recon']:.5f} | KL {row['val_kl']:.1f} | PSNR {row['val_psnr']:.2f} dB "
              f"| SSIM {row['val_ssim']:.4f} | {row['seconds']}s")

        # Only consider "best" once the KL term is fully on, so checkpoints are comparable.
        if epoch >= warmup:
            if val["loss"] < best_val:
                best_val, bad_epochs = val["loss"], 0
                save_ckpt(run_dir / "checkpoints" / "best.pt", epoch, val)
            else:
                bad_epochs += 1
        save_ckpt(run_dir / "checkpoints" / "last.pt", epoch, val)
        if epoch == cfg["epochs"] and not (run_dir / "checkpoints" / "best.pt").exists():
            save_ckpt(run_dir / "checkpoints" / "best.pt", epoch, val)  # very short runs

        write_history_csv(history, run_dir / "history.csv")
        if epoch % cfg["vis_every"] == 0 or epoch == cfg["epochs"] or epoch == start_epoch:
            model.eval()
            recon = model.reconstruct(vis_batch)
            save_comparison_grid([vis_batch, recon, (vis_batch - recon).abs().mean(1, keepdim=True) * 4],
                                 ["input", "VAE recon", "|error| x4"],
                                 run_dir / "figures" / f"recon_epoch{epoch:03d}.png")
            with torch.no_grad():
                save_image_grid(model.decode(fixed_z), run_dir / "figures" / f"samples_epoch{epoch:03d}.png",
                                n_cols=8, title=f"Samples z~N(0,I), epoch {epoch}")
            plot_history(history, ["loss", "recon", "kl", "psnr", "ssim"],
                         run_dir / "figures" / "training_curves.png",
                         title=f"VAE (z={cfg['latent_dim']}, beta={beta:g})")
        if cfg["patience"] and epoch >= warmup and bad_epochs >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    save_json({"best_val_loss": best_val, "epochs_run": len(history), "beta": beta}, run_dir / "train_summary.json")
    print(f"Done. Best checkpoint: {run_dir / 'checkpoints' / 'best.pt'}")


if __name__ == "__main__":
    main()
