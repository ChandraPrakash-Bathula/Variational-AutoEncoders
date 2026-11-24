"""2D / 3D latent-space visualisation for trained AE and VAE runs.

For every run the exported latent codes (latents/<split>.npz, produced by the evaluate scripts) are
projected with PCA, t-SNE and UMAP into 2 and 3 dimensions and saved as
  * static PNGs (matplotlib)               -> <run>/figures/latent_space/
  * interactive HTML (plotly, rotatable 3D) -> <run>/figures/latent_space/
  * raw embeddings (.npz) for later reuse (e.g. the Hugging Face Space)
Points are coloured by a meaningful anatomical / quality label:
  slice_pos  : relative axial height of the slice (does z encode anatomy?)
  brain_area : foreground fraction of the slice
  mse        : per-slice reconstruction error

If the latent space is already 2-D or 3-D (latent_dim 2 or 3) it is plotted directly ("raw").
When several runs are given, side-by-side 2D comparison figures are also written.

    python -m analysis.visualize_latents --run-dirs outputs/ae/z128 outputs/vae/z128_beta0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABEL_NAMES = {"slice_pos": "relative axial slice position", "brain_area": "brain foreground fraction",
               "foreground": "foreground fraction", "mse": "reconstruction MSE", "ssim": "reconstruction SSIM"}
DEFAULT_COLORS = ["slice_pos", "attr_Male", "attr_Smiling", "attr_Eyeglasses", "attr_Blond_Hair"]


def label_name(key: str) -> str:
    if key in LABEL_NAMES:
        return LABEL_NAMES[key]
    return key[5:].replace("_", " ") + " (0/1)" if key.startswith("attr_") else key


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def embed(z: np.ndarray, method: str, k: int, seed: int, perplexity: float,
          n_neighbors: int, min_dist: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (embedding N x k, explained_variance_ratio or None)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if method == "raw":
        return z[:, :k], None
    if method == "pca":
        pca = PCA(n_components=k, random_state=seed)
        return pca.fit_transform(z - z.mean(0)), pca.explained_variance_ratio_

    x = StandardScaler().fit_transform(z)          # neighbour methods need comparable scales
    if x.shape[1] > 50:                            # standard speed-up, preserves local structure
        x = PCA(n_components=50, random_state=seed).fit_transform(x)
    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = float(min(perplexity, max(5.0, (len(x) - 1) / 3.0)))
        return TSNE(n_components=k, perplexity=perp, init="pca", learning_rate="auto",
                    random_state=seed).fit_transform(x), None
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise SystemExit("umap-learn is required for --methods umap (pip install umap-learn)") from exc
        return umap.UMAP(n_components=k, n_neighbors=n_neighbors, min_dist=min_dist,
                         random_state=seed).fit_transform(x), None
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def axis_names(method: str) -> Tuple[str, str, str]:
    prefix = {"pca": "PC", "tsne": "t-SNE ", "umap": "UMAP ", "raw": "z"}[method]
    start = 0 if method == "raw" else 1
    return tuple(f"{prefix}{i + start}" for i in range(3))


def scatter_ax(ax, emb, labels, method, title, k, cmap="viridis", s=6):
    names = axis_names(method)
    if k == 2:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, s=s, cmap=cmap, alpha=0.8, linewidths=0)
    else:
        sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=labels, s=s, cmap=cmap, alpha=0.8, linewidths=0)
        ax.set_zlabel(names[2]); ax.view_init(elev=20, azim=-60)
    ax.set_xlabel(names[0]); ax.set_ylabel(names[1]); ax.set_title(title, fontsize=10)
    return sc


def save_static(emb, labels, label_key, method, k, title, path: Path) -> None:
    fig = plt.figure(figsize=(6.4, 5.4))
    ax = fig.add_subplot(111, projection="3d" if k == 3 else None)
    sc = scatter_ax(ax, emb, labels, method, title, k)
    fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.08 if k == 3 else 0.02, label=label_name(label_key))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_interactive(emb, labels, subjects, label_key, method, k, title, path: Path) -> None:
    import pandas as pd
    import plotly.express as px

    names = axis_names(method)
    df = pd.DataFrame({names[0]: emb[:, 0], names[1]: emb[:, 1], label_name(label_key): labels,
                       "subject": subjects})
    kwargs = dict(color=label_name(label_key), color_continuous_scale="Viridis",
                  hover_data=["subject"], title=title, opacity=0.8)
    if k == 3:
        df[names[2]] = emb[:, 2]
        fig = px.scatter_3d(df, x=names[0], y=names[1], z=names[2], **kwargs)
        fig.update_traces(marker=dict(size=2.5))
    else:
        fig = px.scatter(df, x=names[0], y=names[1], **kwargs)
        fig.update_traces(marker=dict(size=5))
    fig.write_html(str(path), include_plotlyjs="cdn")


def plot_variance(z: np.ndarray, title: str, path: Path, max_comp: int = 50) -> None:
    from sklearn.decomposition import PCA

    n = min(max_comp, z.shape[1], len(z))
    ratio = PCA(n_components=n).fit(z - z.mean(0)).explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(np.arange(1, n + 1), ratio, alpha=0.6, label="per component")
    ax.plot(np.arange(1, n + 1), np.cumsum(ratio), "o-", ms=3, color="C3", label="cumulative")
    ax.set_xlabel("principal component"); ax.set_ylabel("explained variance ratio")
    ax.set_title(title, fontsize=10); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_kl_per_dim(mu: np.ndarray, logvar: np.ndarray, title: str, path: Path) -> None:
    """Average KL contribution per latent dimension: flat zeros = collapsed (inactive) dimensions."""
    kl = 0.5 * (mu ** 2 + np.exp(logvar) - 1 - logvar)
    per_dim = kl.mean(0)
    order = np.argsort(-per_dim)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(np.arange(len(per_dim)), per_dim[order])
    ax.set_xlabel("latent dimension (sorted)"); ax.set_ylabel("mean KL (nats)")
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# --------------------------------------------------------------------------- #
def run_label(run_dir: Path) -> str:
    return f"{run_dir.parent.name.upper()} {run_dir.name}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--methods", nargs="+", default=["pca", "tsne", "umap"], choices=["pca", "tsne", "umap"])
    p.add_argument("--dims", nargs="+", type=int, default=[2, 3], choices=[2, 3])
    p.add_argument("--color-by", nargs="+", default=None,
                   help="label keys stored in latents/<split>.npz (default: slice_pos for IXI, "
                        "a few attributes for CelebA)")
    p.add_argument("--max-points", type=int, default=4000)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-html", action="store_true", help="skip interactive plotly files")
    p.add_argument("--comparison-dir", type=Path, default=Path("outputs/analysis/latent_comparison"))
    args = p.parse_args()

    collected: Dict[Tuple[str, int], Dict[str, tuple]] = {}
    for run_dir in args.run_dirs:
        data = np.load(run_dir / "latents" / f"{args.split}.npz", allow_pickle=False)
        colors = args.color_by or [c for c in DEFAULT_COLORS if c in data.files] or ["mse"]
        z, subjects = data["latents"], data["subject_ids"]
        all_labels = {c: data[c] for c in colors}
        if len(z) > args.max_points:
            idx = np.sort(np.random.RandomState(args.seed).choice(len(z), args.max_points, replace=False))
            z, subjects = z[idx], subjects[idx]
            all_labels = {c: v[idx] for c, v in all_labels.items()}
        D = z.shape[1]
        label = run_label(run_dir)
        out_dir = run_dir / "figures" / "latent_space"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {label}: {len(z)} points, latent dim {D}, colours {colors}")

        if D > 3:
            plot_variance(z, f"PCA variance - {label}", out_dir / "pca_explained_variance.png")
        if "logvar" in data.files:
            plot_kl_per_dim(data["latents"], data["logvar"], f"KL per latent dim - {label}",
                            out_dir / "kl_per_dimension.png")

        done_raw = set()
        for k in args.dims:
            if k > D:
                continue
            for method in args.methods:
                m = "raw" if D == k else method
                if m == "raw":
                    if k in done_raw:
                        continue
                    done_raw.add(k)
                emb, ev = embed(z, m, k, args.seed, args.perplexity, args.n_neighbors, args.min_dist)
                title = f"{label} | {m.upper() if m != 'raw' else 'latent (direct)'} {k}D"
                if m == "pca" and ev is not None:
                    title += f" ({ev.sum() * 100:.0f}% var)"
                np.savez_compressed(out_dir / f"embedding_{m}_{k}d.npz", embedding=emb, subject_ids=subjects,
                                    **{f"label_{c}": v for c, v in all_labels.items()})
                for c, labels in all_labels.items():
                    stem = f"{m}_{k}d_{c}"
                    save_static(emb, labels, c, m, k, title, out_dir / f"{stem}.png")
                    if not args.no_html:
                        save_interactive(emb, labels, subjects, c, m, k, title, out_dir / f"{stem}.html")
                    collected.setdefault((m, k, c), {})[label] = (emb, labels, title)
                print(f"   saved {m}_{k}d x {len(all_labels)} colourings")

    # ---- side-by-side comparison across runs (2D) --------------------------
    if len(args.run_dirs) > 1:
        args.comparison_dir.mkdir(parents=True, exist_ok=True)
        for (m, k, c), runs in collected.items():
            if k != 2 or len(runs) < 2:
                continue
            fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6), squeeze=False)
            for ax, (label, (emb, labels, title)) in zip(axes[0], runs.items()):
                sc = scatter_ax(ax, emb, labels, m, title, 2)
            fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.85, label=label_name(c))
            fig.savefig(args.comparison_dir / f"compare_{m}_2d_{c}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        print(f"Comparison figures -> {args.comparison_dir}")


if __name__ == "__main__":
    main()
