"""Preprocess IXI T1-weighted brain MRI volumes into 2D axial slices.

Pipeline per subject
  1. load NIfTI, reorient to RAS (so axis 2 is always superior/inferior)
  2. intensity normalisation: divide by the 99.5th percentile of non-zero voxels, clip to [0, 1]
  3. take axial slices inside a configurable relative z-range, every `--slice-stride`-th slice
  4. rotate to a radiological display orientation, zero-pad to a square, resize to `--image-size`
  5. drop slices with too little foreground (neck / vertex slices)

Subjects (not slices!) are split into train / val / test with a fixed seed, so no subject
ever appears in more than one split. Output: <out-dir>/{train,val,test}.npz + split.json

IXI is released under CC BY-SA 3.0: https://brain-development.org/ixi-dataset/

Usage
  python data/preprocess_ixi.py --raw-dir data/raw/IXI-T1 --download
  python data/preprocess_ixi.py --raw-dir data/raw/IXI-T1          # already downloaded
"""
from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

IXI_T1_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"


def download_ixi(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_path = raw_dir / "IXI-T1.tar"
    if not tar_path.exists():
        print(f"Downloading {IXI_T1_URL} (~4.5 GB) ...")
        urllib.request.urlretrieve(IXI_T1_URL, tar_path)
    print("Extracting ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(raw_dir)


def normalise_volume(vol: np.ndarray) -> np.ndarray:
    nonzero = vol[vol > 0]
    if nonzero.size == 0:
        raise ValueError("empty volume")
    scale = np.percentile(nonzero, 99.5)
    return np.clip(vol / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


def slice_to_square(sl: np.ndarray, size: int) -> np.ndarray:
    """Rotate for display, zero-pad to a square and resize (anti-aliased) to size x size."""
    sl = np.rot90(sl)
    h, w = sl.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    padded[top:top + h, left:left + w] = sl
    resized = np.asarray(Image.fromarray(padded, mode="F").resize((size, size), Image.BILINEAR))
    return np.clip(np.round(resized * 255.0), 0, 255).astype(np.uint8)


def process_subject(path: Path, size: int, z_range, stride: int, min_fg: float):
    img = nib.as_closest_canonical(nib.load(str(path)))
    vol = normalise_volume(np.squeeze(np.asarray(img.dataobj, dtype=np.float32)))
    if vol.ndim != 3:
        raise ValueError(f"expected 3D volume, got {vol.shape}")
    depth = vol.shape[2]
    lo, hi = int(z_range[0] * depth), int(z_range[1] * depth)
    slices, positions = [], []
    for k in range(lo, hi, stride):
        sl = slice_to_square(vol[:, :, k], size)
        if (sl > 25).mean() >= min_fg:
            slices.append(sl)
            positions.append(k / depth)
    return slices, positions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/IXI-T1"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--download", action="store_true", help="download IXI-T1.tar into --raw-dir")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--z-range", type=float, nargs=2, default=(0.30, 0.80),
                   metavar=("LO", "HI"), help="relative axial range to keep")
    p.add_argument("--slice-stride", type=int, default=2)
    p.add_argument("--min-foreground", type=float, default=0.12)
    p.add_argument("--split", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-subjects", type=int, default=None, help="debug: limit subject count")
    args = p.parse_args()

    if args.download:
        download_ixi(args.raw_dir)
    files = sorted(args.raw_dir.rglob("*-T1.nii.gz")) + sorted(args.raw_dir.rglob("*-T1.nii"))
    if args.max_subjects:
        files = files[: args.max_subjects]
    if not files:
        raise SystemExit(f"No *-T1.nii(.gz) files found under {args.raw_dir}")
    print(f"Found {len(files)} T1 volumes")

    per_subject = {}
    for f in tqdm(files, desc="subjects"):
        sid = f.name.split("-T1")[0]
        try:
            slices, pos = process_subject(f, args.image_size, args.z_range,
                                          args.slice_stride, args.min_foreground)
        except Exception as exc:  # corrupt volume etc. -> skip but report
            print(f"[skip] {f.name}: {exc}")
            continue
        if slices:
            per_subject[sid] = (np.stack(slices), np.asarray(pos, dtype=np.float32))

    # --- subject-level split ------------------------------------------------
    ids = sorted(per_subject)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(args.split[0] * n))
    n_val = int(round(args.split[1] * n))
    splits = {"train": ids[:n_train], "val": ids[n_train:n_train + n_val], "test": ids[n_train + n_val:]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, subj in splits.items():
        images = np.concatenate([per_subject[s][0] for s in subj]) if subj else np.zeros((0, args.image_size, args.image_size), np.uint8)
        subject_ids = np.concatenate([[s] * len(per_subject[s][0]) for s in subj]) if subj else np.array([], dtype="<U1")
        slice_pos = np.concatenate([per_subject[s][1] for s in subj]) if subj else np.array([], dtype=np.float32)
        np.savez_compressed(args.out_dir / f"{name}.npz", images=images,
                            subject_ids=subject_ids.astype("<U40"), slice_pos=slice_pos)
        print(f"{name:5s}: {len(subj):4d} subjects, {len(images):6d} slices")

    with open(args.out_dir / "split.json", "w") as f:
        json.dump({"seed": args.seed, "args": {k: str(v) for k, v in vars(args).items()},
                   "splits": splits}, f, indent=2)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
