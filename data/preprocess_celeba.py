"""Download + preprocess CelebA (aligned faces) into 64x64 RGB arrays.

Source: Hugging Face mirror `flwrlabs/celeba` (img_align + identity + 40 attributes), pinned revision.
Uses the official CelebA partition (train 162,770 / valid 19,867 / test 19,962; identities do not
overlap between partitions).

Per image: standard "CelebA-64" preprocessing - centre crop 148x148 of the 178x218 aligned image,
resize to 64x64 (antialiased). Output: <out-dir>/{train,val,test}.npz with
  images (N, 64, 64, 3) uint8, subject_ids (celeb id, str), attr_<Name> (N,) float32 0/1 for all 40
  attributes. Saved uncompressed for fast loading.

    python data/preprocess_celeba.py
"""
from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

REPO, REVISION = "flwrlabs/celeba", "2d738f56e0e7f925ea36ae7c808ea925264aacec"
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def _process(args):
    data, size, crop = args
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    left, top = (w - crop) // 2, (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop)).resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/celeba/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("data/celeba/processed"))
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--crop", type=int, default=148)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-per-split", type=int, default=None, help="debug: limit images per split")
    args = p.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=str(args.raw_dir),
                      allow_patterns=["img_align+identity+attr/*.parquet"])
    files = sorted((args.raw_dir / "img_align+identity+attr").glob("*.parquet"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for name, prefix in SPLITS.items():
            shards = [f for f in files if f.name.startswith(prefix + "-")]
            table = pa.concat_tables([pq.read_table(f) for f in shards])
            if args.max_per_split:
                table = table.slice(0, args.max_per_split)
            attr_names = [c for c in table.column_names if c not in ("image", "celeb_id")]
            blobs = [d["bytes"] for d in table.column("image").to_pylist()]
            images = np.stack(list(pool.map(_process, [(b, args.image_size, args.crop) for b in blobs],
                                            chunksize=256)))
            payload = {"images": images,
                       "subject_ids": np.asarray([str(i) for i in table.column("celeb_id").to_pylist()], dtype="<U12")}
            for a in attr_names:
                payload[f"attr_{a}"] = np.asarray(table.column(a).to_pylist(), dtype=np.float32)
            np.savez(args.out_dir / f"{name}.npz", **payload)
            counts[name] = {"images": int(len(images)), "identities": int(len(set(payload["subject_ids"])))}
            print(f"{name:5s}: {len(images):6d} images, {counts[name]['identities']} identities, shape {images.shape[1:]}")

    tr, te = (set(np.load(args.out_dir / f"{s}.npz")["subject_ids"]) for s in ("train", "test"))
    print(f"identity overlap train/test: {len(tr & te)}")
    with open(args.out_dir / "split.json", "w") as f:
        json.dump({"source": REPO, "revision": REVISION, "crop": args.crop, "image_size": args.image_size,
                   "counts": counts, "attributes": attr_names}, f, indent=2)


if __name__ == "__main__":
    main()
