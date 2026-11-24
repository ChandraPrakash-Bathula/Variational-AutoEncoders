#!/usr/bin/env bash
# Train + evaluate AE and VAE for several latent sizes, then build all figures.
# Usage (from the repo root):   bash scripts/run_all.sh
#   LATENTS="16 128" BETA=0.5 EPOCHS=60 bash scripts/run_all.sh
set -euo pipefail

LATENTS=${LATENTS:-"2 3 16 64 128"}
BETA=${BETA:-0.1}
EPOCHS=${EPOCHS:-100}
MAIN_Z=${MAIN_Z:-128}          # run used for the headline AE-vs-VAE comparison

for z in $LATENTS; do
  python -m autoencoder.train --latent-dim "$z" --epochs "$EPOCHS"
  python -m autoencoder.evaluate --run-dir "outputs/ae/z${z}"
  python -m vae.train --latent-dim "$z" --beta "$BETA" --epochs "$EPOCHS"
  python -m vae.evaluate --run-dir "outputs/vae/z${z}_beta${BETA}"
done

for z in $LATENTS; do
  python -m analysis.visualize_latents \
    --run-dirs "outputs/ae/z${z}" "outputs/vae/z${z}_beta${BETA}" \
    --comparison-dir "outputs/analysis/latent_comparison_z${z}"
done

python -m analysis.compare_models --ae-run "outputs/ae/z${MAIN_Z}" --vae-run "outputs/vae/z${MAIN_Z}_beta${BETA}"
echo "All done. Results in outputs/"
