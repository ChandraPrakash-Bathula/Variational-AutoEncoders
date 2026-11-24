#!/bin/bash
#SBATCH --job-name=ml_sweep_an
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# Per-latent latent-space comparisons + results table over all runs (mirrors scripts/run_all.sh).
set -euo pipefail
cd /home/cbathula/AutoEncoders-VariationalAutoEncoders
P=~/.conda/envs/medlatent/bin/python
LATENTS=${LATENTS:-"2 3 16 64 128"}; BETA=${BETA:-0.1}; MAIN_Z=${MAIN_Z:-128}
for z in $LATENTS; do
  $P -u -m analysis.visualize_latents --run-dirs "outputs/ae/z${z}" "outputs/vae/z${z}_beta${BETA}" \
    --comparison-dir "outputs/analysis/latent_comparison_z${z}"
done
$P -u -m analysis.compare_models --ae-run "outputs/ae/z${MAIN_Z}" --vae-run "outputs/vae/z${MAIN_Z}_beta${BETA}"
echo "$(date -Is) done"
