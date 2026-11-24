#!/bin/bash
#SBATCH --job-name=ml_analysis
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# sbatch slurm/analysis.sbatch 128 0.1
set -euo pipefail
cd /home/cbathula/AutoEncoders-VariationalAutoEncoders
P=~/.conda/envs/medlatent/bin/python
Z=${1:-128}; BETA=${2:-0.1}
AE=outputs/ae/z${Z}; VAE=outputs/vae/z${Z}_beta${BETA}
$P -u -m analysis.visualize_latents --run-dirs "$AE" "$VAE"
$P -u -m analysis.compare_models --ae-run "$AE" --vae-run "$VAE"
echo "$(date -Is) done"
