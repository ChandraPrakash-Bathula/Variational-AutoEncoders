#!/bin/bash
#SBATCH --job-name=ml_sweep
#SBATCH --partition=gpu,machinelearning,ccstar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x_%A_%a.out
# sbatch --array=1-8%4 slurm/sweep_train.sbatch
# Task i -> (model, latent) over LATENTS x {ae, vae}. Delegates to train.sbatch (resume-safe).
set -euo pipefail
cd /home/cbathula/AutoEncoders-VariationalAutoEncoders
LATENTS=(${LATENTS:-2 3 16 64})
BETA=${BETA:-0.1}
i=$((SLURM_ARRAY_TASK_ID - 1))
Z=${LATENTS[$((i / 2))]}
MODEL=$([[ $((i % 2)) == 0 ]] && echo ae || echo vae)
exec bash slurm/train.sbatch "$MODEL" "$Z" "$BETA"
