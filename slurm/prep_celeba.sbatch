#!/bin/bash
#SBATCH --job-name=ml_prep_celeba
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
cd /home/cbathula/AutoEncoders-VariationalAutoEncoders
export HF_HOME=/home/cbathula/hf_cache HF_HUB_DISABLE_PROGRESS_BARS=1
~/.conda/envs/medlatent/bin/python -u data/preprocess_celeba.py --workers 8
