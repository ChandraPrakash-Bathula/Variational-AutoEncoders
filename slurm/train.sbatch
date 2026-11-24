#!/bin/bash
#SBATCH --job-name=ml_train
#SBATCH --partition=gpu,machinelearning,ccstar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x_%j.out
# sbatch slurm/train.sbatch ae  128
# sbatch slurm/train.sbatch vae 128 0.1
# Train + evaluate one model. Resumes from last.pt on requeue/resubmit.
set -euo pipefail
cd /home/cbathula/AutoEncoders-VariationalAutoEncoders
P=~/.conda/envs/medlatent/bin/python
MODEL=$1; Z=$2; BETA=${3:-0.1}
if [[ "$MODEL" == ae ]]; then
  PKG=autoencoder; RUN=outputs/ae/z${Z}; EXTRA=()
else
  PKG=vae; RUN=outputs/vae/z${Z}_beta${BETA}; EXTRA=(--beta "$BETA")
fi
echo "$(date -Is) $(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader) :: $MODEL z=$Z ${EXTRA[*]}"
RESUME=()
[[ -f $RUN/checkpoints/last.pt ]] && RESUME=(--resume "$RUN/checkpoints/last.pt")
if [[ ! -f $RUN/train_summary.json ]]; then
  $P -u -m $PKG.train --latent-dim "$Z" "${EXTRA[@]}" --num-workers 6 "${RESUME[@]}"
fi
$P -u -m $PKG.evaluate --run-dir "$RUN"
echo "$(date -Is) done"
