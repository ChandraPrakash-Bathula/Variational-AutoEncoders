#!/bin/bash
#SBATCH --job-name=ml_run
#SBATCH --partition=gpu,machinelearning,ccstar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x_%A_%a.out
# sbatch --array=1-N%K slurm/run_manifest.sbatch manifests/runs_v2.tsv
# Row i (after header): dataset model z beta. Train (resume-safe) + evaluate.
set -euo pipefail
cd /home/cbathula/AutoEncoders-VariationalAutoEncoders
P=~/.conda/envs/medlatent/bin/python
IFS=$'\t' read -r DS MODEL Z BETA <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")"
SUFFIX=""; OUT=outputs; XTRA=()
[[ $DS == celeba ]] && { SUFFIX=_celeba; OUT=outputs_celeba; }
[[ $DS == celeba100 ]] && { SUFFIX=_celeba; OUT=outputs_celeba_e100; XTRA=(--epochs 100 --output-dir $OUT); }
if [[ $MODEL == ae ]]; then PKG=autoencoder; RUN=$OUT/ae/z$Z; EXTRA=()
else PKG=vae; RUN=$OUT/vae/z${Z}_beta$BETA; EXTRA=(--beta "$BETA"); fi
echo "$(date -Is) $(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader) :: $DS $MODEL z=$Z ${EXTRA[*]}"
RESUME=(); [[ -f $RUN/checkpoints/last.pt ]] && RESUME=(--resume "$RUN/checkpoints/last.pt")
[[ -f $RUN/train_summary.json ]] || $P -u -m $PKG.train --config $PKG/config$SUFFIX.yaml --latent-dim "$Z" "${EXTRA[@]}" --num-workers 6 "${XTRA[@]}" "${RESUME[@]}"
$P -u -m $PKG.evaluate --run-dir "$RUN" --num-workers 4
echo "$(date -Is) done"
