#!/bin/bash
#SBATCH --job-name=ml_reeval
#SBATCH --partition=gpu,machinelearning,ccstar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# Re-run evaluation for every trained run whose metrics lack the realism (FID) block, e.g. runs
# evaluated with older code or whose evaluation failed. Never retrains.
set -uo pipefail
cd /home/cbathula/AutoEncoders-VariationalAutoEncoders
P=~/.conda/envs/medlatent/bin/python
for run in outputs/ae/z* outputs/vae/z* outputs_celeba/ae/z* outputs_celeba/vae/z* outputs_celeba_e100/ae/z* outputs_celeba_e100/vae/z*; do
  [[ -f $run/checkpoints/best.pt ]] || { echo "SKIP (no checkpoint) $run"; continue; }
  if $P -c "import json,sys; sys.exit(0 if 'fid_recon' in json.load(open('$run/metrics_test.json')) else 1)" 2>/dev/null; then
    echo "ok   $run"; continue; fi
  pkg=$([[ $run == */ae/* ]] && echo autoencoder || echo vae)
  echo "EVAL $run"; $P -u -m $pkg.evaluate --run-dir "$run" --num-workers 4 || echo "FAILED $run"
done
echo "$(date -Is) done"
