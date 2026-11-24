#!/bin/bash
#SBATCH --job-name=ml_final_an
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# sbatch slurm/final_analysis.sbatch ixi|celeba
# Per latent size: latent-space plots of AE, VAE beta=0.1, VAE beta=1 side by side; then the results
# table + trend figure over all runs of the dataset.
set -euo pipefail
cd /home/cbathula/AutoEncoders-VariationalAutoEncoders
P=~/.conda/envs/medlatent/bin/python
DS=$1; OUT=outputs; TITLE="IXI T1 brain MRI (128x128)"
[[ $DS == celeba ]] && { OUT=outputs_celeba; TITLE="CelebA faces (64x64 RGB)"; }
[[ $DS == celeba100 ]] && { OUT=outputs_celeba_e100; TITLE="CelebA faces (64x64 RGB), 100 epochs"; }
for zdir in $(ls -d $OUT/ae/z* | sort -V); do
  z=${zdir##*/z}
  runs=($OUT/ae/z$z); for b in 0.1 1; do [[ -d $OUT/vae/z${z}_beta$b/latents ]] && runs+=($OUT/vae/z${z}_beta$b); done
  $P -u -m analysis.visualize_latents --run-dirs "${runs[@]}" --comparison-dir $OUT/analysis/latent_comparison_z$z
done
$P -u -m analysis.compare_models --output-dir $OUT --out-dir $OUT/analysis --title "$TITLE" \
  --ae-run $OUT/ae/z128 --vae-run $OUT/vae/z128_beta0.1
echo "$(date -Is) done"
