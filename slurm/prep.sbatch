#!/bin/bash
#SBATCH --job-name=ml_prep
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# Download IXI-T1 (resumable, size-checked), extract, slice + subject-level split -> data/processed/
set -euo pipefail
cd /home/cbathula/AutoEncoders-VariationalAutoEncoders
P=~/.conda/envs/medlatent/bin/python
RAW=data/raw/IXI-T1
TAR=$RAW/IXI-T1.tar
SIZE=4840816640
mkdir -p "$RAW"
echo "$(date -Is) $(hostname) start"
if [[ "$(stat -c %s "$TAR" 2>/dev/null || echo 0)" != "$SIZE" ]]; then
  for attempt in 1 2 3 4 5; do
    curl -sS -L --fail --retry 5 --retry-delay 20 -C - -o "$TAR" \
      https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar && break \
      || { echo "attempt $attempt failed"; sleep 60; }
  done
fi
got=$(stat -c %s "$TAR"); [[ "$got" == "$SIZE" ]] || { echo "SIZE MISMATCH: $got vs $SIZE"; exit 1; }
echo "$(date -Is) download ok"
# --download: tar already present, so this only extracts, then preprocesses
$P -u data/preprocess_ixi.py --raw-dir "$RAW" --download
echo "$(date -Is) done"
