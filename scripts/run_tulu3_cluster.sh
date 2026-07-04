#!/bin/bash
#SBATCH --job-name=tulu3_extract
#SBATCH --output=logs/tulu3_extract_%j.out
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G

# Usage: sbatch scripts/run_tulu3_cluster.sh
#
# Extracts layer-wise hidden states from all 3 AllenAI Tulu-3-8B checkpoints
# (SFT → DPO → RLHF) on hh-rlhf.  Models are fully public — no HF approval needed.
# Estimated wall time: ~8–10 h for n=2000, batch_size=4 across 3 models.

set -euo pipefail

module load python/3.10
module load cuda/11.8

# ── Edit this to match your cluster home directory ──────────────────────────
PROJECT_DIR="/work/lt83cico_mathAi/mathematics_and_ml_internship_embedding_space_alignment"
# ────────────────────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"
mkdir -p logs

# Load HF token if present (optional — Tulu-3 models are ungated)
if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] Starting Tulu-3 extraction (n=2000, batch_size=4) ..."

"$PROJECT_DIR/.venv/bin/python" scripts/extract_embeddings.py \
    --trajectory tulu3 \
    --n-rows 2000 \
    --batch-size 4 \
    2>&1 | tee logs/tulu3_extract_$SLURM_JOB_ID.log

echo "[$(date)] Extraction complete."
echo "Output files:"
ls -lh data/embeddings/allenai--*/layers.h5 2>/dev/null || echo "  (none found — check the log)"
