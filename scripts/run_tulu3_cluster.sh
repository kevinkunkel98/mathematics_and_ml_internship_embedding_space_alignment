#!/bin/bash
#SBATCH --job-name=mathml_tulu3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --partition=paula
#SBATCH --time=36:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

source myenv/bin/activate

nvidia-smi

cd mathematics_and_ml_internship_embedding_space_alignment

python scripts/extract_embeddings.py \
    --trajectory tulu3 \
    --n-rows 2000 \
    --batch-size 4
