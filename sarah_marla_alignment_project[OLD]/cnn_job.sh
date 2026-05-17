#!/bin/bash
#SBATCH --job-name=mathAi_project_expl_ai
#SBATCH --output=out_%j.txt
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load python/3.10
module load cuda/11.8

cd /work/lt83cico_mathAi

python main_cnn.py