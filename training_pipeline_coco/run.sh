#!/bin/bash
#SBATCH --job-name=coco_training
#SBATCH --output=out/training_%j.log
#SBATCH --error=error/training_%j.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# %j is the job ID — keeps logs separate per run

module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

pip install --user \
            "transformers==4.46.3" \
                "peft==0.10.0" \
                    pycocotools Pillow pandas pyyaml scikit-learn trl accelerate

cd /work2/lt83cico-mathAi/training_pipeline_coco

python main.py                                                                                                                       ~                                                                                                                       ~                                                                                                                       ~                                                                                                                       ~                                                                                                                       ~                                                                                                                       ~                    