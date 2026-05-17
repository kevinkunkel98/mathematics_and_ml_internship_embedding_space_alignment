  GNU nano 5.6.1                                          vit_job.sh                                                    #!/bin/bash
#SBATCH --job-name=mathAi_project_expl_ai_cnn
#SBATCH --output=out/out_%j.txt
#SBATCH --error=error/err_%j.txt
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load SciPy-bundle
module load matplotlib
module load scikit-learn/1.3.1-gfbf-2023a

cd /work/lt83cico-mathAi/lt83cico-mathAi-1769739602/

nvidia-smi

python -u main.py
