#!/bin/bash
#SBATCH --job-name=3benchmark_piscvi
#SBATCH --mem-per-cpu=36G
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --output=run_piscvi3.out
#SBATCH --error=run_piscvi3.err
# load modules, if needed
module load stack/.2024-05-silent
module load gcc/13.2.0

# activate Conda env
source /cluster/home/mazevedo/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/work/bewi/members/mazevedo/envs
cd /cluster/work/bewi/members/mazevedo/piscvi/src

echo "==== nvidia-smi ===="
nvidia-smi
echo

python -u benchmark_onlytraining_3.py
