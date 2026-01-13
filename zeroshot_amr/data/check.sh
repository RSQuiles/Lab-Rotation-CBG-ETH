#!/bin/bash
#SBATCH --job-name=check_embeddings 
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=24:00:00
#SBATCH --output sbatch_outputs/check_embeddings_%j.out

# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc
conda activate fcr_marina ## CHANGE IT WITH YOUR CONDA ENV

srun python -u check_embeddings.py