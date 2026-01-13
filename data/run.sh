#!/bin/bash
#SBATCH --job-name=data_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=340G
#SBATCH --time=10:00:00
#SBATCH --output sbatch_outputs/slurm_%j.out

# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc
conda activate fcr_marina ## CHANGE IT WITH YOUR CONDA ENV

python -u normalize_gdsc_names.py

