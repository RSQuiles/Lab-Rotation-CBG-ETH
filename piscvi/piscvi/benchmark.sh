#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=128G
#SBATCH --output benchmark_%j.out

# Execute the program
srun pixi run python -u benchmark_rafa.py
