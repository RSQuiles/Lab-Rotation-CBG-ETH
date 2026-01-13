#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --gpus=1
#SBATCH --gres=gpumem:38912m 
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=340G ## CHANGE IT AS NEEDED
#SBATCH --output sweep_log_%j.out

# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc
conda activate fcr_marina ## CHANGE IT WITH YOUR CONDA ENV

# Definition of working directories
if [ -z "$1" ]; then
    echo "Usage: $0 <sweep_id>"
    exit 1
fi

ID="$1"

# Run the sweep
wandb agent "$ID"
