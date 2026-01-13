#!/bin/bash
#SBATCH --job-name=prepare_plates
#SBATCH --gpus=1
#SBATCH --gres=gpumem:38912m
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=340G ## CHANGE IT AS NEEDED

# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc
conda activate fcr_marina ## CHANGE IT WITH YOUR CONDA ENV

# Iterate through all plates
for i in {1..14}; do
    echo ">>> Processing plate $i"
	python prepare_plate.py --plate "$i"
done
