#!/bin/bash
#SBATCH --job-name=FCR_run
#SBATCH --gpus=1 
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=340G ## CHANGE IT AS NEEDED
#SBATCH --output sbatch_outputs/slurm_%j.out
# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc
conda activate fcr_marina ## CHANGE IT WITH YOUR CONDA ENV

# Definition of working directories
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> <description>"
    exit 1
fi

if [ -z "$2" ]; then
	echo "Usage: $0 <experiment_name> <description>"
    exit 1
fi

EXP="$1"
DESCRIP="$2"

export PYTHONUNBUFFERED=1

# Execute the program
python view_umaps.py -e "$EXP" -d "$DESCRIP" --all_drugs