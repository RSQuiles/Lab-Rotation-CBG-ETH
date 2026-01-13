#!/bin/bash
#SBATCH --job-name=FCR_run
#SBATCH --gpus=1
#SBATCH --gres=gpumem:38912m 
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500G ## CHANGE IT AS NEEDED
#SBATCH --output sbatch_outputs/slurm_%j.out
# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc
conda activate fcr_marina ## CHANGE IT WITH YOUR CONDA ENV

# Definition of working directories
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> <description> <parameters>"
    exit 1
fi

if [ -z "$2" ]; then
	echo "Usage: $0 <experiment_name> <description> <parameters>"
    exit 1
fi

OUTDIR="$1"
DESCRIP="$2"
shift 2   # remove the first 2 args so $@ now contains only parameters

mkdir -p "$OUTDIR"/"$DESCRIP"

# Redirect stdout & stderr
exec > >(stdbuf -oL tee "$OUTDIR"/"$DESCRIP"/output_log.out) 2>&1

export PYTHONUNBUFFERED=1

# Execute the program
python fcr_run.py --experiment "$OUTDIR" --description "$DESCRIP" --parameters "$@"
