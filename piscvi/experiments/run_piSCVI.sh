#!/bin/bash
#SBATCH --job-name=piSCVI_run
#SBATCH --gpus=1
#SBATCH --gres=gpumem:38912m 
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G ## CHANGE IT AS NEEDED
#SBATCH --output sbatch_outputs/piscvi_exp_%j.out

# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc

if [ -z "$1" ]; then
     echo "Usage: $0 <config_file_name>"
     exit 1
 fi

CONFIG="$1"

echo "Using config file: $CONFIG"

# Get experiment name
NAME=$(awk -F'"' '/"name"/ {print $4}' "$CONFIG")
mkdir -p "$NAME"

# Redirect stdout & stderr
exec > >(stdbuf -oL tee "$NAME"/output_log.out) 2>&1

export PYTHONUNBUFFERED=1

# Execute the program
pixi run python -u run_model.py --config "$CONFIG"
