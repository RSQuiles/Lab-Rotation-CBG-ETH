#!/bin/bash
#SBATCH --job-name=select_dataset
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500G ## CHANGE IT AS NEEDED
#SBATCH --output select_dataset_%j.out

# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc
conda activate fcr_marina ## CHANGE IT WITH YOUR CONDA ENV

# Read input arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <config_name> [export_path] [control_ratio] [n_genes]"
    exit 1
fi

NAME="$1"
EXPORT_PATH="${2:-}"  # Optional argument with default value ""
RATIO="${3:-3}" # Optional argument with default value "3"
N_GENES="${4:-5000}"  # Optional argument with default value "default"

# Execute the program
# cd /cluster/work/bewi/members/rquiles/experiments/datasets
python -u select_dataset.py --file "$NAME" --ratio "$RATIO" --export "$EXPORT_PATH" --n_genes "$N_GENES"
