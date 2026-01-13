#!/bin/bash
#SBATCH --job-name=shard_adata
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G ## CHANGE IT AS NEEDED
#SBATCH --output sbatch_outputs/shard_adata_%j.out

# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc
conda activate fcr_marina ## CHANGE IT WITH YOUR CONDA ENV

# Read input arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <h5ad_file_path>"
    exit 1
fi

FILE="$1"

# Execute the program
# cd /cluster/work/bewi/members/rquiles/experiments/datasets
srun python -u shard_adata.py "$FILE"