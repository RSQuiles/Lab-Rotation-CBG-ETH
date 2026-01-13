#!/bin/bash
#SBATCH --job-name=process_adata 
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=340G ## CHANGE IT AS NEEDED
#SBATCH --output sbatch_outputs/slurm_%j.out

# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc
conda activate fcr_marina ## CHANGE IT WITH YOUR CONDA ENV

# Initialize an empty list of args
ARGS=()

# Loop over all provided CLI flags and options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --pcs)
            ARGS+=("--pcs")
            ;;
        --hvgs)
            ARGS+=("--hvgs")
            ;;
        --fcr)
            ARGS+=("--fcr")
            ;;
        --piscvi)
            ARGS+=("--piscvi")
            ;;
        --data_path)
            # Shift to get the value
            shift
            ARGS+=("--data_path" "$1")
            ;;
        --model_path)
            # Shift to get the value
            shift
            ARGS+=("--model_path" "$1")
            ;;
        --out)
            # Shift to get the value
            shift
            ARGS+=("--out" "$1")
            ;;
        *)
            echo "Unknown flag: $1. Available flags are --pcs, --hvgs, --fcr, --piscvi, --model_path, --data_path, --out"
            exit 1
            ;;
    esac
    shift
done

# Process the AnnData object
# srun python -u process_adata.py "${ARGS[@]}"
cd $work/piscvi
srun pixi run python -u $work/data/process_adata.py "${ARGS[@]}"