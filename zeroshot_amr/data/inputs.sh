#!/bin/bash
#SBATCH --job-name=inputs_zeroshot_amr 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=340G
#SBATCH --time=3:00:00
#SBATCH --output sbatch_outputs/slurm_%j.out

source ../.venv/bin/activate

ARGS=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --pcs)
            shift
            ARGS+=("--pcs" "$1")
            ;;
        --n_hvgs) 
            shift
            ARGS+=("--n_hvgs" "$1")
            ;;
        --piscvi)
            ARGS+=("--piscvi")
            ;;
        --fcr)
            ARGS+=("--fcr")
            ;;
        --zx)
            ARGS+=("--zx")
            ;;
        --zxt)
            ARGS+=("--zxt")
            ;;
        --zt)
            ARGS+=("--zt")
            ;;
        --data_path)
            shift
            ARGS+=("--data_path" "$1")
            ;;
        --adata_path)
            shift
            ARGS+=("--adata_path" "$1")
            ;;
        --name)
            shift
            ARGS+=("--name" "$1")
            ;;
        --common_files)
            ARGS+=("--common_files")
            ;;
        --fingerprints)
            ARGS+=("--fingerprints")
            ;;
        *)
            echo "Unknown flag: $1. Available flags are --pcs, --n_hvgs, --data_path, --name, --common_files, --fingerprints."
            exit 1
            ;;
    esac
    shift
done

srun uv run python -u generate_inputs.py "${ARGS[@]}"