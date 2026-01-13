#!/bin/bash
#SBATCH --job-name=zeroshot_amr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=24:00:00
#SBATCH --output sbatch_outputs/slurm_%j.out

# Reading arguments
ARGS=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --group)
            GROUP="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --emb_filename)
            EMB_FILENAME="$2"
            shift 2
            ;;
        --sample_embedding_dim)
            SAMPLE_EMBEDDING_DIM="$2"
            shift 2
            ;;
        --fingerprint)
            FINGERPRINT="$2"
            shift 2
            ;;
        --fingerprint_size)
            FINGERPRINT_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown flag: $1. Available flags are: --group, --name, --emb_filename, --sample_embedding_dim, --fingerprint, --fingerprint_size."
            exit 1
            ;;
    esac
done

mkdir -p logs/"$GROUP"/"$NAME"

# Redirect stdout & stderr
exec > >(stdbuf -oL tee "logs/$GROUP/$NAME/output_log.out") 2>&1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

export PYTHONUNBUFFERED=1

echo "Training ResAMR model for experiment group: $GROUP, experiment name: $NAME, embedding filename: $EMB_FILENAME, embedding dimension: $SAMPLE_EMBEDDING_DIM, fingerprint: $FINGERPRINT, fingerprint size: $FINGERPRINT_SIZE"
srun uv run python -u ../repo/code/ResAMR_classifier.py \
    --driams_long_table ../data/combined_long_table.csv \
    --spectra_matrix ../data/"$EMB_FILENAME" \
    --sample_embedding_dim "$SAMPLE_EMBEDDING_DIM" \
    --drugs_df ../data/drug_fingerprints_Mol_selfies.csv \
    --fingerprint_class "$FINGERPRINT" \
    --fingerprint_size "$FINGERPRINT_SIZE" \
    --split_type specific \
    --split_ids ../data/data_splits.csv \
    --experiment_group "$GROUP" \
    --experiment_name "$NAME" \
    --seed 0 \
    --n_epochs 3 \
    --learning_rate 0.0003 \
    --patience 10 \
    --batch_size 512
    # --test_only