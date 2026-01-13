# Read input arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <embedding_filename> <embedding_dimension> <fingerprint size> <group_name> <experiment_name>"
    exit 1
fi

EMB_FILE="$1"
EMB_DIM="$2"
FP_SIZE="$3"
GROUP="$4"
NAME="$5"

# Queue all experiments for the specified group
sbatch run.sh \
    --group "$GROUP" \
    --name "$NAME" \
    --emb_filename "$EMB_FILE" \
    --sample_embedding_dim "$EMB_DIM" \
    --fingerprint "$GROUP" \
    --fingerprint_size "$FP_SIZE" \