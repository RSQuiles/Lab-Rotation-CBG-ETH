#!/bin/bash
#SBATCH --job-name=evaluate_amr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=10:00:00
#SBATCH --output sbatch_outputs/evaluate_%j.out

# Reading arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <group name> [experiment name]"
    exit 1
fi

if [ ! -z "$2" ]; then
    echo "Evaluating predictions for experiment group: $1, experiment name: $2"

    export PYTHONUNBUFFERED=1

    srun uv run python -u evaluate_prediction.py \
        --group "$1" \
        --name "$2"
    exit 0
fi

GROUP="$1"
LOG_DIR="./logs/$GROUP"

export PYTHONUNBUFFERED=1

# Check directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Directory not found: $LOG_DIR"
    exit 1
fi

for dir in "$LOG_DIR"/*/; do
    # Skip if no subdirectories exist
    [ -d "$dir" ] || continue

    NAME="$(basename "$dir")"

    echo "Evaluating predictions for experiment group: $GROUP, experiment name: $NAME"

    srun uv run python -u evaluate_prediction.py \
        --group "$GROUP" \
        --name "$NAME" \
        --per_line_drug
done