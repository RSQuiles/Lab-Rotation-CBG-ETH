#!/bin/bash

if [ "$1" == "--common_files" ]; then
    sbatch inputs.sh --common_files
    exit 0

elif [ "$1" == "--fingerprints" ]; then
    sbatch inputs.sh --fingerprints
    exit 0
fi

# Default input types (all)
input_types=('pcs' 'n_hvgs' 'piscvi' 'fcr')

# If user specifies input types, override defaults
specified_types=()

for arg in "$@"; do
    case "$arg" in
        --pcs)     specified_types+=("pcs") ;;
        --n_hvgs)  specified_types+=("n_hvgs") ;;
        --piscvi)  specified_types+=("piscvi") ;;
        --fcr)      specified_types+=("fcr") ;;
    esac
done

# If user specified any, use only those
if [ ${#specified_types[@]} -gt 0 ]; then
    input_types=("${specified_types[@]}")
fi


# SUBMIT JOBS
for input_type in "${input_types[@]}"; do

    # Principal Components
    if [ "$input_type" == "pcs" ]; then
        for number in 10 25 50; do
            echo "Submitting PCS ${number}"
            sbatch inputs.sh --pcs ${number}
        done
    fi

    # Highly Variable Genes
    if [ "$input_type" == "n_hvgs" ]; then
        for n_hvgs in 1000 3000 5000; do
            echo "Submitting HVG ${n_hvgs}"
            sbatch inputs.sh --n_hvgs ${n_hvgs}
        done
    fi

    # piscVI embeddings
    if [ "$input_type" == "piscvi" ]; then
        echo "Submitting piscVI"
        sbatch inputs.sh --piscvi
    fi

    # FCR embeddings
    if [ "$input_type" == "fcr" ]; then
        echo "Submitting FCR"
        sbatch inputs.sh --fcr --zx
    fi

done



