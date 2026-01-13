#!/bin/bash

srun uv run python3 ../repo/code/ResAMR_classifier.py \
    --driams_long_table data/combined_long_table.csv \
    --spectra_matrix data/rawSpectra_data.npy \
    --sample_embedding_dim 6000 \
    --drugs_df data/drug_fingerprints_Mol_selfies.csv \
    --fingerprint_class morgan_1024 \
    --fingerprint_size 1024 \
    --split_type specific \
    --split_ids data/data_splits.csv \
    --experiment_group rawMS_MorganFing \
    --experiment_name ResMLP \
    --seed 0 \
    --n_epochs 2 \
    --learning_rate 0.0003 \
    --patience 10 \
    --batch_size 128
