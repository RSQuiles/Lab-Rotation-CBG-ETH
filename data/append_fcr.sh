#!/bin/bash
#SBATCH --job-name=append_fcr 
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500G ## CHANGE IT AS NEEDED
#SBATCH --output sbatch_outputs/append_fcr_%j.out

cd $work/piscvi
srun pixi run python -u $work/data/append_fcr_embeddings.py \
    --model_path /cluster/work/bewi/members/rquiles/experiments/GDSC_controls/run1 \
    --adata_path /cluster/work/bewi/data/tahoe100/h5ad/gdsc_controls_processed.h5ad \
    --processed_adata_path /cluster/scratch/rquiles/gdsc/controls/controls_log1p.h5ad \
    --output_path /cluster/work/bewi/data/tahoe100/h5ad/gdsc_controls_processed_fcr.h5ad