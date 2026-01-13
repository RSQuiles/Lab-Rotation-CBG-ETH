sbatch process_adata.sh \
    --data_path /cluster/work/bewi/data/tahoe100/h5ad/controls_merged.h5ad \
    --out /cluster/work/bewi/data/tahoe100/h5ad/controls_processed.h5ad \
    --pcs \
    --hvgs \
    --piscvi \
    --model_path /cluster/work/bewi/members/rquiles/piscvi/experiments/tahoe_controls_counts \

    