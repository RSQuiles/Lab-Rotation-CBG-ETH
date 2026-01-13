import time
print("Starting benchmark script at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)
import sys
import os
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import scanpy as sc
import anndata as AnnData

# from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
sys.path.append(os.path.abspath("../src"))
from scib_core_gmm import Benchmarker, BioConservation, BatchCorrection

warnings.simplefilter("ignore")
sc.logging.print_header()

BATCH_KEY = 'replicate'
CELL_TYPE_KEY = 'cell_type'

# adata = sc.read("../data/NBlarge/sn_tumor_cells_NB_hvg.h5ad")
# adata.var_names = adata.var['gene_name'].copy()

#adata = sc.read("../results/models/piscVI_complete.h5ad")

adata = sc.read("../results/models/piscVI_val.h5ad")

print(adata.obsm.keys(), flush=True)

all_keys = [
    # "scVI_hvg","scVI_tanh", "scVI_nb", 
    # "piscVI", "piscVI_pathways", "piscVI_rnd", 
    # "piscVI_kegg", "piscVI_kegg_pathways", "piscVI_kegg_rnd", 
    # "piscVI_kegg_tanh", "piscVI_kegg_tanh_pathways", 
    # "piscVI_kegg_tanh2", "piscVI_kegg_tanh2_pathways", 
    # "piscVI_kegg_new", "piscVI_kegg_new_pathways", "piscVI_kegg_new_tanh", 
    # "piscVI_reactome", "piscVI_reactome_rnd",
    "scVI_val", "piscVI_val", "piscVI_val_pathways", "piscVI_val_rnd",
]

# for key in all_keys:
#     metrics = pd.read_csv(f"../results/models/{key}/metrics.csv")
#     metrics["model"] = key
#     if key == all_keys[0]:
#         all_metrics = metrics
#     else:
#         all_metrics = pd.concat([all_metrics, metrics], axis=0)

# #Save all metrics
# all_metrics.to_csv("../results/models/all_metrics.csv", index=False)

#Benchmark
all_keys = [f"X_{key}" for key in all_keys]

#for var in ['cell_type', 'cell_state', 'Response']:
var = 'condition'

valid_idx = adata.obs[var].notnull()
adata = adata[valid_idx].copy()

bm = Benchmarker(
    adata,
    batch_key=BATCH_KEY,
    label_key=var,
    bio_conservation_metrics=BioConservation(),
    batch_correction_metrics=BatchCorrection(),
    embedding_obsm_keys=all_keys,
    n_jobs=4,
)
bm.benchmark()

# Save results
nonMinMax_bm = bm.get_results(min_max_scale=False)
nonMinMax_bm.to_csv("../results/models/benchmark_results_val.csv", index=True)
bm.plot_results_table(save_dir="../results/figures/raw_val/", min_max_scale=False)

