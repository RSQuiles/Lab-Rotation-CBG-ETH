import sys
import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as AnnData

adata = sc.read("../results/models/piscVI_final.h5ad")
print("Loaded adata")

# print(adata_test_1.obs.keys(), flush=True)
# print(adata_test_1.uns.keys(), flush=True)
# print(adata_test_1.obsm.keys(), flush=True)
# print(adata_test_1.obsp.keys(), flush=True)

# get difference between adata_test_1 and adata_test_2
# for store in ["obs", "uns", "obsm", "obsp"]:
#     extra_keys_in_adata_test_1 = set(getattr(adata_test_1, store).keys()) - set(getattr(adata_test_2, store).keys())
#     extra_keys_in_adata_test_2 = set(getattr(adata_test_2, store).keys()) - set(getattr(adata_test_1, store).keys())
    
#     if extra_keys_in_adata_test_1:
#         print(f"Keys in adata_test_1.{store} not in adata_test_2.{store}: {extra_keys_in_adata_test_1}", flush=True)
#     if extra_keys_in_adata_test_2:
#         print(f"Keys in adata_test_2.{store} not in adata_test_1.{store}: {extra_keys_in_adata_test_2}", flush=True)


all_keys = [
    "scVI_tanh", "scVI_nb",
    "piscVI_kegg_tanh2", "piscVI_kegg_tanh2_pathways", "piscVI_kegg_tanh2_rnd", "piscVI_kegg_tanh2_pathways_rnd",
    "piscVI_kegg_new", "piscVI_kegg_new_pathways", "piscVI_kegg_new_tanh",
    "piscVI_kegg_tanh", "piscVI_kegg_tanh_pathways", "piscVI_kegg_tanh_rnd", "piscVI_kegg_tanh_pathways_rnd"
    ]

for file in [ "piscVI_kegg", "piscVI_tanh", "piscVI_new", "scVI"]:
    adata_kegg = sc.read(f"../results/models/{file}.h5ad")
    print(f"Processing piscVI version {file}", flush=True)
    
    if file == "piscVI_tanh":
        keys = ["piscVI_kegg_tanh2", "piscVI_kegg_tanh2_pathways", "piscVI_kegg_tanh2_rnd", "piscVI_kegg_tanh2_pathways_rnd"]
    elif file == "piscVI_new":
        keys = ["piscVI_kegg_new", "piscVI_kegg_new_pathways", "piscVI_kegg_new_tanh"]
    elif file == "scVI":
        keys = ["scVI_tanh", "scVI_nb"]
    elif file == "piscVI_kegg":
        keys = ["piscVI_kegg_tanh", "piscVI_kegg_tanh_pathways", "piscVI_kegg_tanh_rnd", "piscVI_kegg_tanh_pathways_rnd"]

    for key in keys:
        adata.obsm[f"X_{key}"] = adata_kegg.obsm[f"X_{key}"]
        try:
            adata.obsm[f"umap_{key}"] = adata_kegg.obsm[f"umap_{key}"]
            adata.obs[f'leiden_{key}'] = adata_kegg.obs[f'leiden_{key}']
            adata.obs[f'gmm_{key}'] = adata_kegg.obs[f'gmm_{key}']
            adata.uns[f'leiden_{key}'] = adata_kegg.uns[f'leiden_{key}']
            adata.uns[f'leiden_{key}_colors'] = adata_kegg.uns[f'leiden_{key}_colors']
            adata.uns[f'neighbors_{key}'] = adata_kegg.uns[f'neighbors_{key}']
            adata.uns[f'umap_{key}'] = adata_kegg.uns[f'umap_{key}']
            adata.obsp[f'neighbors_{key}_connectivities'] = adata_kegg.obsp[f'neighbors_{key}_connectivities']
            adata.obsp[f'neighbors_{key}_distances'] = adata_kegg.obsp[f'neighbors_{key}_distances']
        except:
            print(f"Skipping some keys for {key} due to missing data", flush=True)

    adata.write("../results/models/piscVI_final_2.h5ad")

print("All keys processed. Final check for missing keys...", flush=True)
for key in all_keys:
    if f"X_{key}" not in adata.obsm.keys():
        print(f"Key X_{key} is still missing after processing all versions", flush=True)
