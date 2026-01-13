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
from sklearn.metrics import (
    silhouette_score, 
    adjusted_rand_score, 
    normalized_mutual_info_score,
    homogeneity_score,
)
from sklearn.mixture import GaussianMixture
import anndata as AnnData
# import scvi
# print(f'Imported scvi at time {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', flush=True)
# from scvi.model import SCVI
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

sys.path.append(os.path.abspath("../src"))
# from model import InformedSCVI
# from pathway import get_pathway_masks, get_random_masks, filter_genes
# from train import plot_loss, split_train_val

print(f"Torch cuda available: {torch.cuda.is_available()}", flush=True)
warnings.simplefilter("ignore")
sc.logging.print_header()
sc.settings.figsize = (10, 10)
sc.settings.figdir = "../results/figures/"
#scvi.settings.seed = 0
#print("Last run with scvi-tools version:", scvi.__version__, flush=True)
sns.set_theme()
torch.set_float32_matmul_precision("high")

BATCH_KEY = 'donor_id'
CELL_TYPE_KEY = 'cell_type'


# def compute_metrics(adata, adata_piscVI, key):
#     model = InformedSCVI.load(f"../results/models/{key}", adata=adata)

#     with open(f"../results/models/{key}/train_time.txt", "r") as f:
#         train_time = f.read().strip()

#     adata.obsm[f"X_{key}"] = adata_piscVI.obsm[f"X_{key}"].copy()
#     adata.obs[f"leiden_{key}"] = adata_piscVI.obs[f"leiden_{key}"].copy()
#     adata.uns[f"leiden_{key}"] = adata_piscVI.uns[f"leiden_{key}"].copy()

#     # Calculate metrics
#     elbo = model.get_elbo().item()
#     reconstruction_error = model.get_reconstruction_error()['reconstruction_loss'].item()
#     silhouette = silhouette_score(adata.obsm[f"X_{key}"], adata.obs[f"leiden_{key}"], metric="euclidean")
#     metrics = pd.DataFrame(
#         {
#             "elbo": [elbo],
#             "reconstruction_error": [reconstruction_error],
#             "silhouette": [silhouette],
#             "train_time": [train_time],
#         }   
#     )
#     for var in ['cell_type','cell_state']: #,'Response']:  
#         # if var == 'Response':
#         #     valid_idx = adata.obs['Response'].notnull()
#         #     adata = adata[valid_idx].copy()
#         ARI = adjusted_rand_score(adata.obs[var], adata.obs[f"leiden_{key}"])
#         NMI = normalized_mutual_info_score(adata.obs[var], adata.obs[f"leiden_{key}"])
#         homogeneity = homogeneity_score(adata.obs[var], adata.obs[f"leiden_{key}"])
#         # Get number of categories of that variable
#         n_categories = adata.obs[var].nunique()
#         gmm = GaussianMixture(n_components=n_categories, random_state=42)
#         gmm.fit(adata.obsm[f"X_{key}"])
#         adata.obs[f"gmm_{var}_{key}"] = gmm.predict(adata.obsm[f"X_{key}"])
#         ARI_gmm = adjusted_rand_score(adata.obs[var], adata.obs[f"gmm_{var}_{key}"])
#         NMI_gmm = normalized_mutual_info_score(adata.obs[var], adata.obs[f"gmm_{var}_{key}"])
#         homogeneity_gmm = homogeneity_score(adata.obs[var], adata.obs[f"gmm_{var}_{key}"])
#         metrics_var = pd.DataFrame(
#             {
#                 f"ARI {var}": [ARI],
#                 f"NMI {var}": [NMI],
#                 f"homogeneity {var}": [homogeneity],
#                 f"ARI gmm {var}": [ARI_gmm], 
#                 f"NMI gmm {var}": [NMI_gmm],
#                 f"homogeneity gmm {var}": [homogeneity_gmm],
#             }
#         )

#         metrics = pd.concat([metrics, metrics_var], axis=1)
#     metrics.to_csv(f"../results/models/{key}/metrics.csv", index=False)

#     return adata


# adata = sc.read("../data/NBlarge/sn_tumor_cells_NB_hvg.h5ad")
# adata.var_names = adata.var['gene_name'].copy()
print('Ready to load adata', flush=True)
adata = sc.read("../../results/models/piscVI_complete.h5ad")

# print('Starting scVI models at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)
# adata = compute_metrics(
#     adata=adata,
#     adata_piscVI=adata_piscVI,
#     key="scVI_hvg",
# )
# adata.write("../results/models/piscVI.h5ad")

# print('Starting piscVI models at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)

# # Standard piscVI
# genes_per_pathway, genes_per_circuit, circuits_per_pathway = get_pathway_masks()
# adata, genes_per_pathway, genes_per_circuit, circuits_per_pathway = filter_genes(adata, genes_per_pathway, genes_per_circuit, circuits_per_pathway)
# frac = genes_per_circuit.sum(axis=1).mean() / genes_per_circuit.shape[1]
# print(f"Fraction of genes per circuit: {frac:.2f}", flush=True)
# print(f"Number of genes: {genes_per_circuit.shape[1]}", flush=True)
# rnd_genes_per_pathway, rnd_genes_per_circuit, rnd_circuits_per_pathway = get_random_masks(adata.var_names, genes_per_circuit.shape[0], genes_per_pathway.shape[0], frac=frac, seed=42)

# masks_keggNB = [genes_per_circuit, circuits_per_pathway]
# masks_keggNB_rnd = [rnd_genes_per_circuit, rnd_circuits_per_pathway]
# masks_keggNB_pathways = [genes_per_pathway]
# masks_keggNB_pathways_rnd = [rnd_genes_per_pathway]

# masks_list = [masks_keggNB, masks_keggNB_pathways, masks_keggNB_rnd, masks_keggNB_pathways_rnd]
# keys_list = ["piscVI","piscVI_pathways", "piscVI_rnd", "piscVI_pathways_rnd"]

# for masks, key in zip(masks_list, keys_list):
#     print(f"Running model {key} with masks {[mask.shape for mask in masks]}", flush=True)
#     adata = compute_metrics(
#         adata=adata,
#         adata_piscVI=adata_piscVI,
#         key=key,
#     )

# # Different setups
# distributions_list = ["zinb", "nb", "nb"]
# activations_list = ["tanh", "relu", "tanh"]

# for distribution, activation in zip(distributions_list, activations_list):
#     key = f"piscVI_{distribution}_{activation}"
#     print(f"Running model {key}", flush=True)
#     adata = compute_metrics(
#         adata=adata,
#         adata_piscVI=adata_piscVI,
#         key=key,
#     )


# distributions_list = ["normal", "normal"]
# activations_list = ["tanh", "sigmoid"]

# for distribution, activation in zip(distributions_list, activations_list):
#     key = f"piscVI_{distribution}_{activation}"
#     print(f"Running model {key}", flush=True)
#     adata = compute_metrics(
#         adata=adata,
#         adata_piscVI=adata_piscVI,
#         key=key,
#     )

# del adata

# # Different pathway masks
# adata_kegg = sc.read("../data/NBlarge/sn_tumor_cells_NB_hvg.h5ad")
# adata_kegg.var_names = adata_kegg.var['gene_name'].copy()

# genes_per_pathway, genes_per_circuit, circuits_per_pathway = get_pathway_masks("KEGG")
# adata_kegg, genes_per_pathway, genes_per_circuit, circuits_per_pathway = filter_genes(adata_kegg, genes_per_pathway, genes_per_circuit, circuits_per_pathway)
# frac = genes_per_circuit.sum(axis=1).mean() / genes_per_circuit.shape[1]
# rnd_genes_per_pathway, rnd_genes_per_circuit, rnd_circuits_per_pathway = get_random_masks(adata_kegg.var_names, genes_per_circuit.shape[0], genes_per_pathway.shape[0], frac=frac, seed=42)

# masks_kegg = [genes_per_circuit, circuits_per_pathway]
# masks_kegg_rnd = [rnd_genes_per_circuit, rnd_circuits_per_pathway]
# masks_kegg_pathways = [genes_per_pathway]
# masks_kegg_pathways_rnd = [rnd_genes_per_pathway]

# masks_list = [masks_kegg, masks_kegg_pathways, masks_kegg_rnd, masks_kegg_pathways_rnd]
# keys_list = ["piscVI_kegg", "piscVI_kegg_pathways", "piscVI_kegg_rnd", "piscVI_kegg_pathways_rnd"]

# for masks, key in zip(masks_list, keys_list):
#     print(f"Running model {key} with masks {[mask.shape for mask in masks]}", flush=True)
#     adata_kegg = compute_metrics(
#         adata=adata_kegg,
#         adata_piscVI=adata_piscVI,
#         key=key,  
#     )

# del adata_kegg

# adata_reactome = sc.read("../data/NBlarge/sn_tumor_cells_NB_hvg.h5ad")
# adata_reactome.var_names = adata_reactome.var['gene_name'].copy()

# genes_per_pathway, genes_per_circuit, circuits_per_pathway = get_pathway_masks("Reactome")
# adata_reactome, genes_per_pathway, _, _ = filter_genes(adata_reactome, genes_per_pathway, genes_per_circuit, circuits_per_pathway)
# frac = genes_per_pathway.sum(axis=1).mean() / genes_per_pathway.shape[1]
# rnd_genes_per_pathway, rnd_genes_per_circuit, rnd_circuits_per_pathway = get_random_masks(adata_reactome.var_names, genes_per_pathway.shape[0]*20, genes_per_pathway.shape[0], frac=frac, seed=42)

# masks_reactome = [genes_per_pathway]
# masks_reactome_rnd = [rnd_genes_per_pathway]

# adata_reactome = adata_reactome.copy()

# masks_list = [masks_reactome, masks_reactome_rnd]
# keys_list = ["piscVI_reactome", "piscVI_reactome_rnd"]

# for masks, key in zip(masks_list, keys_list):
#     print(f"Running model {key} with masks {[mask.shape for mask in masks]}", flush=True)
#     adata_reactome = compute_metrics(
#         adata_reactome,
#         key=key, 
#     )

# # Test with more hvgs
# for db in ['KEGG_NB', 'KEGG', 'Reactome']:
#     adata_7k = sc.read(f"../data/NBlarge/sn_tumor_cells_NB_7000hvg.h5ad")
#     adata_7k.var_names = adata_7k.var['gene_name'].copy()
#     genes_per_pathway, genes_per_circuit, circuits_per_pathway = get_pathway_masks(db)
#     adata_7k, genes_per_pathway, genes_per_circuit, circuits_per_pathway = filter_genes(adata_7k, genes_per_pathway, genes_per_circuit, circuits_per_pathway)

#     if db == 'Reactome':
#         masks = [genes_per_pathway]
#         masks_list = [masks]
#         keys_list = [f"piscVI_7k_{db}"]
#     else:
#         masks = [genes_per_circuit, circuits_per_pathway]
#         masks_pathway = [genes_per_pathway]
#         masks_list = [masks, masks_pathway]
#         keys_list = [f"piscVI_7k_{db}", f"piscVI_7k_{db}_pathways"]

#     for masks, key in zip(masks_list, keys_list):
#         print(f"Running model {key} with masks {[mask.shape for mask in masks]}", flush=True)
#         adata_7k = compute_metrics(
#             adata=adata_7k,
#             adata_piscVI=adata_piscVI,
#             key=key,   
#         )

# #Combine all metrics

all_keys = [
    "scVI_hvg", 
    "piscVI", "piscVI_pathways", "piscVI_rnd", "piscVI_pathways_rnd",
    "piscVI_zinb_tanh", "piscVI_nb_relu", "piscVI_nb_tanh",
    "piscVI_normal_tanh", "piscVI_normal_sigmoid",
    "piscVI_kegg", "piscVI_kegg_pathways", "piscVI_kegg_rnd", "piscVI_kegg_pathways_rnd",
    "piscVI_reactome", "piscVI_reactome_rnd",
    "piscVI_7k_KEGG_NB", "piscVI_7k_KEGG_NB_pathways", "piscVI_7k_KEGG", "piscVI_7k_KEGG_pathways", "piscVI_7k_Reactome"
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
var = 'cell_state'

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

bm.plot_results_table(min_max_scale=False)

