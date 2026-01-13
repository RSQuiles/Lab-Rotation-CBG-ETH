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
import scvi
print(f'Imported scvi at time {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', flush=True)
from scvi.model import SCVI
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

sys.path.append(os.path.abspath("../src"))
from model import InformedSCVI
from pathway import get_pathway_masks, get_random_masks, filter_genes
from train import plot_loss, split_train_val

print(f"Torch cuda available: {torch.cuda.is_available()}", flush=True)
warnings.simplefilter("ignore")
sc.logging.print_header()
sc.settings.figsize = (10, 10)
sc.settings.figdir = "../results/figures/"
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__, flush=True)
sns.set_theme()
torch.set_float32_matmul_precision("high")

BATCH_KEY = 'donor_id'
CELL_TYPE_KEY = 'cell_type'


def run_model(adata, 
              activation="relu",
              masks=None,
              likelihood="zinb",
              #layer="X", 
              key="scVI"):
    #Check if model already exists
    if not os.path.exists(f"../results/models/{key}"):
        # Define model
        InformedSCVI.setup_anndata(
            adata,
            #layer=layer,
            batch_key=BATCH_KEY,
        )
        print(type(masks), flush=True)
        model = InformedSCVI(
            adata,
            gene_likelihood=likelihood,
            activation=activation,
            masks=masks,
        )

        # Train model
        train_indices, val_indices = split_train_val(adata, 'cell_state')
        early_stopping_kwargs = {
            "early_stopping": True,
            "early_stopping_patience": 10,
            #"early_stopping_monitor": "validation_loss",
        }
        datasplitter_kwargs = {
            "external_indexing": [train_indices, val_indices],
        }
        training_plan_kwargs = {
            "reduce_lr_on_plateau": True,
            "lr_patience": 8, 
            "lr_factor": 0.1 
        }

        start_time = time.time()
        model.train(max_epochs=500, datasplitter_kwargs=datasplitter_kwargs, plan_kwargs=training_plan_kwargs, **early_stopping_kwargs) 
        end_time = time.time()

        # Save model
        model.save(f"../results/models/{key}", overwrite=True, save_anndata=False)
        
        # Print and save training time (in hours, minutes, seconds)
        train_time = end_time - start_time
        hours, rem = divmod(train_time, 3600)
        minutes, seconds = divmod(rem, 60)
        train_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        print(f"Training time for {key}: {train_time}", flush=True)
        os.makedirs(f"../results/models/{key}", exist_ok=True)
        with open(f"../results/models/{key}/train_time.txt", "w") as f:
            f.write(train_time)  
        
        #compute_metrics(model=model, adata=adata, key=key)

    else:
        print(f"Model {key} already exists, proceeding to compute metrics.", flush=True)
        model = InformedSCVI.load(f"../results/models/{key}", adata=adata)
        
    # Save latent representation
    latent = model.get_latent_representation()
    adata.obsm[f"X_{key}"] = latent

    plot_loss(model, save_path=f"../results/figures/{key}/loss.png")

    # Save normalized expression
    adata_subset = adata[adata.obs[CELL_TYPE_KEY] == "neuroblast (sensu Vertebrata)"]
    denoised = model.get_normalized_expression(adata_subset, library_size=1e4)
    denoised = pd.DataFrame(
        denoised,
        index=adata_subset.obs_names,
        columns=adata_subset.var_names,
    )
    os.makedirs(f"../results/models/{key}", exist_ok=True)
    denoised.to_csv(f"../results/models/{key}/denoised_data.csv")

    # Clustering
    if "X_pca" not in adata.obsm.keys():
        sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep=f"X_{key}", key_added=f"neighbors_{key}")
    sc.tl.umap(adata, key_added=f"umap_{key}", neighbors_key=f"neighbors_{key}", min_dist=0.3)
    sc.tl.leiden(adata, key_added=f"leiden_{key}", resolution=0.5, neighbors_key=f"neighbors_{key}")
    sc.settings.figdir = f"../results/figures/{key}/"
    for feature in [f"leiden_{key}", CELL_TYPE_KEY, "cell_state", "Response", BATCH_KEY]:
        sc.pl.embedding(adata, basis=f"umap_{key}", color=feature, title=feature, frameon=False, show=False, save=f"_{feature}.png")

        #compute_metrics(adata=adata, key=key)
    
    return adata

def compute_metrics(adata, key, model=None):
    if model is None:
        model = InformedSCVI.load(f"../results/models/{key}", adata=adata)

    with open(f"../results/models/{key}/train_time.txt", "r") as f:
        train_time = f.read().strip()

    
    # Calculate metrics
    elbo = model.get_elbo().item()
    reconstruction_error = model.get_reconstruction_error()['reconstruction_loss'].item()
    silhouette = silhouette_score(adata.obsm[f"X_{key}"], adata.obs[f"leiden_{key}"], metric="euclidean")
    metrics = pd.DataFrame(
        {
            "elbo": [elbo],
            "reconstruction_error": [reconstruction_error],
            "silhouette": [silhouette],
            "train_time": [train_time],
        }   
    )
    for var in ['cell_type','cell_state','Response']:  
        ARI = adjusted_rand_score(adata.obs[var], adata.obs[f"leiden_{key}"])
        NMI = normalized_mutual_info_score(adata.obs[var], adata.obs[f"leiden_{key}"])
        homogeneity = homogeneity_score(adata.obs[var], adata.obs[f"leiden_{key}"])
        # Get number of categories of that variable
        n_categories = adata.obs[var].nunique()
        gmm = GaussianMixture(n_components=n_categories, random_state=42)
        gmm.fit(adata.obsm[f"X_{key}"])
        adata.obs[f"gmm_{var}_{key}"] = gmm.predict(adata.obsm[f"X_{key}"])
        ARI_gmm = adjusted_rand_score(adata.obs[var], adata.obs[f"gmm_{var}_{key}"])
        NMI_gmm = normalized_mutual_info_score(adata.obs[var], adata.obs[f"gmm_{var}_{key}"])
        homogeneity_gmm = homogeneity_score(adata.obs[var], adata.obs[f"gmm_{var}_{key}"])
        metrics_var = pd.DataFrame(
            {
                f"ARI {var}": [ARI],
                f"NMI {var}": [NMI],
                f"homogeneity {var}": [homogeneity],
                f"ARI gmm {var}": [ARI_gmm], 
                f"NMI gmm {var}": [NMI_gmm],
                f"homogeneity gmm {var}": [homogeneity_gmm],
            }
        )

        metrics = pd.concat([metrics, metrics_var], axis=1)
    metrics.to_csv(f"../results/models/{key}/metrics.csv", index=False)

    return


adata = sc.read("../data/NBlarge/sn_tumor_cells_NB_hvg.h5ad")
adata.var_names = adata.var['gene_name'].copy()

# adata = sc.read("../results/models/piscVI.h5ad")

#Initial plots and save
# sc.pp.pca(adata)
# sc.pp.neighbors(adata, n_pcs=30, n_neighbors=10)
# sc.tl.umap(adata, min_dist=0.2)
# sc.tl.leiden(adata)
# sc.settings.figdir = "../results/figures/raw/"
# for feature in ["leiden", CELL_TYPE_KEY, "cell_state", "Response", BATCH_KEY]:
#     sc.pl.umap(adata, color=feature, title=feature, frameon=False, show=False, save=f"_initial_{feature}.png")

# Standard scVI (hvgs)
# sc.pp.highly_variable_genes(
#     adata_hvg, flavor="seurat_v3", layer="RNA", n_top_genes=2000, subset=True
# )

# print('Starting scVI models at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)
# adata = run_model(
#     adata,
#     key="scVI_hvg",
# )
# adata.write("../results/models/piscVI.h5ad")

# # # Standard scVI (functional genes)

# # adata_hvg = adata_hvg[:, ~adata_hvg.var['ensembl_gene_id'].isna()]
# # adata_hvg = adata_hvg.copy()

# # adata_hvg = run_model(
# #     adata_hvg,
# #     key="scVI_hvg_funct",
# # )

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

# adata = adata.copy()

# masks_list = [masks_keggNB, masks_keggNB_pathways, masks_keggNB_rnd, masks_keggNB_pathways_rnd]
# keys_list = ["piscVI","piscVI_pathways", "piscVI_rnd", "piscVI_pathways_rnd"]

# for masks, key in zip(masks_list, keys_list):
#     print(f"Running model {key} with masks {[mask.shape for mask in masks]}", flush=True)
#     adata = run_model(
#         adata,
#         key=key,
#         masks=masks,
#     )
#     adata.write("../results/models/piscVI.h5ad")

# # Different setups
# distributions_list = ["zinb", "nb", "nb"]
# activations_list = ["tanh", "relu", "tanh"]

# for distribution, activation in zip(distributions_list, activations_list):
#     key = f"piscVI_{distribution}_{activation}"
#     print(f"Running model {key}", flush=True)
#     adata = run_model(
#         adata,
#         key=key,
#         likelihood=distribution,
#         activation=activation,
#         masks=masks_keggNB,
#     )
#     adata.write("../results/models/piscVI.h5ad")

# sc.pp.log1p(adata)

# distributions_list = ["normal", "normal"]
# activations_list = ["tanh", "sigmoid"]

# for distribution, activation in zip(distributions_list, activations_list):
#     key = f"piscVI_{distribution}_{activation}"
#     print(f"Running model {key}", flush=True)
#     adata = run_model(
#         adata,
#         key=key,
#         likelihood=distribution,
#         activation=activation,
#         masks=masks_keggNB,
#     )
#     adata.write("../results/models/piscVI.h5ad")

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

# adata_kegg = adata_kegg.copy()

# masks_list = [masks_kegg, masks_kegg_pathways, masks_kegg_rnd, masks_kegg_pathways_rnd]
# keys_list = ["piscVI_kegg", "piscVI_kegg_pathways", "piscVI_kegg_rnd", "piscVI_kegg_pathways_rnd"]

# for masks, key in zip(masks_list, keys_list):
#     print(f"Running model {key} with masks {[mask.shape for mask in masks]}", flush=True)
#     adata_kegg = run_model(
#         adata_kegg,
#         key=key,
#         masks=masks,   
#     )
#     adata.obsm[f"X_{key}"] = adata_kegg.obsm[f"X_{key}"]
#     adata.write("../results/models/piscVI_2.h5ad")

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
#     adata_reactome = run_model(
#         adata_reactome,
#         key=key,
#         masks=masks,   
#     )
#     adata.obsm[f"X_{key}"] = adata_reactome.obsm[f"X_{key}"]
#     adata.write("../results/models/piscVI_2.h5ad")

# Test with more hvgs
for db in ['Reactome']:
    adata_7k = sc.read(f"../data/NBlarge/sn_tumor_cells_NB_7000hvg.h5ad")
    adata_7k.var_names = adata_7k.var['gene_name'].copy()
    genes_per_pathway, genes_per_circuit, circuits_per_pathway = get_pathway_masks(db)
    adata_7k, genes_per_pathway, genes_per_circuit, circuits_per_pathway = filter_genes(adata_7k, genes_per_pathway, genes_per_circuit, circuits_per_pathway)

    if db == 'Reactome':
        masks = [genes_per_pathway]
        masks_list = [masks]
        keys_list = [f"piscVI_7k_{db}"]
    else:
        masks = [genes_per_circuit, circuits_per_pathway]
        masks_pathway = [genes_per_pathway]
        masks_list = [masks, masks_pathway]
        keys_list = [f"piscVI_7k_{db}", f"piscVI_7k_{db}_pathways"]

    for masks, key in zip(masks_list, keys_list):
        print(f"Running model {key} with masks {[mask.shape for mask in masks]}", flush=True)
        adata_7k = run_model(
            adata_7k,
            key=key,
            masks=masks,   
        )
        adata.obsm[f"X_{key}"] = adata_7k.obsm[f"X_{key}"]
        adata.write("../results/models/piscVI_4.h5ad")

# #Combine all metrics

# all_keys = [
#     "scVI_hvg", 
#     "piscVI", "piscVI_pathways", "piscVI_rnd", "piscVI_pathways_rnd",
#     "piscVI_zinb_tanh", "piscVI_nb_relu", "piscVI_nb_tanh",
#     "piscVI_normal_tanh", "piscVI_normal_sigmoid",
#     "piscVI_kegg", "piscVI_kegg_pathways", "piscVI_kegg_rnd", "piscVI_kegg_pathways_rnd",
#     "piscVI_reactome", "piscVI_reactome_rnd",
#     "piscVI_7k_KEGG_NB", "piscVI_7k_KEGG_NB_pathways", "piscVI_7k_KEGG", "piscVI_7k_KEGG_pathways", "piscVI_7k_Reactome"
# ]

# for key in all_keys:
#     metrics = pd.read_csv(f"../results/models/{key}/metrics.csv")
#     metrics["model"] = key
#     if key == all_keys[0]:
#         all_metrics = metrics
#     else:
#         all_metrics = pd.concat([all_metrics, metrics], axis=0)

# #Save all metrics
# all_metrics.to_csv("../results/models/all_metrics.csv", index=False)

# #Benchmark
# all_keys = [f"X_{key}" for key in all_keys]

# for var in ['cell_type', 'cell_state', 'Response']:
#     bm = Benchmarker(
#         adata,
#         batch_key=BATCH_KEY,
#         label_key=var,
#         bio_conservation_metrics=BioConservation(),
#         batch_correction_metrics=BatchCorrection(),
#         embedding_obsm_keys=all_keys,
#         n_jobs=2,
#     )
#     bm.benchmark()

#     bm.plot_results_table(min_max_scale=False)

