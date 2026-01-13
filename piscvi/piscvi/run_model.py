import time
print("Starting script at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), flush=True)
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
#from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

sys.path.append(os.path.abspath("../src"))
from scib_core_gmm import Benchmarker, BioConservation, BatchCorrection
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

BATCH_KEY = 'replicate'
CELL_TYPE_KEY = 'cell_type'
KEY_OF_INTEREST = 'condition'


"""Script to run the InformedSCVI model on the given AnnData object with specified parameters."""


def run_model(adata, 
              activation="relu",
              masks=None,
              likelihood="zinb",
              #layer="X", 
              key="scVI"):
    """
    Train the model if it does not already exist, otherwise load, cluster, and compute metrics.
    """
    #Check if model already exists
    if not os.path.exists(f"../results/models/{key}/model.pt"):
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
        train_indices, val_indices = split_train_val(adata, KEY_OF_INTEREST)
        early_stopping_kwargs = {
            "early_stopping": True,
            "early_stopping_patience": 30,
            #"early_stopping_monitor": "validation_loss",
        }
        datasplitter_kwargs = {
            "external_indexing": [train_indices, val_indices],
        }
        training_plan_kwargs = {
            "reduce_lr_on_plateau": True,
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
        elbo = model.get_elbo().item()
        reconstruction_error = model.get_reconstruction_error()['reconstruction_loss'].item()
        os.makedirs(f"../results/models/{key}", exist_ok=True)
        with open(f"../results/models/{key}/train_metrics.txt", "w") as f:
            f.write(f"Training time: {train_time}\n")
            f.write(f"Reconstruction error: {reconstruction_error}\n")
            f.write(f"ELBO: {elbo}\n")

        # Save latent representation
        latent = model.get_latent_representation()
        adata.obsm[f"X_{key}"] = latent

        plot_loss(model, save_path=f"../results/figures/{key}/loss.png")
        os.makedirs(f"../results/models/{key}", exist_ok=True)

        # Save normalized expression
        #adata_subset = adata[adata.obs[CELL_TYPE_KEY] == "neuroblast (sensu Vertebrata)"]
        # denoised = model.get_normalized_expression(adata_subset, 
        #                                            transform_batch=None, 
        #                                            log_transform=True,
        #                                            n_samples=1,
        #                                            batch_size=1024, 
        # )
        # denoised = pd.DataFrame(
        #     denoised,
        #     index=adata_subset.obs_names,
        #     columns=adata_subset.var_names,
        # )
        # os.makedirs(f"../results/models/{key}", exist_ok=True)
        # denoised.to_csv(f"../results/models/{key}/denoised_data.csv")

    else:
        print(f"Model {key} already exists, proceeding to compute metrics.", flush=True)
        model = InformedSCVI.load(f"../results/models/{key}", adata=adata)
        
        # Save latent representation
        latent = model.get_latent_representation()
        adata.obsm[f"X_{key}"] = latent

        plot_loss(model, save_path=f"../results/figures/{key}/loss.png")

    # Clustering
    try:
        cluster_and_plot(adata, key)
    except:
        print(f"Clustering failed for {key}, continuing without clustering.", flush=True)
    
    try:
        compute_metrics(adata, key)
    except:
        print(f"Metrics computation failed for {key}, continuing without metrics.", flush=True)   
    
    return adata

def cluster_and_plot(adata, key):
    """
    Perform clustering and plotting with Leiden and GMM.
    """
    print(f"Clustering and plotting for {key}", flush=True)

    if "X_pca" not in adata.obsm.keys():
        sc.pp.pca(adata)

    if f"neighbors_{key}" not in adata.uns.keys():
        print(f"Computing neighbors for {key}", flush=True)
        sc.pp.neighbors(adata, use_rep=f"X_{key}", key_added=f"neighbors_{key}", n_neighbors=20)
    if f"umap_{key}" not in adata.obsm.keys():
        print(f"Computing UMAP for {key}", flush=True)
        sc.tl.umap(adata, key_added=f"umap_{key}", neighbors_key=f"neighbors_{key}", min_dist=0.6)
    if f"leiden_{key}" not in adata.obs.keys():
        print(f"Computing Leiden clustering for {key}", flush=True)
        sc.tl.leiden(adata, key_added=f"leiden_{key}", resolution=0.3, neighbors_key=f"neighbors_{key}")

    if f"gmm_{key}" not in adata.obs.keys():
        # Get number of categories of that variable
        n_categories = adata.obs[KEY_OF_INTEREST].nunique()
        gmm = GaussianMixture(n_components=n_categories, random_state=42)
        gmm.fit(adata.obsm[f"X_{key}"])
        adata.obs[f"gmm_{key}"] = gmm.predict(adata.obsm[f"X_{key}"])
            
    sc.settings.figdir = f"../results/figures/{key}/"
    for feature in [f"leiden_{key}", f"gmm_{key}", CELL_TYPE_KEY, KEY_OF_INTEREST, BATCH_KEY]:
        sc.pl.embedding(adata, basis=f"umap_{key}", color=feature, title=feature, frameon=False, show=False, save=f"_{feature}.png")

    print('Finished generating figures')

def compute_metrics(adata, key):
    """
    Compute clustering metrics and save them with the training metrics.
    """
    print("Computing metrics")

    # Retrieve training metrics
    with open(f"../results/models/{key}/train_metrics.txt", "r") as f:
        train_time = f.readline().strip().split(": ")[1]
        reconstruction_error = float(f.readline().strip().split(": ")[1])
        elbo = float(f.readline().strip().split(": ")[1])

    # Calculate clustering metrics
    print(f"Computing silhouette scores of leiden for {key}", flush=True)
    silhouette = silhouette_score(adata.obsm[f"X_{key}"], adata.obs[f"leiden_{key}"], metric="euclidean")
    print(f"Computing silhouette scores of gmm for {key}", flush=True)
    silhouette_gmm = silhouette_score(adata.obsm[f"X_{key}"], adata.obs[f"gmm_{key}"], metric="euclidean")
    metrics = pd.DataFrame(
        {
            "elbo": [elbo],
            "reconstruction_error": [reconstruction_error],
            "silhouette leiden": [silhouette],
            "silhouette gmm": [silhouette_gmm],
            "train_time": [train_time],
        }   
    )
    for var in [CELL_TYPE_KEY, KEY_OF_INTEREST]: 
        print(f"Computing metrics for {var} for {key}", flush=True)
        # # Check if there are any null values in the variable of interest
        # if adata.obs[var].isnull().any():
        #     valid_idx = adata.obs['Response'].notnull()
        #     adata = adata[valid_idx].copy() 
        silhouette_var = silhouette_score(adata.obsm[f"X_{key}"], adata.obs[var], metric="euclidean")
        ARI = adjusted_rand_score(adata.obs[var], adata.obs[f"leiden_{key}"])
        NMI = normalized_mutual_info_score(adata.obs[var], adata.obs[f"leiden_{key}"])
        homogeneity = homogeneity_score(adata.obs[var], adata.obs[f"leiden_{key}"])
        ARI_gmm = adjusted_rand_score(adata.obs[var], adata.obs[f"gmm_{key}"])
        NMI_gmm = normalized_mutual_info_score(adata.obs[var], adata.obs[f"gmm_{key}"])
        homogeneity_gmm = homogeneity_score(adata.obs[var], adata.obs[f"gmm_{key}"])
        metrics_var = pd.DataFrame(
            {
                f"silhouette {var}": [silhouette_var],
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


adata = sc.read("../data/kang/kang_processed.h5ad")

db = "KEGG_old"

# adata_saved = sc.read("../results/models/piscVI.h5ad")

# adata.obs = adata_saved.obs.copy()
# adata.obsm = adata_saved.obsm.copy()
# adata.uns = adata_saved.uns.copy()
# adata.obsp = adata_saved.obsp.copy()


# del adata_saved

#Initial plots and save
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=10)
sc.tl.umap(adata, min_dist=0.2)
sc.tl.leiden(adata)
sc.settings.figdir = "../results/figures/raw_val/"
for feature in ["leiden", CELL_TYPE_KEY, KEY_OF_INTEREST, BATCH_KEY]:
    sc.pl.umap(adata, color=feature, title=feature, frameon=False, show=False, save=f"_initial_{feature}.png")

adata = run_model(
    adata,
    key="scVI_val",
)
# adata = run_model(
#     adata,
#     key="scVI_nb",
#     likelihood="nb",
# )
adata.write("../results/models/piscVI_val.h5ad")

genes_per_pathway, genes_per_circuit, circuits_per_pathway = get_pathway_masks(db)
adata, genes_per_pathway, genes_per_circuit, circuits_per_pathway = filter_genes(adata, genes_per_pathway, genes_per_circuit, circuits_per_pathway)
frac = genes_per_circuit.sum(axis=1).mean() / genes_per_circuit.shape[1]
print(f"Fraction of genes per circuit: {frac:.2f}", flush=True)
print(f"Number of genes: {genes_per_circuit.shape[1]}", flush=True)
rnd_genes_per_pathway, rnd_genes_per_circuit, rnd_circuits_per_pathway = get_random_masks(adata.var_names, genes_per_circuit.shape[0], genes_per_pathway.shape[0], frac=frac, seed=42)

masks_kegg = [genes_per_circuit, circuits_per_pathway]
masks_kegg_rnd = [rnd_genes_per_circuit, rnd_circuits_per_pathway]
masks_kegg_pathways = [genes_per_pathway]
#masks_kegg_pathways_rnd = [rnd_genes_per_pathway]

adata = adata.copy()

masks_list = [masks_kegg, masks_kegg_pathways, masks_kegg_rnd]
keys_list = ["piscVI_val", "piscVI_val_pathways", "piscVI_val_rnd"]

for masks, key in zip(masks_list, keys_list):
    print(f"Running model {key} with masks {[mask.shape for mask in masks]}", flush=True)
    adata = run_model(
        adata,
        key=key,
        masks=masks,
    )
    adata.write("../results/models/piscVI_val.h5ad")