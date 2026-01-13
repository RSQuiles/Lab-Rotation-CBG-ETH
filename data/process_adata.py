print("Importing packages...")

from piscvi.model import InformedSCVI
from piscvi.pathway import mask_anndata
# from fcr.validation.predictions import generate_embeddings
print("Imported custom packages successfully.")

import scanpy as sc
import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
import typing
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import json
print("Imported all packages successfully.")

# Define what to process
parser = ArgumentParser()
parser.add_argument("--pcs", action="store_true", help="Whether to precompute PCs")
parser.add_argument("--hvgs", action="store_true", help="Whether to precompute HVGs")
parser.add_argument("--fcr", action="store_true", help="Whether to precompute FCR embeddings")
parser.add_argument("--piscvi", action="store_true", help="Whether to precompute piSCVI embeddings")
parser.add_argument("--data_path", type=str, default="/cluster/work/bewi/data/tahoe100/h5ad/controls_merged_hvg_log1p.h5ad",)
parser.add_argument("--model_path", type=str, help="Path to the pre-trained FCR or piSCVI model")
parser.add_argument("--out", type=str, help="Output path for the processed AnnData object")
args = parser.parse_args()

data_path = args.data_path
_data_path = "/cluster/work/bewi/data/tahoe100/h5ad/controls_mini.h5ad"
print(f"Loading AnnData from {data_path}...")

# Define output path (default: overwrite input file)
if args.out is None:
    output_path = data_path
else:
    output_path = args.out

adata = sc.read_h5ad(data_path)

# Precompute FCR embeddings
if args.fcr:
    if args.model_path is None:
        raise ValueError("Model path must be provided to precompute FCR embeddings.")
    print("Precomputing FCR embeddings...")
    adata = generate_embeddings(args.model_path, adata=adata, return_adata=True)

    print(f"Writing processed AnnData to {output_path}...")
    adata.write(output_path)

# Precompute piSCVI embeddings
if args.piscvi:
    if args.model_path is None:
        raise ValueError("Model path must be provided to precompute piSCVI embeddings.")
    print("Precomputing piSCVI embeddings...")
    # Mask AnnData based on pathways
    adata_copy = adata.copy()
    adata_masked, _ = mask_anndata(adata_copy) # Pass private copy to avoid modifying original adata
    model = InformedSCVI.load(os.path.join(args.model_path, "results/models"), adata=adata_masked)
    latent = model.get_latent_representation()
    adata.obsm["X_piSCVI"] = latent

    print(f"Writing processed AnnData to {output_path}...")
    adata.write(output_path)

# Precompute PCs
if args.pcs:
    print("Precomputing PCs...")
    n_pcs = 100
    sc.tl.pca(adata, n_comps=n_pcs)

    print(f"Writing processed AnnData to {output_path}...")
    adata.write(output_path)

# Precompute HVGs
if args.hvgs:
    n_hvg = 5000
    print("Precomputing HVGs...")
    if "highly_variable" in adata.var.columns and adata.var["highly_variable"].sum() >= n_hvg:
        print("HVGs already computed in adata.var['highly_variable']. Using existing values.")
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3', subset=False)

    print(f"Writing processed AnnData to {output_path}...")
    adata.write(output_path)