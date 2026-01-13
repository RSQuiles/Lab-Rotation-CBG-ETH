print("Importing packages...")
from fcr.validation.predictions import generate_embeddings

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

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the pre-trained FCR model")
parser.add_argument("--adata_path", type=str, help="Path to the input AnnData object")
parser.add_argument("--processed_adata_path", type=str, help="Path to the logtransformed AnnData")
parser.add_argument("--output_path", type=str, help="Path to save the AnnData object with FCR embeddings")
args = parser.parse_args()

print(f"Loading AnnData from {args.adata_path}...")
adata = sc.read_h5ad(args.adata_path)

print("Generating corresponding logtransformed AnnData...")
processed_adata = sc.read_h5ad(args.processed_adata_path)
sample_ids = processed_adata.obs_names
use_adata = processed_adata[sample_ids].copy()

print("Precomputing FCR embeddings...")
latents = generate_embeddings(args.model_path, adata=use_adata, return_adata=False)

# Align latents with adata
print("Aligning FCR embeddings with original AnnData...")
adata.obsm["ZXs"] = latents["ZX"]
adata.obsm["ZXTs"] = latents["ZXT"]
adata.obsm["ZTs"] = latents["ZT"]

# Save updated AnnData
print(f"Writing processed AnnData with FCR embeddings to {args.output_path}...")
adata.write(args.output_path)

