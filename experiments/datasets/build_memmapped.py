import scanpy as sc
import scipy.sparse
import numpy as np
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="Path to the input")

args = parser.parse_args()
data_path = Path(args.path)
print("Loading AnnData...")
adata = sc.read_h5ad(data_path)

# Convert to dense if necessary
X = adata.X
if scipy.sparse.issparse(X):
    print("Converting to dense...")
    X = X.toarray()  # or .A

# Save as a memory-mappable .npy file
print("Saving to .npy file...")
genes_path = data_path.with_name("genes.npy")
np.save(genes_path, X.astype(np.float32))
