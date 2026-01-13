import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

import scanpy as sc
import numpy as np
from scipy import sparse
import random
import numpy as np
import scipy.sparse as sp
import pandas as pd
import anndata as ad
import json
import argparse
import os

# Select plate to prepare
parser = argparse.ArgumentParser()
parser.add_argument("--plate", type=str, help="Number of the Tahoe100 plate to prepare")
args = parser.parse_args()

plate_number = args.plate
plate_path = f"/cluster/work/bewi/data/tahoe100/h5ad/plate{plate_number}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"

# Load plate data
print(f"Loading plate {plate_number} from {plate_path}...")
adata = sc.read_h5ad(plate_path)
print(f"Loaded plate with {adata.n_obs} samples and {adata.n_vars} genes")

# Filtering the plate
change_col_names = {"drug": "Agg_Treatment", "cell_line": "covariates"}
control_name = "DMSO_TF"

adata.obs = adata.obs.rename(columns=change_col_names)
adata.obs["control"] = adata.obs[change_col_names["drug"]] == control_name
adata.obs["control"] = adata.obs["control"].astype(int)
adata.uns["fields"] = []
adata.obs["dose"] = adata.obs["drugname_drugconc"].str.split(",").str[1].astype(float)

# Save prepared plate
output_path = f"/cluster/work/bewi/members/rquiles/data/plate{plate_number}_tahoe100M_prepared.h5ad"
adata.write_h5ad(output_path)