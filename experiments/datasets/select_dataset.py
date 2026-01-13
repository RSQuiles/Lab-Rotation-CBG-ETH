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

# Parse cell_lines and drugs from json file
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="Name of dataset_config.json file")
parser.add_argument("--n_genes", type=int, default=5000, help="Number of highly variable genes to select")
parser.add_argument("--balanced", action="store_true", help="Whether to balance the dataset across cell lines")
parser.add_argument("--export", type=str, default=None, help="Path to export the selected dataset")
parser.add_argument("--ratio", type=int, default=3, help="Ratio of controls to treated samples per cell line")
args = parser.parse_args()

n_genes = args.n_genes
export_base_dir = args.export
control_ratio = args.ratio

# Read dataset configuration
config_name = args.file
config_path = os.path.join("/cluster/work/bewi/members/rquiles/experiments/datasets", config_name)
with open(config_path, "r") as f:
    arguments = json.load(f)

# Read arguments
lines_select = arguments["cell_lines"]
control_name = "DMSO_TF"
drugs_select = arguments.get("drugs", [])
drugs_select.append(control_name)
balanced = args.balanced

export_name = arguments["name"]

normalize_total = arguments.get("normalize_total", True)
log_transform = arguments.get("log_transform", True)
scale_data = arguments.get("scale", False)

# Use metadata to filter the plates
data_dir = "/cluster/work/bewi/data/tahoe100/h5ad/"
metadata_path = "/cluster/work/bewi/members/rquiles/experiments/datasets/obs_metadata.parquet"

print("Inspecting metadata...")
metadata = pd.read_parquet(metadata_path)

select = metadata[(metadata["cell_name"].isin(lines_select)) & (metadata["drug"].isin(drugs_select))]
print(f"Selected {select.shape[0]} samples from {len(lines_select)} cell lines and {len(drugs_select)-1} drugs (with controls)")

## SUBSET TO BALANCE THE NUMBER OF CONTROLS
keep_rows = []

for cell_line in lines_select:
    # Treated (non-control) rows for this cell line
    treated_mask = (
        (select["cell_name"] == cell_line) & (select["drug"] != control_name)
    )
    treated_idx = select.index[treated_mask]
    n_treated = len(treated_idx)
    if n_treated > 0:
        keep_rows.extend(treated_idx)

    # Controls: keep only a fraction of treated count
    print(f"Using control ratio: {control_ratio}")
    if n_treated > 0:
        n_controls = n_treated // control_ratio
    else:
        n_controls = len(select)

    control_mask = (
        (select["cell_name"] == cell_line) & (select["drug"] == control_name)
    )
    ctrl_idx = select.index[control_mask][:n_controls]
    cell_line_ratio = len(ctrl_idx)/len(treated_idx) if len(treated_idx) > 0 else 1/control_ratio
    print(f"Control ratio for cell line {cell_line}: {cell_line_ratio}")
    if len(ctrl_idx) > 0:
        keep_rows.extend(ctrl_idx)

# Randomize row order and subset
random.shuffle(keep_rows)
select = select.loc[keep_rows]

if balanced:
    ## SUBSET TO BALANCE THE NUMBER OF SAMPLES PER CELL LINE
    n_samples = []
    for cell_line in lines_select:
        mask = select["cell_name"] == cell_line
        row_indexes = select[mask].index
        n_samples.append(len(row_indexes))

    # Groups can be at most 1.5 the size of the smallest group
    min_samples = round(min(n_samples) * 1.5)
    print(f"N_samples per cell line: {n_samples}")
    print(f"Minimum samples per cell line: {min_samples}")

    keep_rows = []
    # Sample according to the smallest group
    for cell_line in lines_select:
        mask = select["cell_name"] == cell_line
        row_indexes = select[mask].index
        row_indexes = row_indexes[:min_samples]
        if len(row_indexes) > 0:
            keep_rows.extend(row_indexes)
            print(f"Length of {cell_line}: {len(row_indexes)}")
        
    # Randomize row order and subset by row indices (DataFrame)
    random.shuffle(keep_rows)
    select = select.loc[keep_rows]

# Determine which plates to inspect
plates = np.unique(select["plate"])

# Loop through plates and append the matching rows
final_adata = None

for i, plate in enumerate(plates):
    print(f"Loading {plate}...")
    data_path = f"/cluster/work/bewi/data/tahoe100/h5ad/{plate}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
    adata = sc.read_h5ad(data_path, backed="r")
    print(f"Subsetting {plate}...")
    
    ids = select[select["plate"] == plate]["BARCODE_SUB_LIB_ID"].values
    subset = adata[adata.obs_names.isin(ids), :].to_memory()
    # Initialize or append final AnnData
    if final_adata is None:
        final_adata = subset
    else:
        final_adata = ad.concat([final_adata, subset], join="outer")

# Ensure we found at least some cells across selected plates
if final_adata is None or final_adata.n_obs == 0:
    raise RuntimeError(
        "No matching cells were found for the requested cell lines/drugs across the selected plates. "
        "Check that BARCODE_SUB_LIB_ID values exist in the plate .h5ad files and that filters are not too strict."
    )

# Randomize the rows in final_adata
n = final_adata.n_obs  # number of cells
perm = np.random.permutation(n)
adata = final_adata[perm, :]

# PREPROCESSING of the AnnData to match the FCR method code
# Build a consistent export path with .h5ad extension so overwriting works as expected
if export_base_dir is None:
    export_base_dir = "/cluster/work/bewi/members/rquiles/experiments/datasets"
    
export_name = arguments["name"]
if not export_name.endswith(".h5ad"):
    export_name = f"{export_name}.h5ad"
export_path = os.path.join(export_base_dir, export_name)

change_col_names = {"drug": "Agg_Treatment"}

## UPDATE COLUMNS NEW DATASET
adata.obs = adata.obs.rename(columns=change_col_names)
adata.obs["control"] = adata.obs[change_col_names["drug"]] == control_name
adata.obs["control"] = adata.obs["control"].astype(int)
adata.uns["fields"] = []
adata.obs["dose"] = adata.obs["drugname_drugconc"].str.split(",").str[1].astype(float)

## PREPROCESS AND EXPORT
print("Preprocessing the selected dataset...")
if normalize_total:
    print("Normalizing total counts per cell...")
    sc.pp.normalize_total(adata, target_sum=1e6)
if log_transform:
    print("Log-transforming the data...")
    sc.pp.log1p(adata)
    print("Subsetting highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, subset=True)
else:
    print("Subsetting highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, flavor='seurat_v3', subset=True)
    
if scale_data:
    print("Scaling the data...")
    sc.pp.scale(adata, max_value=10)

print(f"Exporting to {export_path}...")
adata.write(export_path)
print("Exported selected dataset successfully.")