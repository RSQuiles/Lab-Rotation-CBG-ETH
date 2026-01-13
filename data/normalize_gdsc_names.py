import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
import re
import os

def norm(x):
    return re.sub(r"[^0-9A-Za-z]", "", str(x)).upper()

def map_variant(x, variant_to_canonical):
    return variant_to_canonical[x]


extracted_path = "/cluster/work/bewi/data/tahoe100/metadata/controls_col_cellCode_with_sensitivity.csv"
print (f"Reading data from {extracted_path}...")
df = pd.read_csv(extracted_path)

print("Preliminary cell name normalization...")
df["_norm"] = df["cell_name"].map(norm)
cell_names = df['_norm'].unique()
cell_lines = df["cell_line"].unique()

print("Mapping cell lines to cell names...")
mapping = {}
for line in cell_lines:
    mask = df["cell_line"] == line
    mapped_names = list(df[mask]["_norm"].unique())
    print(f"{line}: ({len(mapped_names)}, {mapped_names})")
    mapping[line] = mapped_names

print("Creating variant to canonical name mapping...")
variant_to_canonical = {}
for cvcl, names in mapping.items():
    # pick canonical name = shortest
    canonical = min(names, key=len)

    for name in names:
        variant_to_canonical[name] = canonical

print("Applying normalization to cell names...")
df["norm_name"] = df["_norm"].map(lambda x: map_variant(x, variant_to_canonical))

out_path = "/cluster/work/bewi/data/tahoe100/metadata/controls_col_cellCode_with_sensitivity_norm_names.csv"
print(f"Exporting normalized names to {out_path}...")
df.to_csv(out_path, index=False)
