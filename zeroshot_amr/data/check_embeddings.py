print("Importing packages...")
import scanpy as sc
import pandas as pd
import numpy as np
import re
import os
import glob
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def knn_indices(X, k=15):
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    return nn.kneighbors(return_distance=False)

def knn_overlap(idx1, idx2):
    """
    Average Jaccard similarity between neighbor sets
    """
    overlaps = []
    for a, b in zip(idx1, idx2):
        overlaps.append(len(set(a) & set(b)) / len(set(a) | set(b)))
    return np.mean(overlaps)

base_dir = "/cluster/work/bewi/members/rquiles/zeroshot_amr/data"
out_path = os.path.join(base_dir, "embeddings_knn_comparisons.csv")

print("Loading embeddings...")
files = sorted(Path(base_dir).glob("raw_data*.npy"))
embeddings = {}
for file in files:
    name = "_".join(str(file).split("/")[-1].split(".")[0].split("_")[-2:])
    embeddings[name] = np.load(file)
    print(f"  {name}: {embeddings[name].shape}")

# ----------------------------
# Sanity check: same number of samples
# ----------------------------
n_samples = {X.shape[0] for X in embeddings.values()}
if len(n_samples) != 1:
    raise ValueError("Embeddings do not all have the same number of samples!")

N = n_samples.pop()
print(f"\nTotal samples: {N}")

# ----------------------------
# Subsample consistently
# ----------------------------
subset_size = 30_000
random_seed = 42

if subset_size < N:
    print(f"Subsampling to {subset_size} samples...")
    rng = np.random.default_rng(random_seed)
    subset_idx = rng.choice(N, size=subset_size, replace=False)

    for name in embeddings:
        embeddings[name] = embeddings[name][subset_idx]
else:
    print("No subsampling applied (subset_size >= N)")

knn_idx = {}
print("\nComputing KNN indices for:")
for name, X in embeddings.items():
    print(name)
    knn_idx[name] = knn_indices(X)

knn_results = {}
print("\nComparing:")
for name1 in embeddings.keys():
    idx1 = knn_idx[name1]

    for name2 in embeddings.keys():
        print(f"{name1} vs {name2}")
        idx2 = knn_idx[name2]
        knn_results[(name1, name2)] = knn_overlap(idx1, idx2)

# Output results
print("\nOutput KNN Overlap Results:")
names = list(embeddings.keys())

knn_df = pd.DataFrame(
    index=names,
    columns=names,
    data=[[knn_results[(i, j)] for j in names] for i in names]
)

print("\nkNN overlap matrix:")
print(knn_df.round(3))

# Save results
knn_df.to_csv(out_path)
print(f"Results saved to {out_path}")