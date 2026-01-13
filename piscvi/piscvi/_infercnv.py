
import os
import scanpy as sc
import infercnvpy as cnv
import matplotlib.pyplot as plt
import scanpy.external as sce
import numpy as np
import random
import pandas as pd
import warnings
import harmonypy

# Ensure results/figures directory exists
os.makedirs("results/figures", exist_ok=True)

# Set seeds
seed = 42
np.random.seed(seed)
random.seed(seed)

warnings.simplefilter("ignore")

sc.settings.set_figure_params(figsize=(10, 10))
# Load the data
adata = sc.read("/cluster/work/bewi/members/mazevedo/data/NBAtlas/NBAtlas_raw.h5ad")

# Filtering
adata = adata[:, adata.var_names != 'MALAT1']
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Normalization and scaling
scaled_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
adata.layers["log1p_norm"] = sc.pp.log1p(scaled_counts["X"], copy=True)
adata.X = adata.layers["log1p_norm"]
sc.pp.regress_out(adata, ['nCount_RNA', 'percent_mito'])
sc.pp.scale(adata, max_value=10)

# Initial UMAP and clustering
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, flavor='igraph', n_iterations=2)

# Save initial UMAP plots
for feature in ["Cell_type", "INSS_stage", "Study", "Sample"]:
    sc.pl.umap(adata, color=feature, show=False, save=f"_initial_{feature}.png")

# Batch effect removal
sce.pp.harmony_integrate(adata, 'Sample')
adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, flavor='igraph', n_iterations=2)

# Save post-Harmony UMAP plots
for feature in ["Study", "Sample", "Cell_type"]:
    sc.pl.umap(adata, color=feature, show=False, save=f"_harmony_{feature}.png")

# InferCNV annotations
gtf_file = "./data/gencode_anot/gencode.v43.annotation.gtf"
cnv.io.genomic_position_from_gtf(gtf_file, adata, gtf_gene_id="gene_name")

# CNV inference
cnv.tl.infercnv(
    adata,
    reference_key="Cell_type",
    reference_cat=["B cell", "Myeloid", "NK cell", "Plasma", "T cell", "RBCs", "pDC", "Endothelial"],
    window_size=250,
    calculate_gene_values=True,
)
cnv.pl.chromosome_heatmap(adata, groupby="Cell_type", show=False, save="_chromosome_heatmap_celltype.png")

# CNV clustering
cnv.tl.pca(adata)
cnv.pp.neighbors(adata)
cnv.tl.leiden(adata)
cnv.pl.chromosome_heatmap(adata, groupby="cnv_leiden", dendrogram=True, show=False, save="_chromosome_heatmap_leiden.png")

# CNV UMAP visualization
cnv.tl.umap(adata)
cnv.tl.cnv_score(adata)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
ax4.axis("off")
cnv.pl.umap(adata, color="cnv_leiden", legend_loc="on data", ax=ax1, show=False)
cnv.pl.umap(adata, color="cnv_score", ax=ax2, show=False)
cnv.pl.umap(adata, color="Cell_type", ax=ax3, show=False)
plt.savefig("results/figures/cnv_umap_combined.png")
plt.close(fig)

# Assign CNV status
# Compute mean CNV score per cnv_leiden cluster
cnv_leiden_means = adata.obs.groupby('cnv_leiden')['cnv_score'].mean()

# Define threshold: Clusters with above-median CNV scores are considered malignant
threshold = 0.0175 # This can change and should be evaluated by looking at the CNV scores obtained
                  # do it interactiviley at the begining to see the scores and the cluster in a Jupyter notebook

malignant_clusters = cnv_leiden_means[cnv_leiden_means > threshold].index.tolist() # In this case we are averaging the cnv scores per cluster
# Assign malignant labels based on cnv_leiden
adata.obs['cnv_status'] = np.where(adata.obs['cnv_leiden'].isin(malignant_clusters), 'tumor', 'normal')

# Print summary
print(f"Identified {len(malignant_clusters)} malignant clusters: {malignant_clusters}")
adata.obs['cnv_status'].value_counts()

# CNV heatmaps for tumor and normal cells
cnv.pl.chromosome_heatmap(adata[adata.obs["cnv_status"] == "tumor", :], show=False, save="_cnv_tumor.png")
cnv.pl.chromosome_heatmap(adata[adata.obs["cnv_status"] == "normal", :], show=False, save="_cnv_normal.png")

# Save processed data
adata.write("./data/NBAtlas/NBAtlas_log1p_harmony_umap_nn_leiden_cnv_endothelial_ref.h5ad", compression="gzip")
