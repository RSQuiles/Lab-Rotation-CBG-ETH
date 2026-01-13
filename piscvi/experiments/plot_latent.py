import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import sys
import os
sys.path.append(os.path.abspath("../src"))
import argparse
import json
import matplotlib.colors as mcolors

from model import InformedSCVI
from pathway import get_pathway_masks
from pathway import filter_genes

def visualize_piscvi_results(
    model_path,
    adata,
    drug_key="Agg_Treatment",  # Column with drug names
    dose_key="dose",           # Column with drug dosages
    cell_type_key="cell_name", # Column with cell types
    n_top_pathways=10,         # Number of top pathways to show
    layers_to_plot=[0, 1],     # Which hidden layers to analyze
    output_dir="./figures/"
):
    """
    Comprehensive visualization of piSCVI results including:
    - Latent space UMAPs
    - Drug dosage effects
    - Pathway activation patterns
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = InformedSCVI.load(model_path, adata=adata)
    
    # 1. Get latent representation
    print("Computing latent representation...")
    latent = model.get_latent_representation()
    adata.obsm["X_piscvi"] = latent
    
    # 2. Get hidden layer activations (pathway activations)
    print("Extracting pathway activations...")
    activations = model.get_hidden_activations(
        adata=adata,
        layers_to_capture=layers_to_plot
    )
    
    # Store activations in adata
    for layer_name, act in activations.items():
        adata.obsm[f"X_{layer_name}"] = act
    
    # 3. Compute UMAP on latent space
    print("Computing UMAP...")
    sc.pp.neighbors(adata, use_rep="X_piscvi", n_neighbors=15)
    sc.tl.umap(adata, min_dist=0.3)
    
    # ===== VISUALIZATION 1: Basic UMAP with drug and cell type =====
    print("Creating basic UMAPs...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Cell types
    sc.pl.umap(adata, color=cell_type_key, ax=axes[0], show=False, 
               title="Cell Types", frameon=False)
    
    # Drug treatments
    sc.pl.umap(adata, color=drug_key, ax=axes[1], show=False,
               title="Drug Treatment", frameon=False, legend_loc='right margin')
    
    # Drug dosages (continuous)
    if dose_key in adata.obs.columns:
        sc.pl.umap(adata, color=dose_key, ax=axes[2], show=False,
                   title="Drug Dosage", frameon=False, cmap='viridis')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_overview.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ===== REUSABLE Data =====
    drugs = list(adata.obs[drug_key].unique())
    drugs.remove("DMSO_TF")
    n_drugs = len(drugs)

    dose_values = adata.obs[dose_key].unique()

    # Load pathway names
    try:
        genes_per_pathway, genes_per_circuit, circuits_per_pathway = get_pathway_masks(db="KEGG")
        pathway_names = genes_per_pathway.index.values
        print(f"Pathway names: {pathway_names[:5]}...")
        circuit_names = genes_per_circuit.index.values if genes_per_circuit is not None else None
        print(f"Circuit names: {circuit_names[:5]}...")
    except:
        print("Warning: Could not load pathway names, using indices")
        pathway_names = None
        circuit_names = None

    # Pathways with greatest change between control and treated
    print("Computing DE pathways for each drug...")
    de_pathways = {}

    for layer_idx in layers_to_plot:
        layer_name = f"layer_{layer_idx}"
        if layer_name not in activations:
            continue
        
        act = activations[layer_name]  # Shape: (n_cells, n_pathways_in_layer)
        de_pathways[layer_name] = {}
        
        for drug in drugs:
            control_mask = adata.obs[drug_key] == "DMSO_TF"
            treated_mask = adata.obs[drug_key] == drug
            
            # Compute mean activation for control and treated samples
            control_activation = act[control_mask].mean(axis=0)
            treated_activation = act[treated_mask].mean(axis=0)
            
            # Compute absolute difference (change magnitude)
            activation_change = np.abs(treated_activation - control_activation)
            
            # Select top pathways by change magnitude
            top_indices = np.argsort(activation_change)[-n_top_pathways:][::-1]
            
            # Get pathway names
            if layer_idx == 0 and circuit_names is not None:
                names = [circuit_names[i] if i < len(circuit_names) else f"Circuit_{i}" 
                        for i in top_indices]
            elif layer_idx == 1 and pathway_names is not None:
                names = [pathway_names[i] if i < len(pathway_names) else f"Pathway_{i}"
                        for i in top_indices]
            else:
                names = [f"Feature_{i}" for i in top_indices]
            
            # Update dictionary
            de_pathways[layer_name][drug] = {
                "indices": top_indices,
                "names": names,
                "changes": activation_change[top_indices]
            }
    
    # ===== VISUALIZATION 2: Dosage response per drug =====
    print("Creating drug-specific dosage plots...")
    if drug_key in adata.obs.columns and dose_key in adata.obs.columns: 
        fig, axes = plt.subplots(1, n_drugs, figsize=(6*n_drugs, 5))
        if n_drugs == 1:
            axes = [axes]
        
        for idx, drug in enumerate(drugs):
            mask = adata.obs[drug_key] == drug
            adata_subset = adata[mask].copy()

            norm = mcolors.SymLogNorm(linthresh=0.01, vmin=0, vmax=5)
            sc.pl.umap(adata_subset, color=dose_key, ax=axes[idx], show=False,
                       title=f"{drug} - Dosage", cmap='YlOrRd', frameon=False,
                       norm=norm)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/umap_per_drug_dosage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # ===== VISUALIZATION 3: Pathway activation patterns =====
    print("Analyzing pathway activations...")
    
    for layer_idx in layers_to_plot:

        act = activations[f"layer_{layer_idx}"]

        for drug in drugs:
            # Plot heatmap: cells x top pathways
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Sort cells by dosage for better visualization
            if drug_key in adata.obs.columns:
                mask = adata.obs[drug_key] == drug
            else:
                mask = np.ones(len(adata), dtype=bool)

            # Optionally, sort only within this drug by dose
            if dose_key in adata.obs.columns:
                sort_idx = np.argsort(adata.obs.loc[mask, dose_key].values)
            else:
                sort_idx = np.arange(mask.sum())

            # Apply both mask and sorting
            top_indices = de_pathways[f"layer_{layer_idx}"][drug]["indices"] 
            names = de_pathways[f"layer_{layer_idx}"][drug]["names"]
            act_subset = act[mask][sort_idx][:, top_indices]
            
            sns.heatmap(act_subset.T, cmap='viridis', ax=ax, 
                    yticklabels=names, xticklabels=False,
                    cbar_kws={'label': 'Activation'})
            ax.set_xlabel('Cells (sorted by dose)')
            ax.set_ylabel('Pathway/Circuit')
            ax.set_title(f'Top Activated Pathways/Circuits - Layer {layer_idx} - Drug {drug}')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/pathway_heatmap_layer{layer_idx}_{drug}.png", 
                    dpi=300, bbox_inches='tight')
            plt.close()
        
            # ===== VISUALIZATION 4: UMAP colored by top pathway activations =====
            n_show = min(6, n_top_pathways)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i in range(n_show):
                pathway_idx = top_indices[i]
                adata.obs[f"pathway_{pathway_idx}"] = act[:, pathway_idx]
                
                sc.pl.umap(adata, color=f"pathway_{pathway_idx}", 
                        ax=axes[i], show=False,
                        title=f"{names[i][:40]}", 
                        cmap='Reds', frameon=False)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/umap_pathway_activations_layer{layer_idx}_{drug}.png",
                    dpi=300, bbox_inches='tight')
            plt.close()
    
            # ===== VISUALIZATION 5: Drug response per pathway =====
            print("Creating drug-pathway response plots...")

            # Prepare plots            
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            for i in range(min(10, len(top_indices))):
                pathway_idx = top_indices[i]
                
                # Create scatter plot: dose vs activation, colored by drug
                mask = adata.obs[drug_key] == drug
                x = adata.obs.loc[mask, dose_key].values
                y = act[mask][:, pathway_idx]
                axes[i].scatter(x, y, alpha=0.5, label=drug, s=20)
                
                axes[i].set_xlabel('Dosage')
                axes[i].set_ylabel('Activation')
                pathway_name = names[i][:30] if i < len(names) else f"Feature_{pathway_idx}"
                axes[i].set_title(pathway_name, fontsize=9)
                axes[i].legend(fontsize=6)
                
            plt.suptitle(f'Pathway Activation vs Drug Dosage - Layer {layer_idx} - {drug}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/dose_response_layer{layer_idx}_{drug}.png",
                    dpi=300, bbox_inches='tight')
            plt.close()
    
        """
        # ===== VISUALIZATION 6: Correlation between pathways and drugs =====
        print("Computing pathway-drug correlations...")
    
        layer_name = f"layer_{layer_idx}"
        if layer_name not in activations:
            continue
        
        act = activations[layer_name]
        
        # Compute mean activation per drug
        drug_pathway_matrix = np.zeros((len(drugs), act.shape[1]))
        
        for i, drug in enumerate(drugs):
            mask = adata.obs[drug_key] == drug
            drug_pathway_matrix[i] = act[mask].mean(axis=0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(15, max(6, len(drugs)*0.5)))
        sns.heatmap(drug_pathway_matrix, cmap='RdBu_r', center=0,
                    yticklabels=drugs, xticklabels=False,
                    cbar_kws={'label': 'Mean Activation'})
        ax.set_xlabel('Pathways/Circuits')
        ax.set_ylabel('Drug Treatment')
        ax.set_title(f'Drug-Pathway Association - Layer {layer_idx}')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/drug_pathway_heatmap_layer{layer_idx}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        """
    
    print(f"All visualizations saved to {output_dir}")
    return adata, activations

# ===== USAGE =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize PISCVI Results")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = json.load(open(args.config, 'r'))
    base_dir = config["name"]
    model_path = os.path.join(base_dir, "results", "models")
    output_dir=os.path.join(base_dir, "results", "figures")
    drug_key="Agg_Treatment"
    dose_key="dose"
    cell_type_key= config.get("cell_key", "cell_name")

    # Load your data
    adata = sc.read_h5ad(config["data_path"])
    # Need to use filtered version
    genes_per_pathway, genes_per_circuit, circuits_per_pathway = get_pathway_masks(db="KEGG")
    adata, genes_per_pathway, genes_per_circuit, circuits_per_pathway = filter_genes(adata, genes_per_pathway, genes_per_circuit, circuits_per_pathway)
    
    # Run visualization
    adata, activations = visualize_piscvi_results(
        model_path=model_path,
        adata=adata,
        drug_key=drug_key,
        dose_key=dose_key,
        cell_type_key=cell_type_key,
        n_top_pathways=10,
        layers_to_plot=[0, 1],  # 0=circuits, 1=pathways
        output_dir=output_dir
    )