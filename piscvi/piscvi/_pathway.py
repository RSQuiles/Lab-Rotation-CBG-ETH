import numpy as np
import random
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("."))
from utils import get_resource_path
from itertools import chain, repeat
from pathlib import Path

def get_pathway_masks(db="KEGG"):
    print("Current directory:", os.getcwd())
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCES = os.path.join(BASE_DIR, "resources")

    if db == "KEGG" or db == "KEGG_NB":
        if db == "KEGG":
            genes_per_pathway = pd.read_csv(
            os.path.join(RESOURCES, "genes_per_pathway.csv"), index_col=0)
            genes_per_circuit = pd.read_csv(
            os.path.join(RESOURCES, "genes_per_circuit_backward.csv"), index_col=0)
        if db == "KEGG_NB":
            genes_per_pathway = pd.read_csv(
                os.path.join(RESOURCES, "genes_per_pathway_NB.csv"), index_col=0)
            genes_per_circuit = pd.read_csv(
                os.path.join(RESOURCES, "genes_per_circuit_backward_NB.csv"), index_col=0)
        circuits_per_pathway = pd.DataFrame(
            index=genes_per_pathway.index.values, columns=genes_per_circuit.index.values, data=0
        )
        for i, circuit_name in enumerate(genes_per_circuit.index.values):
            for j, pathway_name in enumerate(genes_per_pathway.index.values):
                #Circuit names start with pathway names
                if circuit_name.startswith(pathway_name):
                    circuits_per_pathway.iloc[j, i] = 1
        return genes_per_pathway, genes_per_circuit, circuits_per_pathway

    elif db == "KEGG_old":
        genes_per_pathway, genes_per_circuit, circuits_per_pathway = get_adj_matrices()
        return genes_per_pathway, genes_per_circuit, circuits_per_pathway

    elif db == "Reactome":
        genes_per_pathway = get_reactome_adj()
        return genes_per_pathway, None, None

    else:
        raise ValueError("Database not recognized. Use 'KEGG', 'KEGG_NB', 'KEGG_old' or 'Reactome'.")


def get_random_masks(genes_list, n_circuits, n_pathways, frac=0.5, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    rnd_pathway_names = [f"rnd{i}" for i in range(n_pathways)]
    rnd_genes_per_pathway = pd.DataFrame(
        np.random.binomial(1, frac, (n_pathways, len(genes_list))),
        columns=genes_list,
        index=rnd_pathway_names,
    )

    rnd_circuit_names = [
        f"{rnd_pathway_names[random.randint(0, n_pathways - 1)]}_{i}"
        for i in range(n_circuits)
    ]
        
    rnd_genes_per_circuit = pd.DataFrame(
        np.random.binomial(1, frac, (n_circuits, len(genes_list))),
        columns=genes_list,
        index=rnd_circuit_names
    )

    rnd_circuits_per_pathway = pd.DataFrame(
        index=rnd_pathway_names, columns=rnd_circuit_names, data=0
    )

    for i, circuit_name in enumerate(rnd_circuit_names):
        for j, pathway_name in enumerate(rnd_pathway_names):
            #Circuit names start with pathway names
            if circuit_name.startswith(pathway_name):
                rnd_circuits_per_pathway.iloc[j, i] = 1
        
    return rnd_genes_per_pathway, rnd_genes_per_circuit, rnd_circuits_per_pathway

def filter_genes(adata, genes_per_pathway, genes_per_circuit, circuits_per_pathway, min_genes_per_circuit=4):
    print(f"Filtering genes based on pathways and circuits with minimum {min_genes_per_circuit} genes per circuit")
    adata_genes = adata.var_names.values.tolist()
    pathway_genes = genes_per_pathway.columns.tolist()
    circuit_genes = genes_per_circuit.columns.tolist()

    pathway_genes_filtered = [gene for gene in pathway_genes if gene in adata_genes]
    circuit_genes_filtered = [gene for gene in circuit_genes if gene in adata_genes]
    # Intersect both lists to get genes that are in both pathways and circuits
    genes_filtered = sorted(set(pathway_genes_filtered) & set(circuit_genes_filtered))

    adata_filtered = adata[:, genes_filtered].copy()

    #Filter columns of genes_per_pathway
    genes_per_pathway_filtered = genes_per_pathway[genes_filtered]
    #Filter columns of genes_per_circuit
    genes_per_circuit_filtered = genes_per_circuit[genes_filtered]

    #Filter rows of genes_per_circuit
    circuits_with_enough_genes = genes_per_circuit_filtered.sum(axis=1) >= min_genes_per_circuit
    genes_per_circuit_filtered = genes_per_circuit_filtered.loc[circuits_with_enough_genes, :]
    #Filter columns of circuits_per_pathway
    circuits_per_pathway_filtered = circuits_per_pathway.loc[:, circuits_with_enough_genes]

    print(f"Circuits in genes_per_circuit: {genes_per_circuit_filtered.shape[0]}")
    print(f"Circuits in circuits_per_pathway: {circuits_per_pathway_filtered.shape[1]}")

    return adata_filtered, genes_per_pathway_filtered, genes_per_circuit_filtered, circuits_per_pathway_filtered

#CODE FROM ORIGINAL PIVAE

def get_reactome_adj(pth=None):
    """
    Parse a GMT file to create a pathway adjacency matrix.
    
    Parameters:
    -----------
    pth : str or Path, optional
        Path to the GMT file. If None, uses the default Reactome pathways file.
    
    Returns:
    --------
    pandas.DataFrame
        A binary adjacency matrix where rows are genes and columns are pathways.
        A value of 1 indicates the gene belongs to the pathway.
    """
    # Use default pathway file if none provided
    if pth is None:
        pth = get_resource_path("c2.cp.reactome.v7.5.1.symbols.gmt")

    # Dictionary to store pathway-gene relationships
    pathways = {}

    # Parse the GMT file line by line
    with Path(pth).open("r") as f:
        for line in f:
            # Each line has format: pathway_name, description, gene1, gene2, ...
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    # Create a long-format DataFrame with each row representing a gene-pathway association
    reactome = pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )

    # Transform to a binary adjacency matrix (genes × pathways)
    reactome = (
        reactome.drop_duplicates()          # Remove any duplicated entries
        .assign(belongs_to=1)               # Add a column with value 1 for pivot
        .pivot(
            columns="geneset",              # Pathways become columns
            index="genesymbol",             # Genes become rows
            values="belongs_to"             # Value of 1 if gene belongs to pathway
        )
        .fillna(0)                          # 0 if gene doesn't belong to pathway
    )

    return reactome.T


def read_circuit_adj(with_effectors=False, gene_list=None):
    """
    Read the circuit adjacency matrix from the resource folder.
    
    Parameters:
    -----------
    with_effectors : bool, default=False
        If True, keeps the original values. If False, binarizes the matrix.
    gene_list : list or None, default=None
        If provided, filters the matrix to include only these genes.
    
    Returns:
    --------
    pandas.DataFrame
        Circuit adjacency matrix where rows are genes and columns are circuits.
    """
    path = get_resource_path("pbk_circuit_hsa_sig.tar.xz")
    adj = pd.read_csv(path, sep=",", index_col=0)
    
    # Convert gene symbols to uppercase for consistency
    adj.index = adj.index.str.upper()
    
    # Binarize the matrix if not using effector values
    if not with_effectors:
        adj = 1 * (adj > 0)

    # Filter to specified genes if provided
    if gene_list is not None:
        adj = adj.loc[adj.index.intersection(gene_list), :]

    # Clean up circuit IDs and remove specific pathways
    adj.columns = adj.columns.str.replace(" ", ".")
    to_remove = adj.columns.str.contains("hsa04218")
    adj = adj.loc[:, ~to_remove]

    return adj


def build_pathway_adj_from_circuit_adj(circuit_adj):
    """
    Build a pathway adjacency matrix from a circuit adjacency matrix.
    
    This function aggregates circuit-gene relationships to pathway-gene relationships.
    
    Parameters:
    -----------
    circuit_adj : pandas.DataFrame
        Circuit adjacency matrix where rows are genes and columns are circuits.
    
    Returns:
    --------
    pandas.DataFrame
        Pathway adjacency matrix where rows are pathways and columns are genes.
    """
    # Transpose to get circuits as rows
    tmp_adj = circuit_adj.T
    tmp_adj.index.name = "circuit"
    tmp_adj = tmp_adj.reset_index()
    
    # Extract pathway ID from circuit ID (format: xxx-pathwayID-xxx)
    tmp_adj["pathway"] = tmp_adj.circuit.str.split("-").str[1]
    tmp_adj = tmp_adj.drop("circuit", axis=1)
    
    # Group by pathway and check if any circuit in the pathway connects to the gene
    adj = 1 * tmp_adj.groupby("pathway").any()

    return adj



def build_circuit_pathway_adj(circuit_adj, pathway_adj):
    """
    Build a circuit-to-pathway adjacency matrix.
    
    Parameters:
    -----------
    circuit_adj : pandas.DataFrame
        Circuit adjacency matrix where rows are genes and columns are circuits.
    pathway_adj : pandas.DataFrame
        Pathway adjacency matrix where rows are pathways and columns are genes.
    
    Returns:
    --------
    pandas.DataFrame
        Binary matrix indicating which circuits belong to which pathways.
    """
    # Matrix multiplication followed by binarization to establish connections
    return (1 * (pathway_adj.dot(circuit_adj) > 0))


def get_adj_matrices(gene_list=None):
    """
    Get all adjacency matrices needed for pathway analysis.
    
    Parameters:
    -----------
    gene_list : list or None, default=None
        If provided, filters matrices to include only these genes.
    
    Returns:
    --------
    tuple
        (circuit_adj, circuit_to_pathway) - Adjacency matrices for 
        circuit-gene and circuit-pathway relationships.
    """
    circuit_adj = read_circuit_adj(with_effectors=False, gene_list=gene_list)
    pathway_adj = build_pathway_adj_from_circuit_adj(circuit_adj)
    circuit_to_pathway = build_circuit_pathway_adj(circuit_adj, pathway_adj)

    return pathway_adj, circuit_adj.T, circuit_to_pathway


def read_circuit_names():
    """
    Read and preprocess the circuit names from the resource folder.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing circuit IDs, names, and extracted effector information.
    """
    path = get_resource_path("circuit_names.tsv.tar.xz")
    circuit_names = pd.read_csv(path, sep="\t")
    
    # Preprocess circuit IDs and extract effector information
    circuit_names.hipathia_id = circuit_names.hipathia_id.str.replace(" ", ".")
    circuit_names["effector"] = circuit_names.name.str.split(": ").str[-1]
    
    return circuit_names


def build_hipathia_renamers():
    """
    Build dictionaries to map between internal IDs and human-readable names.
    
    Returns:
    --------
    tuple
        (circuit_renamer, pathway_renamer, circuit_to_effector) - Dictionaries for
        mapping between IDs and names/effectors.
    """
    # Read the circuit names data
    circuit_names = read_circuit_names()
    circuit_names = circuit_names.rename(
        columns={"name": "circuit_name", "hipathia_id": "circuit_id"}
    )
    
    # Extract pathway information from circuit IDs
    circuit_names["pathway_id"] = circuit_names["circuit_id"].str.split("-").str[1]
    circuit_names["pathway_name"] = circuit_names["circuit_name"].str.split(":").str[0]
    
    # Create mapping dictionaries
    circuit_renamer = circuit_names.set_index("circuit_id")["circuit_name"].to_dict()
    pathway_renamer = circuit_names.set_index("pathway_id")["pathway_name"].to_dict()
    circuit_to_effector = (
        circuit_names.set_index("circuit_name")["effector"].str.strip().to_dict()
    )

    return circuit_renamer, pathway_renamer, circuit_to_effector

# ADDED by Rafa:
def mask_anndata(adata, db="KEGG", mask_mode="kegg"):
    adata = adata.copy()
    # Generate Masks
    print("Generating masks...")
    genes_per_pathway, genes_per_circuit, circuits_per_pathway = get_pathway_masks(db)
    adata, genes_per_pathway, genes_per_circuit, circuits_per_pathway = filter_genes(adata, genes_per_pathway, genes_per_circuit, circuits_per_pathway)
    frac = genes_per_circuit.sum(axis=1).mean() / genes_per_circuit.shape[1]
    print(f"Fraction of genes per circuit: {frac:.2f}", flush=True)
    print(f"Number of genes: {genes_per_circuit.shape[1]}", flush=True)
    rnd_genes_per_pathway, rnd_genes_per_circuit, rnd_circuits_per_pathway = get_random_masks(adata.var_names, genes_per_circuit.shape[0], genes_per_pathway.shape[0], frac=frac, seed=42)

    masks_kegg = [genes_per_circuit, circuits_per_pathway]
    masks_kegg_rnd = [rnd_genes_per_circuit, rnd_circuits_per_pathway]
    masks_kegg_pathways = [genes_per_pathway]
    masks_kegg_pathways_rnd = [rnd_genes_per_pathway]

    # Select mask to use
    if mask_mode == "kegg":
        masks = masks_kegg
    elif mask_mode == "kegg_rnd":
        masks = masks_kegg_rnd
    elif mask_mode == "kegg_pathways":
        masks = masks_kegg_pathways
    elif mask_mode == "kegg_pathways_rnd":
        masks = masks_kegg_pathways_rnd
    else:
        raise ValueError(f"Unknown mask type: {mask_mode}")

    return adata, masks

