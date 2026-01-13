print("Importing packages...")
import scanpy as sc
import anndata as ad
import numpy as np
import re
import os
import pandas as pd
from argparse import ArgumentParser
import typing
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import json

def norm(x):
    return re.sub(r"[^0-9A-Za-z]", "", str(x)).upper()

def norm_drug_gdsc(d):
    return d.split("___")[0]

def generate_combined_table(
    data_path: str,
    table_name: str = "combined_long_table.csv",
):
    """
    # Check if file already exists
    if os.path.exists(table_name):
        print(f"Combined long table {table_name} already exists. Skipping generation.")
        return
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print("Generating combined long table...")
    # Eliminate non-GDSC rows:
    df["_norm"] = df["cell_name"].map(norm)

    matched = df[df["drug"].notna()].copy()
    matched.rename(columns={
        "BARCODE_SUB_LIB_ID": "sample_id",
        "_norm": "species",
        "sensitivity": "response",
    }, inplace=True
    )

    # Placeholder for dataset variable
    matched["dataset"] = "any"

    # Normalize drug names
    matched["drug"] = matched["drug"].map(norm_drug_gdsc)

    # Eliminate Cetuximab (not a small molecule)
    matched = matched[~(matched["drug"] == "Cetuximab")]

    # Reorder columns
    matched = matched[["species", "sample_id", "drug", "response", "dataset"]]

    # EXPORT CSV FILE
    matched.to_csv(table_name, index=False)
    print(f"Saved combined long table to {table_name}.")
    

def generate_data_splits(
    metadata_table: str,
    splits_name: str = "data_splits.csv",
):
    """
    # Check if file already exists
    if os.path.exists(splits_name):
        print(f"Data splits file {splits_name} already exists. Skipping generation.")
        return
    """
    print(f"Loading metadata from {metadata_table}...")
    df = pd.read_csv(metadata_table)

    print("Generating data splits...")
    # Unique cell lines and drugs
    unique_lines = np.array(df["species"].unique())
    unique_drugs = np.array(df["drug"].unique())

    # Determine zeroshot subsets
    train_lines = unique_lines[:35]
    seen_lines = unique_lines[35:40]
    zeroshot_lines = unique_lines[40:]
    print(f"Zeroshot cell lines: {zeroshot_lines}")

    zeroshot_drugs = unique_drugs[:10]
    print(f"Zeroshot drugs: {zeroshot_drugs}")

    # Remove zeroshot drugs from train set
    print("Removing zero-shot drugs from training set...")
    df = df[~((df["species"].isin(train_lines)) & (df["drug"].isin(zeroshot_drugs)))].copy()
    df.to_csv(metadata_table, index=False)
    print(f"Filtered metadata saved to {metadata_table}.")

    # Assigning splits
    splits = df[["sample_id", "species"]].drop_duplicates("sample_id").copy()
    splits["Set"] = "train"

    # Seen cell lines: 80/20 split
    for line in seen_lines:
        idx = splits.index[splits["species"] == line]
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=42
        )
        splits.loc[train_idx, "Set"] = "train"
        splits.loc[test_idx, "Set"] = "test"

    # Zeroshot cell lines: all test
    splits.loc[splits["species"].isin(zeroshot_lines), "Set"] = "test"

    # Save splits
    splits.to_csv(splits_name, index=False)
    print(f"Saved data splits to {splits_name}.")


def generate_embeddings(
    metadata_table: str,
    adata_path: str,
    export_path: str,
    pc_input: bool = False,
    n_pcs: int = None,
    hvg_input: bool = False,
    n_hvg: int = None,
    fcr_input: bool = False,
    use_zx: bool = False,
    use_zxt: bool = False,
    use_zt: bool = False,
    piscvi_input: bool = False,
):
    print(f"Loading metadata from {metadata_table}...")
    df = pd.read_csv(metadata_table)

    # Load AnnData
    print(f"Loading AnnData from {adata_path}...")
    adata = sc.read_h5ad(adata_path)

    # GENERATE AND EXPORT NPY FILE
    sample_ids = df["sample_id"].unique()

    # Map sample_ids to AnnData indices
    idx = adata.obs_names.get_indexer(sample_ids)
    if (idx == -1).any():
        missing = sample_ids[idx == -1]
        raise ValueError(f"These sample_ids are missing from adata.obs_names: {missing}")

    print("Matching IDs and exporting NPY file...")
    if hvg_input:
        extension = f"_hvg_{n_hvg}"
        # Ensure HVG info exists
        if adata.var.get("highly_variable") is None:
            raise ValueError("adata.var['highly_variable'] not found.")

        if adata.var["highly_variable"].sum() < n_hvg:
            raise ValueError(f"adata.var['highly_variable'] contains fewer than {n_hvg} genes.")

        # Slice HVGs
        print(f"Selecting top {n_hvg} HVGs...")
        adata_hvg = adata[:, adata.var['highly_variable_rank'] < n_hvg]

        # Slice unique sample rows
        print("Extracting HVG embeddings...")
        embedding_matrix = adata_hvg.X[idx].toarray()

    elif pc_input:
        extension = f"_pcs_{n_pcs}"
        print(f"Selecting first {n_pcs} PCs...")
        embedding_matrix = adata.obsm["X_pca"][idx, :n_pcs]

    elif fcr_input:
        extension = "_fcr"
        print("Selecting FCR embeddings...")
        if use_zx:
            embedding_matrix = adata.obsm["ZXs"][idx]
        elif use_zxt:
            embedding_matrix = adata.obsm["ZXTs"][idx]
        elif use_zt:
            embedding_matrix = adata.obsm["ZTs"][idx]
        else:
            raise ValueError("One of ZX, ZXT, ZT must be true for FCR embeddings.")

    elif piscvi_input:
        extension = "_piscvi"
        print("Selecting piSCVI embeddings...")
        embedding_matrix = adata.obsm["X_piSCVI"][idx]

    else:
        raise ValueError("No valid embedding type selected.")

    npy_name = os.path.join(export_path, f"raw_data{extension}.npy")
    np.save(npy_name, embedding_matrix)
    print(f"Saved embeddings to {npy_name}.")

    return

def generate_fingerprints(
        drugs_smiles_path: str = "/cluster/work/bewi/members/rquiles/data/gdsc_smiles.csv",
        export_path: str = "/cluster/work/bewi/members/rquiles/zeroshot_amr/data/drug_fingerprints_Mol_selfies.csv"
        ):
    from rdkit.Chem import AllChem
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import MACCSkeys
    print(f"Loading drug SMILES from {drugs_smiles_path}...")
    df_smiles = pd.read_csv(drugs_smiles_path)

    def bitvect_to_bitstring(fp):
        arr = np.zeros((fp.GetNumBits(),), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return ''.join(str(x) for x in arr)   # turn into string
        
    def generate_morgan_fp(drug_df, size):
        mols = [Chem.MolFromSmiles(smiles) for smiles in drug_df["smiles"].values]
        fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=size)
        morgan_fps = [fpgen.GetFingerprint(mol) for mol in mols]
        
        drug_df[f"morgan_{size}_fp"] = [bitvect_to_bitstring(fp) for fp in morgan_fps]

    # Morgan Fingerprints
    print("Generating Morgan fingerprints...")
    generate_morgan_fp(df_smiles, 512)
    generate_morgan_fp(df_smiles, 1024)

    # MACCS Keys
    print("Generating MACCS keys...")
    mols = [Chem.MolFromSmiles(smiles) for smiles in df_smiles["smiles"].values]
    maccs_fps = [MACCSkeys.GenMACCSKeys(x) for x in mols]
    df_smiles["MACCS_fp"] = [bitvect_to_bitstring(fp) for fp in maccs_fps]

    # Export fingerprints
    df_smiles[["drug", "morgan_512_fp", "morgan_1024_fp", "MACCS_fp"]].to_csv(export_path, index=False)
    print(f"Saved drug fingerprints to {export_path}.")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--common_files",
        action="store_true",
        help="Whether to generate common files (combined table, splits)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/cluster/work/bewi/data/tahoe100/metadata/controls_col_cellCode_with_sensitivity_reduced.csv",
        help="Path to the input data file"
    )
    parser.add_argument(
        "--adata_path",
        type=str,
        default="/cluster/work/bewi/data/tahoe100/h5ad/gdsc_controls_processed_fcr.h5ad",
        help="Path to the input data file"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the experiment / dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Export path."
    )
    parser.add_argument(
        "--pcs",
        type=int,
        help="Number of principal components of the gene expression data to compute"
    )
    parser.add_argument(
        "--n_hvgs",
        type=int,
        help="Number of highly variable genes to select"
    )
    parser.add_argument(
        "--fcr",
        action="store_true",
        help="Whether to generate inputs for FCR model"
    )
    parser.add_argument(
        "--zx",
        action="store_true",
        help="Whether to use ZX latents as input"
    )
    parser.add_argument(
        "--zxt",
        action="store_true",
        help="Whether to use ZXT latents as input"
    )
    parser.add_argument(
        "--zt",
        action="store_true",
        help="Whether to use ZT latents as input"
    )
    parser.add_argument(
        "--piscvi",
        action="store_true",
        help="Whether to use piscvi model for generating inputs"
    )
    parser.add_argument(
        "--fingerprints",
        action="store_true",
        help="Whether to generate drug fingerprints"
    )
    args = parser.parse_args()

    # Define the input type to use for the samples
    pc_input = args.pcs is not None
    hvg_input = args.n_hvgs is not None
    fcr_input = args.fcr

    if sum([pc_input, hvg_input, fcr_input]) > 1:
        raise ValueError("Only one input type [pcs, n_hvgs, fcr] can be specified at a time.")

    n_pcs = None
    if pc_input:
        n_pcs = args.pcs
    
    n_hvg = None
    if hvg_input:
        n_hvg = args.n_hvgs

    if fcr_input:
        if sum([args.zx, args.zxt, args.zt]) != 1:
            raise ValueError("When using --fcr option, exactly one of --zx, --zxt, --zt must be specified.")

    # Define working mode
    if args.common_files:
        print("Generating common files:")
        generate_combined_table(
            data_path=args.data_path,
        )
        generate_data_splits(
            metadata_table= "combined_long_table.csv",
        )

    elif args.fingerprints:
        print("Generating drug fingerprints...")
        generate_fingerprints()

    # Generate specific embeddings
    else:
        # Define output path
        if args.output_path is None:
            output_path = "/cluster/work/bewi/members/rquiles/zeroshot_amr/data"
        else:
            output_path = args.output_path
        os.makedirs(output_path, exist_ok=True)

        # Generate embeddings
        generate_embeddings(
            metadata_table = "combined_long_table.csv",
            adata_path = args.adata_path,
            export_path = output_path,
            pc_input=pc_input,
            n_pcs=n_pcs,
            hvg_input=hvg_input,
            n_hvg=n_hvg,
            fcr_input=fcr_input,
            use_zx=args.zx,
            use_zxt=args.zxt,
            use_zt=args.zt,
            piscvi_input=args.piscvi,
        )

# LEGACY CODE
def generate_inputs_controls(
    name: str,
    data_path: str,
    export_path: str = None,
    drug_code: str = "morgan",
    pc_input: bool = False,
    hvg_input: bool = False,
    fcr_input: bool = False,
    use_zx: bool = False,
    use_zxt: bool = False,
    use_zt: bool = False,
    piscvi_input: bool = False,
    n_pcs: int = None,
    n_hvg: int = None,
    not_table: bool = False,
    not_embeddings: bool = False,
    not_splits: bool = False
):
    # Define export directory
    if export_path is None:
        export_path = os.path.join("/cluster/work/bewi/members/rquiles/zeroshot_amr/data", name)

    os.makedirs(export_path, exist_ok=True)

    #######################################
    ##########  Prepare script ############
    #######################################

    controls_path = "/cluster/work/bewi/data/tahoe100/metadata/controls_col_cellCode_with_sensitivity.csv"
    print(f"Loading data from {controls_path}...")
    df = pd.read_csv(controls_path)
    imported_table = False # Flag to indicate if processes df is available

    # Exported file names
    table_name = os.path.join(export_path, "combined_long_table.csv")
    npy_name = os.path.join(export_path, f"raw_data.npy")
    splits_name = os.path.join(export_path, "data_splits.csv")

    #######################################
    ######### COMBINED LONG TABLE #########
    #######################################
    if not not_table:
        print("Generating combined long table...")

        df.rename(columns={
            "BARCODE_SUB_LIB_ID": "sample_id",
            "cell_line": "species",
            "sensitivity": "response",
        }, inplace=True)

        df["dataset"] = "any"  # Placeholder for dataset variable

        df = df[["species", "sample_id", "drug", "response", "dataset"]] # Order of columns
        imported_table = True
        
        # EXPORT CSV FILE
        df.to_csv(table_name, index=False)
        print(f"Saved combined long table to {table_name}.")

    #######################################
    ############  EMBEDDINGS ##############
    #######################################
    if not not_embeddings:
        print("Selecting embeddings...")

        if not imported_table:
            print(f"Loading metadata from {table_name}...")
            df = pd.read_csv(table_name)
            imported_table = True

        # Load AnnData
        print(f"Loading data from {data_path}...")
        adata = sc.read_h5ad(data_path)

        # GENERATE AND EXPORT NPY FILE
        sample_ids = df["sample_id"].unique()

        # Map sample_ids to AnnData indices
        idx = adata.obs_names.get_indexer(sample_ids)
        if (idx == -1).any():
            missing = sample_ids[idx == -1]
            raise ValueError(f"These sample_ids are missing from adata.obs_names: {missing}")

        print("Matching IDs and exporting NPY file...")
        if hvg_input:
            # Ensure HVG info exists
            if adata.var.get("highly_variable") is None:
                raise ValueError("adata.var['highly_variable'] not found.")

            if adata.var["highly_variable"].sum() < n_hvg:
                raise ValueError(f"adata.var['highly_variable'] contains fewer than {n_hvg} genes.")

            # Slice HVGs
            print(f"Selecting top {n_hvg} HVGs...")
            adata_hvg = adata[:, adata.var['highly_variable_rank'] < n_hvg]

            # Slice unique sample rows
            print("Extracting HVG embeddings...")
            embedding_matrix = adata_hvg.X[idx].toarray()

        elif pc_input:
            print(f"Selecting first {n_pcs} PCs...")
            embedding_matrix = adata.obsm["X_pca"][idx, :n_pcs]

        elif fcr_input:
            if use_zx:
                embedding_matrix = adata.obsm["ZXs"][idx]
            elif use_zxt:
                embedding_matrix = adata.obsm["ZXTs"][idx]
            elif use_zt:
                embedding_matrix = adata.obsm["ZTs"][idx]
            else:
                raise ValueError("One of ZX, ZXT, ZT must be true for FCR embeddings.")

        elif piscvi_input:
            embedding_matrix = adata.obsm["X_piSCVI"][idx]

        else:
            raise ValueError("No valid embedding type selected.")

        np.save(npy_name, embedding_matrix)
        print(f"Saved embeddings to {npy_name}.")

    #######################################
    ############# DATA SPLITS #############
    #######################################
    if not not_splits:
        print("Generating data splits...")

        if not imported_table:
            print(f"Loading metadata from {table_name}...")
            df = pd.read_csv(table_name)
            imported_table = True

        # Unique cell lines and drugs
        unique_cells = np.array(df["species"].unique())
        unique_drugs = np.array(df["drug"].unique())
        rng = np.random.default_rng(42)

        df_unique_lines = df[["sample_id", "species"]].drop_duplicates()
        df_unique_drugs = df[["sample_id", "drug"]].drop_duplicates()

        # ============================================================
        # CELL LINES: SETUP 1 — Zero-shot: pick 5 cell lines entirely as test set
        # ============================================================
        n_zs_cells = 5
        zero_shot_cells = rng.choice(unique_cells, size=n_zs_cells, replace=False)
        zero_shot_cells = set(zero_shot_cells)

        print(f"[Setup 1] Zero-shot cell lines: {zero_shot_cells}")

        df_unique_lines["Set"] = np.where(df_unique_lines["species"].isin(zero_shot_cells),
                                    "test", "train")

        # Export setup 1
        splits1_name = os.path.join(export_path, "data_splits_lines_zeroshot.csv")
        df_unique_lines[["sample_id", "Set"]].to_csv(splits1_name, index=False)
        print(f"Saved Setup 1 zero-shot splits (cell_lines) to {splits1_name}")


        # ============================================================
        # CELL LINES: SETUP 2 — Seen-cell-lines split: for chosen cell lines, split 80/20
        # ============================================================
        chosen_seen_cells = zero_shot_cells
        print(f"[Setup 2] Split-on-seen cell lines: {chosen_seen_cells}")

        # For only these cell lines, apply an 80/20 split at sample level
        for cell in chosen_seen_cells:
            idx = df_unique_lines.index[df_unique_lines["species"] == cell]
            train_idx, test_idx = train_test_split(
                idx, test_size=0.2, random_state=42
            )
            df_unique_lines.loc[train_idx, "Set"] = "train"
            df_unique_lines.loc[test_idx, "Set"] = "test"

        # Export setup 2
        splits2_name = os.path.join(export_path, "data_splits_lines_baseline.csv")
        df_unique_lines[["sample_id", "Set"]].to_csv(splits2_name, index=False)
        print(f"Saved Setup 2 80/20 seen-cell splits to {splits2_name}")

        # ============================================================
        # DRUGS: SETUP 1 — Zero-shot: pick 5 drugs entirely as test set
        # ============================================================
        n_zs_drugs = 5
        zero_shot_drugs = rng.choice(unique_drugs, size=n_zs_drugs, replace=False)
        zero_shot_drugs = set(zero_shot_drugs)

        print(f"[Setup 1] Zero-shot drugs: {zero_shot_drugs}")

        df_unique_drugs["Set"] = np.where(df_unique_drugs["drug"].isin(zero_shot_drugs),
                                    "test", "train")

        # Export setup 1
        splits1_name = os.path.join(export_path, "data_splits_drugs_zeroshot.csv")
        df_unique_drugs[["sample_id", "Set"]].to_csv(splits1_name, index=False)
        print(f"Saved Setup 1 zero-shot splits (drugs) to {splits1_name}")


        # ============================================================
        # DRUGS: SETUP 2 — Seen-drugs split: for chosen drugs, split 80/20
        # ============================================================
        chosen_seen_drugs = zero_shot_drugs
        print(f"[Setup 2] Split-on-seen drugs: {chosen_seen_drugs}")

        # For only these drugs, apply an 80/20 split at sample level
        for drug in chosen_seen_drugs:
            idx = df_unique_drugs.index[df_unique_drugs["drug"] == drug]
            train_idx, test_idx = train_test_split(
                idx, test_size=0.2, random_state=42
            )
            df_unique_drugs.loc[train_idx, "Set"] = "train"
            df_unique_drugs.loc[test_idx, "Set"] = "test"

        # Export setup 2
        splits2_name = os.path.join(export_path, "data_splits_drugs_baseline.csv")
        df_unique_drugs[["sample_id", "Set"]].to_csv(splits2_name, index=False)
        print(f"Saved Setup 2 80/20 seen-drug splits to {splits2_name}")


        print("Finished generating both zero-shot setups.")

        print(f"All files correctly saved to {export_path}.")

    return