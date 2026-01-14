import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from argparse import ArgumentParser
print("Imports successful!")

def generate_sets(args):
    # Import necessary data
    pred_name = f"../experiments/outputs/{args.group}/{args.name}_DRIAMS-any_specific_results/test_set_seed0.csv"
    print(f"Loading predictions from {pred_name}...")
    pred = pd.read_csv(pred_name)
    # Eliminate duplicate entries (for duplicate drugs)
    print("Eliminating duplicate entries...")
    pred.drop_duplicates(inplace=True)
    df = pd.read_csv("../data/combined_long_table.csv")
    splits = pd.read_csv("../data/data_splits.csv")

    # Process to determine experiment sets
    train_ids = splits[splits["Set"] == "train"]["sample_id"].values
    test_ids = splits[splits["Set"] == "test"]["sample_id"].values

    # Cell lines
    all_lines = df["species"].unique()
    test_lines = df[df["sample_id"].isin(test_ids)]["species"].unique()
    train_lines = df[df["sample_id"].isin(train_ids)]["species"].unique()
    train_only_lines = train_lines[:35]
    train_random_lines = train_lines[35:40]
    zeroshot_lines = [line for line in test_lines if line not in train_lines]

    print(f"All lines ({len(all_lines)}): {all_lines}")
    print("\n")
    print(f"Train-Only lines ({len(train_only_lines)}): {train_only_lines}")
    print("\n")
    print(f"Test lines ({len(test_lines)}): {test_lines}")
    print("\n")
    print(f"Train-Random Split lines ({len(train_random_lines)}): {train_random_lines}")
    print("\n")
    print(f"Zeroshot lines ({len(zeroshot_lines)}): {zeroshot_lines}")

    # Drugs
    all_drugs = df["drug"].unique()
    train_drugs = df[df["species"].isin(train_only_lines)]["drug"].unique()
    zeroshot_drugs = [drug for drug in all_drugs if drug not in train_drugs]

    print(f"All Drugs: ({len(all_drugs)}): {sorted(all_drugs)}")
    print("\n")
    print(f"Train Drugs: ({len(train_drugs)}): {sorted(train_drugs)}")
    print("\n")
    print(f"Zeroshot Drugs: ({len(zeroshot_drugs)}): {sorted(zeroshot_drugs)}")

    # Define the following sets (by sample_id)
    pred["experiment"] = "_" # placeholder

    # a) Random splits
    mask = (pred["species"].isin(train_random_lines)) & (pred["drug"].isin(train_drugs))
    pred.loc[mask, "experiment"] = "random_split"

    # b) Cell line zeroshot
    mask = (pred["species"].isin(zeroshot_lines)) & (pred["drug"].isin(train_drugs))
    pred.loc[mask, "experiment"] = "cell_line_zeroshot"

    # c) Drug zeroshot
    mask = (pred["species"].isin(train_random_lines)) & (pred["drug"].isin(zeroshot_drugs))
    pred.loc[mask, "experiment"] = "drug_zeroshot"

    # d) Drug and Cell line zeroshot
    mask = (pred["species"].isin(zeroshot_lines)) & (pred["drug"].isin(zeroshot_drugs))
    pred.loc[mask, "experiment"] = "cell_line_drug_zeroshot"

    assert len(pred[pred["experiment"] == "_"]) == 0

    # Generate 4 different dataframes
    print("Generating the different experiment sets...")
    random_split = pred[pred["experiment"] == "random_split"].copy()
    cell_line_zeroshot = pred[pred["experiment"] == "cell_line_zeroshot"].copy()
    drug_zeroshot = pred[pred["experiment"] == "drug_zeroshot"].copy()
    cell_line_drug_zeroshot = pred[pred["experiment"] == "cell_line_drug_zeroshot"].copy()

    return {"Random Split": random_split,
            "Cell line Zeroshot": cell_line_zeroshot,
            "Drug Zeroshot": drug_zeroshot,
            "Cell line Drug Zeroshot": cell_line_drug_zeroshot}


def plot_prc(df, args, set_name, output_dir):
    print("Plotting Precision-Recall Curve...")
    auprc = average_precision_score(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    fig, ax = plt.subplots()

    ax.set_title(f"{set_name} - FP: {args.group} - Embedding: {args.name} - Precision-Recall Curve (AUPRC = {auprc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    sns.lineplot(x=recall, y=precision)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prc_{args.group}_{args.name}_{set_name}.png")
    plt.close()


def plot_roc(df, args, set_name, output_dir):
    print("Plotting ROC Curve...")
    fpr, tpr, thresholds = roc_curve(df["response"], df["Predictions"])
    auroc = roc_auc_score(y_true, y_pred)

    fig, ax = plt.subplots()

    ax.set_title(f"{set_name} - FP: {args.group} - Embedding: {args.name} - ROC (AUROC = {auroc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")

    ax.plot([0, 1], [0, 1], "k--")

    sns.lineplot(x=fpr, y=tpr)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_{args.group}_{args.name}_{set_name}.png")
    plt.close()

def compute_additional_metrics(y_true, y_pred, set_name, output_dir):
    print("Computing Balanced Accuracy and MCC...")
    # BACC
    bal_acc = balanced_accuracy_score(y_true, y_pred > 0.5)
    print(f"Balanced accuracy: {bal_acc:.3f}")

    # MCC
    mcc = matthews_corrcoef(y_true, y_pred > 0.5)
    print(f"Matthews correlation coefficient: {mcc:.3f}")

    # Write result to .csv file
    if not os.path.exists(f"{output_dir}/metrics.csv"):
        metrics_df = pd.DataFrame(columns=["Fingerprint", "Embedding", "Set", "Balanced_Accuracy", "MCC"])
        metrics_df.to_csv(f"{output_dir}/metrics.csv", index=False)
    
    metrics_df = pd.read_csv(f"{output_dir}/metrics.csv")
    new_row = {"Fingerprint": args.group,
               "Embedding": args.name,
               "Set": set_name,
               "Balanced_Accuracy": bal_acc,
               "MCC": mcc}
    metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
    metrics_df.to_csv(f"{output_dir}/metrics.csv", index=False)

    return bal_acc, mcc

# Compute MCC per cell line - drug combination
def compute_mcc_per_cellline_drug(df, args, set_name, output_dir, threshold=0.5, min_n=5):
    """
    df must contain: species, drug, experiment, response, Predictions
    """
    rows = []

    # group by cell line and drug
    for (cell_line, drug), g in df.groupby(["species", "drug"]):
        y_true = g["response"].astype(int).values
        y_score = g["Predictions"].values
        y_hat = (y_score > threshold).astype(int)

        n = len(g)
        pos_prop = float(np.mean(y_true))  # proportion of sensitive (assuming 1 = sensitive)
        neg_prop = 1.0 - pos_prop

        # MCC edge case: if only one class present, MCC is not informative
        # sklearn returns 0.0 in many of these cases; we can mark as NaN instead.
        if len(np.unique(y_true)) < 2 or n < min_n:
            mcc = np.nan
        else:
            mcc = matthews_corrcoef(y_true, y_hat)

        rows.append({
            "cell_line": cell_line,
            "drug": drug,
            "MCC": mcc,
            "zero_shot_split_type": set_name,     # or use g["experiment"].iloc[0]
            "drug_embedding": args.group,         # fingerprint name in your codebase
            "cell_embedding": args.name,          # pcs_10 / hvgs_1000 / piscvi etc
            "true_prop_sensitive": pos_prop,
            "true_prop_resistant": neg_prop,
            "n_observations": n,
        })

    out = pd.DataFrame(rows)

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/mcc_per_cellline_drug_{args.group}.csv"

    if not os.path.exists(out_path):
        metrics_df = pd.DataFrame(columns=["cell_line", "drug", "MCC", "zero_shot_split_type", "drug_embedding", "cell_embedding", "true_prop_sensitive", "true_prop_resistant", "n_observations"])
        metrics_df.to_csv(out_path, index=False)
    
    metrics_df = pd.read_csv(out_path)
    metrics_df = pd.concat([metrics_df, out], ignore_index=True)

    # Save table
    metrics_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return metrics_df

# Compute MCC per cell line
def compute_mcc_per_cellline(df, args, set_name, output_dir, threshold=0.5, min_n=5):
    """
    df must contain: species, drug, experiment, response, Predictions
    """
    rows = []

    # group by cell line and drug
    for cell_line, g in df.groupby(["species"]):
        y_true = g["response"].astype(int).values
        y_score = g["Predictions"].values
        y_hat = (y_score > threshold).astype(int)

        n = len(g)
        pos_prop = float(np.mean(y_true))  # proportion of sensitive (assuming 1 = sensitive)
        neg_prop = 1.0 - pos_prop

        # MCC edge case: if only one class present, MCC is not informative
        # sklearn returns 0.0 in many of these cases; we can mark as NaN instead.
        if len(np.unique(y_true)) < 2 or n < min_n:
            mcc = np.nan
        else:
            mcc = matthews_corrcoef(y_true, y_hat)

        rows.append({
            "cell_line": cell_line,
            "MCC": mcc,
            "zero_shot_split_type": set_name,     # or use g["experiment"].iloc[0]
            "drug_embedding": args.group,         # fingerprint name in your codebase
            "cell_embedding": args.name,          # pcs_10 / hvgs_1000 / piscvi etc
            "true_prop_sensitive": pos_prop,
            "true_prop_resistant": neg_prop,
            "n_observations": n,
        })

    out = pd.DataFrame(rows)

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/mcc_per_cellline_{args.group}.csv"

    if not os.path.exists(out_path):
        metrics_df = pd.DataFrame(columns=["cell_line", "MCC", "zero_shot_split_type", "drug_embedding", "cell_embedding", "true_prop_sensitive", "true_prop_resistant", "n_observations"])
        metrics_df.to_csv(out_path, index=False)
    
    metrics_df = pd.read_csv(out_path)
    metrics_df = pd.concat([metrics_df, out], ignore_index=True)

    # Save table
    metrics_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return metrics_df


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--group",
        help="Name of the group to evaluate (e.g. fingerprint name)",
    )
    parser.add_argument(
        "--name",
        help="Name of the specific experiment to evaluate (e.g. pcs_10)",
    )
    parser.add_argument(
        "--per_line_drug",
        action="store_true",
        help="Whether to evaluate per line-drug combination",
    )
    parser.add_argument(
        "--per_line",
        action="store_true",
        help="Whether to evaluate per line",
    )
    args = parser.parse_args()

    sets = generate_sets(args)
    sns.set_theme(style="whitegrid")

    output_dir = f"./results/{args.group}"
    os.makedirs(output_dir, exist_ok=True)

    for set_name, df in sets.items():
        print(f"Evaluating set: {set_name}...")

        if args.per_line:
            print(f"Analyzing MCC per cell line for: {set_name}...")
            df = df[["species", "experiment", "response", "Predictions"]].copy()
            compute_mcc_per_cellline(df, args, set_name, output_dir, threshold=0.5, min_n=5)

        # Analyze MCC per cell line - drug combination
        elif args.per_line_drug:
            print(f"Analyzing MCC per cell line - drug combination for: {set_name}...")
            df = df[["species", "drug", "experiment", "response", "Predictions"]].copy()
            compute_mcc_per_cellline_drug(df, args, set_name, output_dir, threshold=0.5, min_n=5)

        else:
            df = df[["response", "Predictions"]]
            y_true, y_pred = df["response"], df["Predictions"]

            plot_prc(df, args, set_name, output_dir)
            plot_roc(df, args, set_name, output_dir)
            bal_acc, mcc = compute_additional_metrics(y_true, y_pred, set_name, output_dir)






