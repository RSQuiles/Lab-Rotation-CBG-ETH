import json
import matplotlib.pyplot as plt
import os
import time
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import argparse
import scanpy as sc

from fcr import FCR_sim
from fcr import fetch_latest
from fcr.evaluate.evaluate import evaluate, evaluate_classic, evaluate_prediction

# Get path to corresponding CHECKPOINT FILE and LOG FILE    
parser = argparse.ArgumentParser()
parser.add_argument("-e", type=str, help="Name of the experiment") # defines directory
parser.add_argument("-d", type=str, help="Experiment description") # defines sub-directory
parser.add_argument("-f", action="store_true", help="Activates the UMAP output")
parser.add_argument("-r", action="store_true", help="Reuses latest latent representation added to the AnnData object")
parser.add_argument("--epoch", type=int, help="Determines which stage (epoch) of the model to use")
args = parser.parse_args()

experiment_name = args.e
description = args.d
umaps = args.f
reuse = args.r
target_epoch = args.epoch

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = str(os.path.join(script_dir, experiment_name, description))

if target_epoch is None:
    model_path = fetch_latest(model_dir)
    #print("\nModel path: ", model_path, "\n")
else:
    saves_dir = os.path.join(model_dir, "saves")
    suffix = ".pt"
    # Retrieve all model checkpoints in the saves directory
    ckpts = []    
    for root, _, files in os.walk(saves_dir):
        for f in files:
            if f.endswith(suffix):
                ckpts.append(os.path.join(root, f))
    # Get the chekpoint corresponding to the specified epoch
    target_file = next((f for f in ckpts if f"epoch={target_epoch}.pt" in f), None)
    model_path = target_file

log_path = str(os.path.join(script_dir, experiment_name, description, "output_log.out"))
output_dir = str(os.path.join(script_dir, experiment_name, description, "plots"))

# LOAD ARGUMENTS
fcr_model = FCR_sim(model_path=model_path, dataset_mode="all")
args = fcr_model.arguments

####################################################
######## PLOT TRAINING AND EVALUATION STATS ########
####################################################

# Storage
epochs = []
training_stats = {}
evaluation_stats = {}

# Read file line by line
with open(log_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip malformed lines

        # If line has an epoch, collect stats
        if "epoch" in record:
            epochs.append(record["epoch"])

            # Training stats
            for k, v in record["training_stats"].items():
                training_stats.setdefault(k, []).append(v)

            # Evaluation stats
            for split, vals in record["evaluation_stats"].items():
                for i, v in enumerate(vals):
                    key = f"{split}_{i}"  # e.g. "train_0", "test_1"
                    evaluation_stats.setdefault(key, []).append(v)


# Rename evaluation variables
evaluation_stats["Mean training"] = evaluation_stats.pop("train_0")
evaluation_stats["Stddev training"] = evaluation_stats.pop("train_1")
evaluation_stats["Mean test"] = evaluation_stats.pop("test_0")
evaluation_stats["Stddev test"] = evaluation_stats.pop("test_1")

# Plot training stats
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "training_eval_plots.md"), "w") as log:
    for k, vals in training_stats.items():
        fname = f"training_{k.replace(' ', '_')}.png"
        fpath = os.path.join(output_dir, fname)

        plt.figure()
        plt.plot(epochs, vals, marker="o")
        plt.title(f"Training {k}")
        plt.xlabel("Epoch")
        plt.ylabel(k)
        plt.grid(True)
        plt.savefig(fpath, dpi=100)
        plt.close()

        # Append to markdown file
        log.write(f"### Training {k}\n\n")
        log.write(f"![{k}]({fname})\n\n")
    
    # Add shortened KL divergence
    fname = f"training_KL_Divergence_Short.png"
    fpath = os.path.join(output_dir, fname)
    n = 50

    plt.figure()
    plt.plot(epochs[n:],training_stats["KL Divergence"][n:], marker="o")
    plt.title("Training KL Divergence Short")
    plt.xlabel("Epoch")
    plt.ylabel(k)
    plt.grid(True)
    plt.savefig(fpath, dpi=100)
    plt.close()

    log.write(f"### Training KL Divergence Short\n\n")
    log.write(f"![KL Divergence Short](training_KL_Divergence_Short.png)\n\n")

# Plot evaluation stats
with open(os.path.join(output_dir, "training_eval_plots.md"), "a") as log:
    for k, vals in evaluation_stats.items():
        fname = f"eval_{k.replace(' ', '_')}.png"
        fpath = os.path.join(output_dir, fname)

        plt.figure()
        plt.plot(epochs, vals, marker="o")
        plt.title(f"Evaluation {k}")
        plt.xlabel("Epoch")
        plt.ylabel(k)
        plt.grid(True)
        plt.savefig(fpath, dpi=100)
        plt.close()

        # Append to markdown log
        log.write(f"### Evaluation {k}\n\n")
        log.write(f"![{k}]({fname})\n\n")

# Generate summary of the configuration
config_sum_path = str(os.path.join(script_dir, experiment_name, description, "config_sum.json"))
with open(config_sum_path, "w") as f:
    json.dump(args, f, indent=4)


####################################################
################# PLOT UMAP RESULTS ################
####################################################

if umaps:
    # LOAD MODEL AND DATASETS
    trained_model = fcr_model.model
    datasets = fcr_model.dataset

    # LOAD REPRESENTATIONS (if they do not exists yet)
    adata = sc.read(args["data_path"])

    if not reuse:
        print("Computing latent representations...")
        ZXs = []
        ZTs = []
        ZXTs = []
        for data in datasets["loader_tr"]:
            (genes, perts, cf_genes, cf_perts, covariates) = (
                    data[0], data[1], data[2], data[3], data[4:])

            ZX, ZXT, ZT = trained_model.get_latent_presentation(genes, perts, covariates, sample=False) # sigma1=2e-1, sigma2=5e-3, sigma3=1e-3)
            ZXs.extend(ZX)
            ZTs.extend(ZT)
            ZXTs.extend(ZXT)

        ZXs = [e.detach().cpu().numpy() for e in ZXs]
        ZXs = np.array(ZXs)
        print("ZX mean:", ZXs.mean(), "ZX std:", ZXs.std())
        ZXTs = [e.detach().cpu().numpy() for e in ZXTs]
        ZXTs = np.array(ZXTs)
        print("ZXT mean:", ZXTs.mean(), "ZXT std:", ZXTs.std())
        ZTs = [e.detach().cpu().numpy() for e in ZTs]
        ZTs = np.array(ZTs)
        print("ZT mean:", ZTs.mean(), "ZT std:", ZTs.std())

        # Append to adata
        adata.obsm["ZXs"] = ZXs
        adata.obsm["ZTs"] = ZTs
        adata.obsm["ZXTs"] = ZXTs
        # Export to avoid computing again
        adata.write(args["data_path"])

    # UMAP settings
    n_neighbors = 10 
    min_dist = 0.05 
    spread = 3.0
    metric = "cosine"
    size = 10

    # Plot ZX
    print("Plotting ZX UMAP...")
    sc.pp.neighbors(adata, use_rep="ZXs",
                    n_neighbors = n_neighbors,
                    metric = metric)
    
    sc.tl.umap(adata, min_dist=min_dist)
    sc.pl.umap(
        adata,
        color=["cell_name", "Agg_Treatment"],
        frameon=False,
        palette="Set3",
        size=size,
        outline_color="gray",
        outline_width=0.5,
        legend_loc = "on data",
        color_map="Blues",
        vcenter=0.01,
        show=False
    )

    plt.savefig(os.path.join(output_dir,"UMAP_ZXs.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # PLot ZXT
    print("Plotting ZXT UMAP...")
    sc.pp.neighbors(adata, use_rep="ZXTs",
                    n_neighbors = n_neighbors,
                    metric = metric)
    
    sc.tl.umap(adata, min_dist=min_dist)
    sc.pl.umap(
        adata,
        color=["cell_name", "Agg_Treatment"],
        frameon=False,
        palette="Set2",
        size=size,
        outline_color="gray",
        outline_width=0.5,
        legend_loc = "on data",
        color_map="Blues",
        vcenter=0.01,
        show=False
    )

    plt.savefig(os.path.join(output_dir,"UMAP_ZXTs.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot ZT
    print("Plotting ZT UMAP...")
    sc.pp.neighbors(adata, use_rep="ZTs",
                    n_neighbors = n_neighbors,
                    metric = metric)
    
    sc.tl.umap(adata, min_dist=min_dist)
    sc.pl.umap(
        adata,
        color=["cell_name", "Agg_Treatment"],
        frameon=False,
        palette="Set1",
        size=size,
        outline_color="gray",
        outline_width=0.5,
        legend_loc = "on data",
        color_map="Blues",
        vcenter=0.01,
        show=False
    )

    plt.savefig(os.path.join(output_dir,"UMAP_ZTs.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot before FCR
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

    #PCA
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=30)
    sc.tl.umap(adata, min_dist=min_dist)

    # UMAP colored by cell_name
    sc.pl.umap(
        adata,
        color=["cell_name"],
        frameon=False,
        palette="Set3",
        size=size,
        outline_color="gray",
        outline_width=0.5,
        legend_loc = "on data",
        color_map="Blues",
        vcenter=0.01,
        show=False
    )

    plt.savefig(os.path.join(output_dir,"UMAP_cell_name.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # UMAP colored by treatment / dosage
    sc.pl.umap(
        adata,
        color=["Agg_Treatment"],
        frameon=False,
        palette="Set1",
        size=size,
        outline_color="gray",
        outline_width=0.5,
        legend_loc = "on data",
        color_map="Blues",
        vcenter=0.01,
        show=False
    )

    plt.savefig(os.path.join(output_dir,"UMAP_treatment.png"), dpi=300, bbox_inches="tight")
    plt.close()


