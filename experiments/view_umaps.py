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
from typing import Optional

from fcr.validation import plot_umaps, plot_progression


def output_umaps(
    model_dir: str,
    target_epoch: int = None,
    drugs: list = None,
    all_drugs: bool = False,
    lines: list = None,
    plot_prog: bool = False,
    plot_raw: bool = False,
    plot_all: bool = False,
    plot_zx: bool = False,
    plot_zxt: bool = False,
    plot_zt: bool = False,
    sample: bool = False
):
    """
    Generate UMAP visualizations for the specified experiment directory.
    
    Parameters
    ----------
    model_dir : str
        Path to the experiment directory containing model checkpoints.
    target_epoch : int, optional
        Specific epoch to load the model from. If None, uses the latest epoch.
    filter_dict : dict, optional
        Dictionary specifying filters to apply on the data (e.g., {"drug": "DMSO_TF"}).
    plot_prog : bool
        Whether to plot progression of UMAPs over training epochs.
    """
    # Define filter dictionaries if arguments are specified
    filter_dict = None
    # drug_dict = None
    # cell_dict = None

    if drugs is not None:
        # drug_dict = {"Agg_Treatment": [drug, "DMSO_TF"]}
        drugs.append("DMSO_TF")
        filter_dict = {"Agg_Treatment": drugs}

    if lines is not None:
        # cell_dict = {"cell_name": lines}
        if filter_dict is None:
            filter_dict = {"cell_name": lines}
        else:
            filter_dict["cell_name"] = lines

    # Generate progression plots if requested (only)
    if plot_prog:
        plot_progression(
            model_dir,
            rep="ZXs",
            feature="cell_name",
            sample=sample,
            last_epoch=target_epoch,
            freq=10,
            n_cols=5
        )
        return

    # Plot umaps
    plot_umaps(model_dir, 
               target_epoch=target_epoch, 
               # drug_dict=drug_dict,
               # cell_dict=cell_dict,
               filter_dict=filter_dict,
               all_drugs=all_drugs,
               plot_raw=plot_raw,
               plot_all=plot_all,
               plot_zx=plot_zx,
               plot_zxt=plot_zxt,
               plot_zt=plot_zt,
               sample=sample
               )

if __name__ == "__main__":
    # Get path to corresponding CHECKPOINT FILE and LOG FILE    
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=str, help="Name of the experiment") # defines directory
    parser.add_argument("-d", type=str, help="Experiment description") # defines sub-directory
    parser.add_argument("--epoch", default=None, type=int, help="Determines which stage (epoch) of the model to use")
    parser.add_argument("--prog", action="store_true", help="Whether to plot progression or not")
    parser.add_argument("--drugs", nargs="+", default=None, help="If specified, filter to only these drugs")
    parser.add_argument("--all_drugs", action="store_true", help="If specified, filters to all drugs and generates UMAPs for each")
    parser.add_argument("--lines", nargs="+", default=None, help="If specified, filter to only these cell lines")
    parser.add_argument("--sample", action="store_true", help="If specified, samples from latent distributions instead of using their means")
    parser.add_argument("--raw", action="store_true", help="If specified, plots raw UMAPs")
    parser.add_argument("--zx", action="store_true", help="If specified, only plot ZX UMAP")
    parser.add_argument("--zxt", action="store_true", help="If specified, only plot ZXT UMAP")
    parser.add_argument("--zt", action="store_true", help="If specified, only plot ZT UMAP")
    args = parser.parse_args()

    experiment_name = args.e
    description = args.d

    target_epoch = args.epoch
    plot_prog = args.prog
    sample = args.sample
    plot_raw = args.raw
    selected_drugs = list(args.drugs) if args.drugs is not None else None
    all_drugs = args.all_drugs
    selected_lines = list(args.lines) if args.lines is not None else None

    # UMAP plottting selection
    plot_zx = args.zx
    plot_zxt = args.zxt
    plot_zt = args.zt

    plot_all = False
    if not plot_zx and not plot_zxt and not plot_zt:
        plot_all = True  # Default to plotting all

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = str(os.path.join(script_dir, experiment_name, description))

    output_umaps(
        model_dir=model_dir,
        target_epoch=target_epoch,
        drugs=selected_drugs,
        all_drugs=all_drugs,
        lines=selected_lines,
        plot_prog=plot_prog,
        plot_raw=plot_raw,
        plot_all=plot_all,
        plot_zx=plot_zx,
        plot_zxt=plot_zxt,
        plot_zt=plot_zt,
        sample=sample
    )

