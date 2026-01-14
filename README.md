# Lab Rotation — Computational Biology (ETH DBSSE)

This repository contains the code used to generate the results of my **Lab Rotation** in the **Computational Biology Group**  
(**ETH Zurich, DBSSE**) from **September–December 2025**.

The project was carried out under the supervision of  
**Marina Esteban Medina** and **Diane Duroux**.

---

## Repository Structure

The repository is organized into the following directories:

### `data/`
Scripts for generating and inspecting the datasets used in the project.  
All datasets are **subsets of the Tahoe100M dataset**.

### `experiments/`
Scripts and configuration files for running experiments related to  
**FCR method validation and scaling**.

### `fcr/`
Source code for the **FCR method**, adapted and extended from:

> *Learning Identifiable Factorized Causal Representations of Cellular Responses*  
> Mao et al., 2024

### `piscvi/`
Scripts and source code for training **piscVI** models and benchmarking different embeddings  
(e.g. `benchmark_rafa.py`).

This module is largely based on prior work developed within the group by  
**Marina Esteban Medina**.

### `zeroshotamr/`
Source code for the **ZeroShotAMR** model (developed by **Diane Duroux**), along with scripts to:
- generate required input files  
- run ZeroShotAMR experiments in the context of this project

### `tests/`
Utility scripts used to perform various tests throughout the project.

---

## ✨ Notes
- All code is research-oriented and reflects experimental workflows.
- Directory-level READMEs (where present) provide additional details.

