This repository contains the code files which were used to generate the results of the Lab Rotation I carried out in the Computational Biology Group (ETH, DBSSE) from September to December 2025 under the supervision of Marina Esteban Medina and Diane Duroux. It is organized in the following directories:

- data: contains the scripts used for the generation of and inspection of the different datasets used in the project, which are subsets of the Tahoe100M dataset.
- experiments: contains the scripts and config files developed to run experiments in the context of the FCR method validation and scaling.
- fcr: conatins the source code for the FCR method, adapted and extended from the paper "Learning Identifiable Factorized Causal Representations of Cellular Responses" (Mao et al., 2024).
- piscvi: contains the scripts and source code used to train piscVI models, as well as to carry out the benchmarking of different embeddings (benchmark_rafa.py). It is mostly based on worked developed previously in the group by Marina Esteban Medina.
- zeroshotamr: contains the source code of the ZeroShotAMR model, developed by Diane Duroux, as well as the scripts used to generate the necessary input files and to run the corresponding experiments in the context of the present project.
- tests: contains scripts used perform different tests throughout the project.
