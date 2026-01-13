#!/bin/bash
#SBATCH --job-name=srun_test     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # number of tasks per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

srun hostname
