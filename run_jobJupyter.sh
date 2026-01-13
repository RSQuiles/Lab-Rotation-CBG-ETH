#!/bin/bash
#SBATCH --job-name=jupyter_server
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=64G ## CHANGE IT AS NEEDED
#SBATCH --output sbatch_outputs/jupyter_server_%j.out

# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc
conda activate fcr_marina ## CHANGE IT WITH YOUR CONDA ENV

# Root in the adequate folder
cd $work

# Start Jupyter
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
