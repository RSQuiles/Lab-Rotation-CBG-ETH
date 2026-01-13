#!/bin/bash
#SBATCH --job-name=parallel_fcr   
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=32G
#SBATCH --time=48:00:00
#SBATCH --output sbatch_outputs/slurm_%j.out

# Make sure we are in a Bash session and conda is initialized
source ~/.bashrc
conda activate fcr_marina ## CHANGE IT WITH YOUR CONDA ENV

# Definition of working directories
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> <description> <parameters>"
    exit 1
fi

if [ -z "$2" ]; then
	echo "Usage: $0 <experiment_name> <description> <parameters>"
    exit 1
fi

EXP="$1"
DESCRIP="$2"
shift 2   # remove the first 2 args so $@ now contains only parameters

mkdir -p "$EXP"/"$DESCRIP"

# Redirect stdout & stderr
exec > >(stdbuf -oL tee "$EXP"/"$DESCRIP"/output_log.out) 2>&1

echo "JOB_ID"=$SLURM_JOB_ID

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "MASTER_PORT="$MASTER_PORT

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Disable NCCL warnings
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

export PYTHONUNBUFFERED=1

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Execute the program
# For testing and debugging
echo "Python processes should start now..."
srun python -u fcr_run.py --parallel --experiment "$EXP" --description "$DESCRIP" --parameters "$@"
