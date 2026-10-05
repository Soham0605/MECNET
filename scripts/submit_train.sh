#!/bin/bash
#SBATCH --job-name=MECNET_DDP
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=hennig
#SBATCH --account=hennig
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --partition=hpg-rtx6000
#SBATCH --gpus=2
#SBATCH --time=12:00:00

# Launch DDP on 2 local GPUs using torchrun
source $(conda info --base)/etc/profile.d/conda.sh
conda activate matersim

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run DDP across 2 GPUs
torchrun --nproc_per_node=2 src/training/train.py

