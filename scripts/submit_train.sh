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

source $(conda info --base)/etc/profile.d/conda.sh
conda activate matersim

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Disable InfiniBand since both GPUs are on the same local node
export NCCL_IB_DISABLE=1
# Force NCCL to use standard socket/shared-memory interfaces
export NCCL_SOCKET_IFNAME=eth0,enp

# Navigate to the repository root before executing relative paths
cd /blue/hennig/ssavarkar/Ultra-hard-materials/MECNET_github

# Run DDP across 2 GPUs
torchrun --nproc_per_node=2 src/training/train.py

