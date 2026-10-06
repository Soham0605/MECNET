#!/bin/bash
#SBATCH --job-name=ALIGNN_L4_DDP
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=hennig
#SBATCH --account=hennig
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --partition=hpg-turin        # Changed to general hpg partition (or turin)
#SBATCH --gpus=2            # Explicitly requesting 2 L4 GPUs
#SBATCH --time=12:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate matersim

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 1. KEEP: Absolute lifesaver for 24GB L4 cards
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. KEEP: 16 CPUs / 2 GPUs = 8 threads per process
export OMP_NUM_THREADS=8

# 3. KEEP: Safe fallback for single-node DDP communication
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

cd /blue/hennig/ssavarkar/Ultra-hard-materials/MECNET_github

torchrun --nproc_per_node=2 src/training/train_alignn.py