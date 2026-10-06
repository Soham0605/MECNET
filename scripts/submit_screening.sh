#!/bin/bash
#SBATCH --job-name=ALIGNN_L4_predict
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=hennig
#SBATCH --account=hennig
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# 1. Environment Activation
source $(conda info --base)/etc/profile.d/conda.sh
conda activate matersim

# 2. Dynamic Library & GPU Memory Optimizations
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

# 3. Materials Project API Key (replace if not already exported in your ~/.bashrc)
# export MP_API_KEY="your_api_key_here"

# 4. Navigate to Repository Root
cd /blue/hennig/ssavarkar/Ultra-hard-materials/MECNET_github

# 5. Run Single-GPU High-Throughput Inference
python -u src/utils/screen_mp.py
