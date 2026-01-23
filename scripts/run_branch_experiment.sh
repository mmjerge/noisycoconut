#!/bin/bash

#SBATCH --job-name=coconut-branch
#SBATCH --output=logs/branch_%j.out
#SBATCH --error=logs/branch_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=768GB
#SBATCH --time=72:00:00
#SBATCH --account=your-account-here
#SBATCH --constraint=a100_80gb

# Create logs directory
mkdir -p /scratch/mj6ux/Projects/coconut/logs

# Initialize cuda
module load cuda/12.4.1

# Initialize conda
source /sfs/weka/applications/202506_build/software/standard/core/miniforge/24.3.0-py3.11/etc/profile.d/conda.sh
conda activate coconut

# Force conda env to take priority
export PATH="/home/mj6ux/.conda/envs/coconut/bin:$PATH"
export PYTHONNOUSERSITE=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Torchrun: $(which torchrun)"

python run.py --config args/noisy-coconut.yaml
