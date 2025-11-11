#!/bin/bash

#SBATCH --job-name=coconut-branch
#SBATCH --output=logs/branch_%j.out
#SBATCH --error=logs/branch_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=72:00:00
#SBATCH --account=uvasrg_paid
#SBATCH --constraint=a100_80gb

python3 quick_branch_test.py 800
