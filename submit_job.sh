#!/bin/bash
#SBATCH --job-name=oral_prod
#SBATCH --output=logs/slurm_output.txt
#SBATCH --error=logs/slurm_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00

# Load required modules for Oregon State HPC
module load python/3.10
module load cuda/11.8

# Activate virtual environment
source venv/bin/activate

# Ensure log directory exists
mkdir -p logs

# Generate train/validation/test splits (64/16/20)
python create_dataset.py

# Execute training pipeline with validation loop
python train_pytorch.py