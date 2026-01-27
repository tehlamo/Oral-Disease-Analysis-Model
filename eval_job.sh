#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/eval_output.txt
#SBATCH --error=logs/eval_error.txt

# Load required modules before activating venv (required on compute nodes)
module load python/3.10
module load cuda/11.8

# Activate virtual environment
source venv/bin/activate

python evaluate_model.py
