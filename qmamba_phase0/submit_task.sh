#!/bin/bash
#SBATCH --job-name=qmamba_p0
#SBATCH --account MST112145
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,BEGIN     # Send the mail when the job starts and finishes.    
#SBATCH --mail-user=danielwang20040527@gmail.com

source ~/.bashrc
conda activate qmamba

export HF_HOME=/work/$USER/hf_cache
export HF_DATASETS_CACHE=/work/$USER/hf_datasets

cd /path/to/qmamba_phase0
bash run_phase0.sh