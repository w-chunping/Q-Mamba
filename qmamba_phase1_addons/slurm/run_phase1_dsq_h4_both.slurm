#!/bin/bash
#SBATCH --job-name=qmamba_p1_dsq_h4_both
#SBATCH --partition=normal
#SBATCH --account=MST112145
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,BEGIN     # Send the mail when the job starts and finishes.    
#SBATCH --mail-user=danielwang20040527@gmail.com

source ~/.bashrc
module load miniconda3
conda activate qmamba

cd ~/workspace/qmamba/qmamba_phase1_addons

export PYTHONPATH=$PWD
export HF_HOME=/work/$USER/hf_cache
export HF_DATASETS_CACHE=/work/$USER/hf_datasets

python scripts/eval_ppl_recurrent.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 2048 \
  --stride 2048 \
  --batch_size 1 \
  --state_quant dsq \
  --state_bits 4 \
  --out_dir results/phase1 \
  --state_quant_position both