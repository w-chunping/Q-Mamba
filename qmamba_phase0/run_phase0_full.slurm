#!/bin/bash
#SBATCH --job-name=qmamba_p0_full
#SBATCH --partition=dev
#SBATCH --account=MST112145
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,BEGIN     # Send the mail when the job starts and finishes.    
#SBATCH --mail-user=danielwang20040527@gmail.com

source ~/.bashrc
module load miniconda3
conda activate qmamba

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/$USER/hf_cache
export HF_DATASETS_CACHE=/work/$USER/hf_datasets

cd ~/workspace/qmamba/qmamba_phase0

# python scripts/eval_ppl.py \
#   --model state-spaces/mamba2-370m \
#   --loader mamba_ssm \
#   --dtype float16 \
#   --dataset wikitext2 \
#   --seq_len 2048 \
#   --stride 2048 \
#   --batch_size 1 \
#   --out_dir results/phase0

python scripts/benchmark_generate.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --batch_size 64 \
  --prompt_len 100 \
  --gen_len 100 \
  --repeat 5 \
  --out_dir results/phase0