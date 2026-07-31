#!/bin/bash
#SBATCH --job-name=qmamba_p2_h8
#SBATCH --partition=normal
#SBATCH --account=MST112145
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,BEGIN     # Send the mail when the job starts and finishes.    
#SBATCH --mail-user=danielwang20040527@gmail.com

source ~/.bashrc
module load miniconda3
conda activate qmamba

cd ~/workspace/qmamba/qmamba_phase2_addons

export PYTHONPATH=$PWD
export HF_HOME=/work/$USER/hf_cache
export HF_DATASETS_CACHE=/work/$USER/hf_datasets

PYTHONPATH=$PWD python scripts/run_esr_mixer_parallel.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype bfloat16 \
  --calib data/calib/redpajama_128x2048.pt \
  --seq_len 2048 \
  --epochs 3 \
  --block_start 0 \
  --block_end 48 \
  --state_quant dsq \
  --state_bits 8 \
  --qmode esr \
  --train_mode sga \
  --lr 1e-3 \
  --adam_eps 1e-5 \
  --out_dir results/phase2_parallel_b48_lr1e3_3ep_h8

PYTHONPATH=$PWD python scripts/eval_ppl_recurrent.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 2048 \
  --stride 2048 \
  --batch_size 1 \
  --state_quant dsq \
  --state_bits 8 \
  --checkpoint results/phase2_parallel_b48_lr1e3_3ep_h8/mamba2_370m_esr_parallel.pt \
  --out_dir results/phase2_eval_all