#!/bin/bash
#SBATCH --job-name=qmamba_p3_eval_h8
#SBATCH --partition=normal
#SBATCH --account=MST112145
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,BEGIN     # Send the mail when the job starts and finishes.    
#SBATCH --mail-user=danielwang20040527@gmail.com

source ~/.bashrc
module load miniconda3
conda activate qmamba

cd ~/workspace/qmamba/qmamba_phase3

export PYTHONPATH=$PWD
export HF_HOME=/work/$USER/hf_cache
export HF_DATASETS_CACHE=/work/$USER/hf_datasets

PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_smoothquant.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 2048 \
  --stride 2048 \
  --batch_size 1 \
  --sq_scales data/sq/mamba2_370m_rp128x2048_mixer.pt \
  --sq_alpha 0.5 \
  --sq_target mixer \
  --state_quant dsq \
  --state_bits 8 \
  --out_dir results/smoothquant


PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_smoothquant.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 2048 \
  --stride 2048 \
  --batch_size 1 \
  --checkpoint ckpt/esr_rp_b48_lr1e3_3ep_h8/mamba2_370m_esr_parallel.pt \
  --sq_scales data/sq/mamba2_370m_esr_rp128x2048_mixer.pt \
  --sq_alpha 0.5 \
  --sq_target mixer \
  --state_quant dsq \
  --state_bits 8 \
  --out_dir results/smoothquant