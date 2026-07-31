#!/bin/bash
#SBATCH --job-name=qmamba_hadamard
#SBATCH --partition=normal
#SBATCH --account=MST112145
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,BEGIN     # Send the mail when the job starts and finishes.    
#SBATCH --mail-user=danielwang20040527@gmail.com

source ~/.bashrc
module load miniconda3
conda activate qmamba

cd ~/workspace/qmamba/qmamba_hadamard_state_addons

export PYTHONPATH=$PWD
export HF_HOME=/work/$USER/hf_cache
export HF_DATASETS_CACHE=/work/$USER/hf_datasets

PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_hadamard_state_with_dump.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 2048 \
  --stride 2048 \
  --batch_size 1 \
  --max_windows 1 \
  --state_quant none \
  --hadamard_dims none \
  --dump_states results/state_dumps/mamba2_370m_wt2_fp16_states.pt \
  --dump_state_every 128 \
  --dump_state_max_steps 16 \
  --dump_state_max_windows 1 \
  --out_dir results/state_dump_eval

PYTHONPATH=$PWD python scripts/analyze_state_quant_error.py \
  --states results/state_dumps/mamba2_370m_wt2_fp16_states.pt \
  --out_dir results/state_quant_error_wt2 \
  --device cuda \
  --fp8_format e4m3 \
  --hadamard_mode state \
  --int_bits 4 \
  --dsq_range symmetric \
  --heatmap_topk 1