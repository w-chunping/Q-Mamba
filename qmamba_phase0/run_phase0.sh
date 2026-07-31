#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-state-spaces/mamba2-370m}
LOADER=${LOADER:-mamba_ssm}
DTYPE=${DTYPE:-float16}
OUT_DIR=${OUT_DIR:-results/phase0}

mkdir -p "$OUT_DIR"

# 1) Quick smoke test. Should finish quickly and catch loading/tokenizer problems.
python scripts/eval_ppl.py \
  --model "$MODEL" \
  --loader "$LOADER" \
  --dtype "$DTYPE" \
  --dataset wikitext2 \
  --seq_len 2048 \
  --stride 2048 \
  --batch_size 1 \
  --max_windows 8 \
  --out_dir "$OUT_DIR"

# 2) Full WikiText2 PPL.
python scripts/eval_ppl.py \
  --model "$MODEL" \
  --loader "$LOADER" \
  --dtype "$DTYPE" \
  --dataset wikitext2 \
  --seq_len 2048 \
  --stride 2048 \
  --batch_size 1 \
  --out_dir "$OUT_DIR"

# 3) Generation benchmark following the paper's prompt/generation length = 100.
# Start with BS=8/16 if 32 is too large for your first run.
python scripts/benchmark_generate.py \
  --model "$MODEL" \
  --loader "$LOADER" \
  --dtype "$DTYPE" \
  --batch_size 32 \
  --prompt_len 100 \
  --gen_len 100 \
  --repeat 5 \
  --out_dir "$OUT_DIR"
