#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-state-spaces/mamba2-370m}
DTYPE=${DTYPE:-float16}
SEQ_LEN=${SEQ_LEN:-2048}
OUT_DIR=${OUT_DIR:-results/fp8_state_full_w8a8}

METHODS=(
  none
  dsq
  fp8_e4m3
  fp8_e5m2
  fp8_e4m3_scaled_channel
  fp8_e5m2_scaled_channel
  fp8_e4m3_scaled_state
  fp8_e5m2_scaled_state
  fp8_e4m3_dsq
  fp8_e5m2_dsq
)

for m in "${METHODS[@]}"; do
  echo "=== Running $m ==="
  bits=4
  if [[ "$m" == "none" ]]; then bits=0; fi
  PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_fp8_state.py \
    --model "$MODEL" \
    --loader mamba_ssm \
    --dtype "$DTYPE" \
    --dataset wikitext2 \
    --seq_len "$SEQ_LEN" \
    --stride "$SEQ_LEN" \
    --batch_size 1 \
    --sq_scales data/sq/mamba2_370m_rp128x2048_mixer.pt \
    --sq_alpha 0.5 \
    --sq_target mixer \
    --state_quant "$m" \
    --state_bits "$bits" \
    --out_dir "$OUT_DIR"
done
