#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-state-spaces/mamba2-370m}
DTYPE=${DTYPE:-float16}
SEQ_LEN=${SEQ_LEN:-512}
MAX_WINDOWS=${MAX_WINDOWS:-8}
OUT_DIR=${OUT_DIR:-results/hadamard_state_sweep_w8}

# Focused first sweep: can Hadamard make coarse scaling competitive?
# Format: method bits hadamard_dims label
RUNS=(
  "none 8 none fp16"
  "tensor 8 none int8_tensor"
  "tensor 8 state int8_hstate_tensor"
  "head 8 none int8_head"
  "head 8 state int8_hstate_head"
  "channel 8 none int8_channel"
  "channel 8 state int8_hstate_channel"
  "state 8 none int8_state"
  "state 8 state int8_hstate_state"
  "dsq 8 none int8_dsq"
  "dsq 8 state int8_hstate_dsq"
  "fp8_e4m3_scaled_tensor 8 none e4m3_tensor"
  "fp8_e4m3_scaled_tensor 8 state e4m3_hstate_tensor"
  "fp8_e4m3_scaled_head 8 none e4m3_head"
  "fp8_e4m3_scaled_head 8 state e4m3_hstate_head"
  "fp8_e4m3_scaled_channel 8 none e4m3_channel"
  "fp8_e4m3_scaled_channel 8 state e4m3_hstate_channel"
  "fp8_e4m3_scaled_state 8 none e4m3_state"
  "fp8_e4m3_scaled_state 8 state e4m3_hstate_state"
  "fp8_e4m3_dsq 8 none e4m3_dsq"
  "fp8_e4m3_dsq 8 state e4m3_hstate_dsq"
)

for r in "${RUNS[@]}"; do
  read -r method bits hdims label <<<"$r"
  echo "=== Running $label: method=$method bits=$bits hadamard=$hdims ==="
  PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_hadamard_state.py \
    --model "$MODEL" \
    --loader mamba_ssm \
    --dtype "$DTYPE" \
    --dataset wikitext2 \
    --seq_len "$SEQ_LEN" \
    --stride "$SEQ_LEN" \
    --batch_size 1 \
    --max_windows "$MAX_WINDOWS" \
    --state_quant "$method" \
    --state_bits "$bits" \
    --hadamard_dims "$hdims" \
    --out_dir "$OUT_DIR"
done
