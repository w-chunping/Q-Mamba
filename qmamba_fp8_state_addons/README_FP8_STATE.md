# FP8 fake state quantization addons for Q-Mamba experiments

This package adds FP8 fake quantization for recurrent Mamba2 `ssm_state`.
It is compatible with the existing DSQ/ESR pipeline and optional SmoothQuant W8A8 addon.

## Install

From the project root containing `common.py`, `qmamba/quant/state_quant.py`, and your eval scripts:

```bash
unzip qmamba_fp8_state_addons.zip
cp qmamba_fp8_state_addons/qmamba/quant/fp8_state.py qmamba/quant/
cp qmamba_fp8_state_addons/qmamba/patch/patch_mamba2_fp8_state.py qmamba/patch/
cp qmamba_fp8_state_addons/scripts/*.py scripts/
chmod +x scripts/sweep_fp8_state_ppl.sh
export PYTHONPATH=$PWD
```

Import test:

```bash
PYTHONPATH=$PWD python - <<'PY'
from qmamba.quant.fp8_state import quantize_state_fp8
from qmamba.patch.patch_mamba2_fp8_state import patch_mamba2_step_mixed_state_quant
print('fp8 state import ok')
PY
```

## Supported state methods

Raw FP8, no external scale:

```text
fp8_e4m3
fp8_e5m2
```

Externally scaled FP8:

```text
fp8_e4m3_scaled_tensor
fp8_e5m2_scaled_tensor
fp8_e4m3_scaled_head
fp8_e5m2_scaled_head
fp8_e4m3_scaled_channel
fp8_e5m2_scaled_channel
fp8_e4m3_scaled_state
fp8_e5m2_scaled_state
```

DSQ-like two-factor scale, but FP8 payload:

```text
fp8_e4m3_dsq
fp8_e5m2_dsq
fp8_e4m3_dsq_rms
fp8_e5m2_dsq_rms
```

For Mamba2 recurrent state `[B,H,P,N]`:

```text
head granularity:    one external scale per [B,H]
channel granularity: one external scale per [B,H,P]
state granularity:   one external scale per [B,H,N]
```

## Small sweep first

```bash
SEQ_LEN=512 MAX_WINDOWS=8 OUT_DIR=results/fp8_state_sweep_w8 \
  bash scripts/sweep_fp8_state_ppl.sh
```

This compares FP16, INT4 DSQ, raw FP8, scaled FP8, and DSQ-like FP8 on the same 8 windows.

## Single run example

```bash
PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_fp8_state.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 512 \
  --stride 512 \
  --batch_size 1 \
  --max_windows 8 \
  --state_quant fp8_e4m3_scaled_head \
  --out_dir results/fp8_state
```

## With ESR checkpoint

```bash
PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_fp8_state.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 512 \
  --stride 512 \
  --batch_size 1 \
  --max_windows 8 \
  --checkpoint results/phase2_xxx/mamba2_370m_esr_parallel.pt \
  --state_quant fp8_e4m3_scaled_head \
  --out_dir results/fp8_state
```

## With SmoothQuant W8A8, if the SmoothQuant addon is installed

```bash
PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_fp8_state.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 512 \
  --stride 512 \
  --batch_size 1 \
  --max_windows 8 \
  --sq_scales data/sq/mamba2_370m_rp128x2048_mixer.pt \
  --sq_alpha 0.5 \
  --sq_target mixer \
  --state_quant fp8_e4m3_scaled_head \
  --out_dir results/fp8_state_w8a8
```

## Error inspection

Before running a long full PPL sweep, inspect quantization error on actual cached states:

```bash
PYTHONPATH=$PWD python scripts/inspect_fp8_state_error.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --seq_len 256 \
  --step 128 \
  --method fp8_e4m3_scaled_head
```

## Recommended experiment order

1. `seq_len=512`, `max_windows=8`, no ESR, W16A16.
2. Select best FP8 state methods.
3. Full recurrent PPL only for the promising methods.
4. Repeat best methods with W8A8 SmoothQuant.
5. Only if useful, try ESR-style reconstruction for FP8 state.

The most meaningful first candidates are:

```text
fp8_e4m3_scaled_head
fp8_e5m2_scaled_head
fp8_e4m3_dsq
fp8_e5m2_dsq
```
