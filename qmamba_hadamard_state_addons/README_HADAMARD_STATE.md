# Hadamard-enhanced INT/FP8 state quantization for Mamba2

This addon tests a rotation-based state quantization idea:

```text
state -> normalized Hadamard rotation -> INT/FP8 quantization -> inverse Hadamard rotation
```

It is compatible with both INT methods (`tensor/head/channel/state/dsq`) and FP8 methods
(`fp8_e4m3_scaled_tensor`, `fp8_e4m3_scaled_head`, `fp8_e4m3_dsq`, etc.).

This version is an accuracy/prototype implementation. It does **not** fold the rotation into B/C projections yet, and it is not a latency kernel.

## Install

From your project root:

```bash
unzip qmamba_hadamard_state_addons.zip
cp qmamba_hadamard_state_addons/qmamba/quant/hadamard_state.py qmamba/quant/
cp qmamba_hadamard_state_addons/qmamba/patch/patch_mamba2_hadamard_state.py qmamba/patch/
cp qmamba_hadamard_state_addons/scripts/*.py scripts/
chmod +x scripts/sweep_hadamard_state_ppl.sh
export PYTHONPATH=$PWD
```

Import test:

```bash
PYTHONPATH=$PWD python - <<'PY'
from qmamba.quant.hadamard_state import quantize_state_hadamard, apply_hadamard
from qmamba.patch.patch_mamba2_hadamard_state import patch_mamba2_step_hadamard_state_quant
print('hadamard state import ok')
PY
```

## Meaning of `hadamard_dims`

For Mamba2 recurrent state `[B,H,P,N]`:

```text
none    : no rotation
state   : rotate along N, the SSM state dimension
channel : rotate along P, the headdim/channel dimension
both    : rotate along P and N
```

The most important first test is `--hadamard_dims state`, because your FP8 results suggested the N/state axis is the dominant outlier-sensitive axis.

## INT8 examples

Hadamard + INT8 tensor-scale:

```bash
PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_hadamard_state.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 512 \
  --stride 512 \
  --batch_size 1 \
  --max_windows 8 \
  --state_quant tensor \
  --state_bits 8 \
  --hadamard_dims state \
  --out_dir results/hadamard_state
```

Hadamard + INT8 DSQ:

```bash
PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_hadamard_state.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 512 \
  --stride 512 \
  --batch_size 1 \
  --max_windows 8 \
  --state_quant dsq \
  --state_bits 8 \
  --hadamard_dims state \
  --out_dir results/hadamard_state
```

## FP8 examples

Hadamard + E4M3 tensor-scale:

```bash
PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_hadamard_state.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 512 \
  --stride 512 \
  --batch_size 1 \
  --max_windows 8 \
  --state_quant fp8_e4m3_scaled_tensor \
  --hadamard_dims state \
  --out_dir results/hadamard_state
```

Hadamard + E4M3 head-scale:

```bash
PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_hadamard_state.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 512 \
  --stride 512 \
  --batch_size 1 \
  --max_windows 8 \
  --state_quant fp8_e4m3_scaled_head \
  --hadamard_dims state \
  --out_dir results/hadamard_state
```

## Quick sweep

```bash
SEQ_LEN=512 MAX_WINDOWS=8 OUT_DIR=results/hadamard_state_sweep_w8 \
  bash scripts/sweep_hadamard_state_ppl.sh
```

Results are logged to:

```text
results/hadamard_state_sweep_w8/recurrent_ppl_hadamard_state_runs.csv
```

## Error inspection

Compare local reconstruction error before PPL runs:

```bash
PYTHONPATH=$PWD python scripts/inspect_hadamard_state_error.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --seq_len 256 \
  --step 128 \
  --method fp8_e4m3_scaled_tensor \
  --hadamard_dims state
```

Also test `--hadamard_dims none` for the non-rotated baseline.

## Recommended first experiment table

```text
FP16
INT8 tensor
Hadamard-state + INT8 tensor
INT8 DSQ
Hadamard-state + INT8 DSQ
E4M3 tensor
Hadamard-state + E4M3 tensor
E4M3 head
Hadamard-state + E4M3 head
E4M3 state
E4M3 DSQ
```

Success condition:

```text
Hadamard + tensor/head scaling approaches state/DSQ scaling.
```

That would support the claim that Hadamard rotation spreads state-dimension outliers and reduces the need for fine-grained scaling.
