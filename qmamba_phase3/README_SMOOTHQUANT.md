# Q-Mamba SmoothQuant reproduction addons

This package adds SmoothQuant-style W8A8 fake quantization that composes with:

- DSQ state fake quantization (`--state_quant dsq`)
- ESR checkpoints (`--checkpoint ...`)
- future FP8/FP4 state experiments

It is accuracy-side fake quantization, not an INT8 GEMM latency implementation.

## Install

From your current project root containing `common.py`, `qmamba/quant/state_quant.py`, and `qmamba/patch/patch_mamba2_state_quant.py`:

```bash
unzip qmamba_smoothquant_addons.zip
cp qmamba_smoothquant_addons/qmamba/quant/smoothquant.py qmamba/quant/
cp qmamba_smoothquant_addons/scripts/*.py scripts/
export PYTHONPATH=$PWD
```

## 1. Collect SmoothQuant scales

Use the same calibration tokens you used for ESR if available.

```bash
PYTHONPATH=$PWD python scripts/collect_smoothquant_scales.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype bfloat16 \
  --calib data/calib/redpajama_128x2048.pt \
  --num_samples 128 \
  --seq_len 2048 \
  --target mixer \
  --out data/sq/mamba2_370m_rp128x2048_mixer.pt
```

If using ESR checkpoint before W8A8 calibration:

```bash
PYTHONPATH=$PWD python scripts/collect_smoothquant_scales.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype bfloat16 \
  --calib data/calib/redpajama_128x2048.pt \
  --checkpoint results/phase2_xxx/mamba2_370m_esr_parallel.pt \
  --num_samples 128 \
  --seq_len 2048 \
  --target mixer \
  --out data/sq/mamba2_370m_esr_rp128x2048_mixer.pt
```

## 2. W8A8-only full-forward PPL

This is fast and should be tested first.

```bash
PYTHONPATH=$PWD python scripts/eval_ppl_smoothquant.py \
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
  --out_dir results/smoothquant
```

Sweep alpha if needed: `0.3, 0.4, 0.5, 0.6, 0.7`.

## 3. W8A8 + H4 DSQ recurrent PPL

```bash
PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_smoothquant.py \
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
  --state_quant dsq \
  --state_bits 4 \
  --out_dir results/smoothquant
```

## 4. W8A8 + H4 DSQ + ESR checkpoint

```bash
PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_smoothquant.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 512 \
  --stride 512 \
  --batch_size 1 \
  --max_windows 8 \
  --checkpoint results/phase2_xxx/mamba2_370m_esr_parallel.pt \
  --sq_scales data/sq/mamba2_370m_esr_rp128x2048_mixer.pt \
  --sq_alpha 0.5 \
  --sq_target mixer \
  --state_quant dsq \
  --state_bits 4 \
  --out_dir results/smoothquant
```

## Notes

Default target is `mixer`: Mamba2 mixer `in_proj` and `out_proj` only. It excludes `lm_head`.
This is usually the safest reproduction target first.

This implementation uses:

- activation quantization: dynamic symmetric per-token A8 over last dimension
- weight quantization: static symmetric per-output-channel W8
- SmoothQuant scale: `s = max|X|^alpha / max|W|^(1-alpha)`

It is fake quant for accuracy comparison. Do not compare its latency to the paper's CUDA INT8 GEMM implementation.
