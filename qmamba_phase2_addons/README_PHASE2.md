# Q-Mamba Phase 2 addons: ESR-lite code

This package adds a first ESR/block-reconstruction implementation.

It is intentionally called **ESR-lite** because it is a practical first implementation:

- It reconstructs the **Mamba2 mixer output** block by block.
- It trains B/C/Delta slices of the fused `in_proj` using gradient masks.
- It uses a custom PyTorch reference Mamba2 mixer forward with state fake quantization.
- `--qmode esr` implements the paper-style one-step quantization simulation:
  `qstate_t = dA_t * Q(fp_state_{t-1}) + dBx_t`.

It is not yet the final optimized paper implementation. It is for verifying that ESR-style reconstruction
can reduce DSQ H4 loss before we implement a faster/cleaner version.

## Copy into your phase1/project root

From the project root that already has `common.py` and `qmamba/quant/state_quant.py`:

```bash
unzip qmamba_phase2_addons.zip
cp -r qmamba_phase2_addons/qmamba/esr qmamba/
cp qmamba_phase2_addons/scripts/*.py scripts/
export PYTHONPATH=$PWD
```

## 0. Inspect slices

```bash
PYTHONPATH=$PWD python scripts/inspect_mamba2_slices.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16
```

## 1. Prepare small calibration data first

Debug size:

```bash
PYTHONPATH=$PWD python scripts/prepare_calib.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dataset wikitext2 \
  --num_samples 8 \
  --seq_len 256 \
  --out data/calib/wikitext2_8x256.pt
```

Paper-like size later:

```bash
PYTHONPATH=$PWD python scripts/prepare_calib.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dataset wikitext2 \
  --num_samples 128 \
  --seq_len 2048 \
  --out data/calib/wikitext2_128x2048.pt
```

## 2. Debug ESR on only one block

```bash
PYTHONPATH=$PWD python scripts/run_esr_mixer.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --calib data/calib/wikitext2_8x256.pt \
  --num_samples 2 \
  --seq_len 128 \
  --epochs 1 \
  --block_start 0 \
  --block_end 1 \
  --state_quant dsq \
  --state_bits 4 \
  --qmode esr \
  --train_mode sga \
  --out_dir results/phase2_debug
```

You want to see the reconstruction MSE decrease or at least run stably.

## 3. Larger debug run

```bash
PYTHONPATH=$PWD python scripts/run_esr_mixer.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --calib data/calib/wikitext2_8x256.pt \
  --num_samples 8 \
  --seq_len 256 \
  --epochs 1 \
  --block_start 0 \
  --block_end 4 \
  --state_quant dsq \
  --state_bits 4 \
  --qmode esr \
  --train_mode sga \
  --out_dir results/phase2_debug
```

## 4. Train modes

- `sga`: B/C/Delta rows of in_proj + dt_bias.
- `sga_conv`: SGA + conv1d parameters.
- `sga_conv_d`: SGA + conv1d + D.
- `all_mixer`: train the whole mixer, mainly for debugging/upper bound.

## 5. After ESR

The saved file is a `state_dict`, e.g.

```text
results/phase2/mamba2_370m_esr_lite.pt
```

Next we will add a loader option to `eval_ppl_recurrent.py` to load this state_dict and compare:

- DSQ only
- DSQ + ESR-lite

against the paper's DSQ-only and DSQ+ESR trend.
