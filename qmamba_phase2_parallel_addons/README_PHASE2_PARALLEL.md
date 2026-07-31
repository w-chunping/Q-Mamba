# Q-Mamba Phase 2.5: Parallel ESR Forward Addons

This package adds a vectorized PyTorch implementation of the ESR one-step state-quantization approximation for Mamba2 mixer reconstruction.

Files:

```text
qmamba/esr/mamba2_parallel_esr.py
qmamba/esr/sga_delta.py
scripts/validate_parallel_esr_forward.py
scripts/run_esr_mixer_parallel.py
```

## Install into your project

From your project root:

```bash
unzip qmamba_phase2_parallel_addons.zip
cp -r qmamba_phase2_parallel_addons/qmamba/esr/* qmamba/esr/
cp qmamba_phase2_parallel_addons/scripts/*.py scripts/
export PYTHONPATH=$PWD
```

## Validate first

```bash
PYTHONPATH=$PWD python scripts/validate_parallel_esr_forward.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype bfloat16 \
  --calib data/calib/wikitext2_8x256.pt \
  --seq_len 128 \
  --block 0
```

The custom `qmode=none` output should be close to the original mixer output. Then run ESR.

## Debug ESR run

```bash
PYTHONPATH=$PWD python scripts/run_esr_mixer_parallel.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype bfloat16 \
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
  --lr 2e-5 \
  --adam_eps 1e-5 \
  --out_dir results/phase2_parallel_debug
```

## Larger but still safe run

```bash
PYTHONPATH=$PWD python scripts/run_esr_mixer_parallel.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype bfloat16 \
  --calib data/calib/wikitext2_8x256.pt \
  --num_samples 8 \
  --seq_len 256 \
  --epochs 1 \
  --block_start 0 \
  --block_end 48 \
  --state_quant dsq \
  --state_bits 4 \
  --qmode esr \
  --train_mode sga \
  --lr 2e-5 \
  --adam_eps 1e-5 \
  --out_dir results/phase2_parallel_48b
```

For `seq_len=512` or larger, add `--scan_chunk_size 128` or `--scan_chunk_size 256` if memory becomes high.
