# Q-Mamba Phase 1 addons

Copy these files into the existing `qmamba_phase0/` root:

```bash
cp -r qmamba scripts /path/to/qmamba_phase0/
```

Then run from the project root.

## Quick patch sanity test

```bash
python scripts/benchmark_generate_phase1.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --batch_size 8 \
  --prompt_len 32 \
  --gen_len 16 \
  --repeat 1 \
  --state_quant dsq \
  --state_bits 4 \
  --out_dir results/phase1
```

## Recurrent PPL smoke tests

```bash
python scripts/eval_ppl_recurrent.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 512 \
  --stride 512 \
  --batch_size 1 \
  --max_windows 2 \
  --state_quant none \
  --out_dir results/phase1

python scripts/eval_ppl_recurrent.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 512 \
  --stride 512 \
  --batch_size 1 \
  --max_windows 2 \
  --state_quant dsq \
  --state_bits 4 \
  --out_dir results/phase1
```

After smoke tests, use `seq_len=2048`, `stride=2048`.
