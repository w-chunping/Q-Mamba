# Q-Mamba Phase 0: FP16 reproduction and metric logging

This folder gives a minimal Phase-0 baseline for Q-Mamba-style experiments:

- `scripts/eval_ppl.py`: WikiText2/C4/text-file perplexity evaluator.
- `scripts/benchmark_generate.py`: prompt+decode benchmark with latency, tokens/s, and peak memory logging.
- `results/phase0/*.jsonl` and `*.csv`: appended run logs.

## Install sketch

```bash
conda create -n qmamba python=3.10 -y
conda activate qmamba
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install causal-conv1d mamba-ssm --no-build-isolation
```

If your CUDA/PyTorch version differs, install the matching PyTorch wheel first.

## Run

```bash
cd qmamba_phase0
export HF_HOME=/work/$USER/hf_cache
export HF_DATASETS_CACHE=/work/$USER/hf_datasets
bash run_phase0.sh
```

Or manually:

```bash
python scripts/eval_ppl.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 2048 \
  --stride 2048 \
  --batch_size 1 \
  --out_dir results/phase0

python scripts/benchmark_generate.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --batch_size 32 \
  --prompt_len 100 \
  --gen_len 100 \
  --repeat 5 \
  --out_dir results/phase0
```

## Notes

- The paper reports Mamba2-370M FP16 WikiText2 PPL around 14.16 and C4 PPL around 16.95.
- Exact values can shift because of dataset preprocessing, tokenizer choice, dtype, stride, and whether evaluation uses recurrent or parallel mode.
- For Phase 1, patch the state update path and reuse these exact scripts so the logging format stays stable.
