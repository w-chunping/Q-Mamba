# QMamba

**Quantization experiments for the recurrent hidden state of Mamba2.**

QMamba is a research prototype for studying how low-precision state representations affect autoregressive Mamba2 inference. The repository builds from an FP16 baseline through INT state quantization, ESR-style reconstruction, SmoothQuant W8A8, FP8 state quantization, and Hadamard rotations for suppressing state outliers.

This project builds on [**Q-Mamba: Towards More Efficient Mamba Models via Post-Training Quantization**](https://aclanthology.org/2025.findings-acl.551/) by Chen Tianqi et al. (Findings of ACL 2025), which introduced Decoupled Scale Quantization (DSQ) and Efficient Selectivity Reconstruction (ESR) for quantized Mamba models. This repository explores those ideas and extends the state-quantization study with FP8 formats and Hadamard rotations.

The central finding is that Mamba2's recurrent SSM state is highly sensitive to coarse integer quantization, but much more robust to scaled FP8. A normalized Hadamard transform along the state dimension improves FP8 further by redistributing outliers before quantization.

## Highlights

- Reduced WikiText-2 perplexity from **1476.62 with per-tensor INT8 state quantization to 17.33 with tensor-scaled FP8 E4M3**.
- Improved raw FP8 E4M3 from **37.25 to 17.21** by applying a Hadamard transform along the Mamba2 state dimension, without an external scaling factor.
- Combined the Hadamard transform with tensor-scaled FP8 E4M3 to improve perplexity from **17.33 to 15.85**.
- Implemented configurable INT and FP8 fake quantization, scale granularities, recurrent perplexity evaluation, SmoothQuant W8A8, and ESR-style block reconstruction.

## Results

The table below reports full recurrent evaluation of `state-spaces/mamba2-370m` on the WikiText-2 test set. All runs use FP16 model computation, sequence length and stride 2048, 140 windows, and 286,720 evaluated tokens. Lower perplexity is better.

| Recurrent-state method | External scale | Hadamard over state axis | Perplexity |
|---|---:|---:|---:|
| FP16 state | — | No | **14.02** |
| INT8 | Per tensor | No | 1476.62 |
| FP8 E4M3 | None | No | 37.25 |
| FP8 E4M3 | Per tensor | No | 17.33 |
| FP8 E4M3 | None | Yes | 17.21 |
| FP8 E4M3 | Per tensor | Yes | **15.85** |

The FP8 baseline results are recorded in [`qmamba_phase4/results/fp8_state_sweep_full`](qmamba_phase4/results/fp8_state_sweep_full/recurrent_ppl_fp8_state_runs.csv), and the Hadamard results are recorded in [`qmamba_hadamard_state_addons/results/hadamard_state`](qmamba_hadamard_state_addons/results/hadamard_state/recurrent_ppl_hadamard_state_runs.csv).

> [!NOTE]
> These are accuracy-side fake-quantization experiments. The current PyTorch implementation emulates quantization and dequantization but does not store the state in packed INT8/FP8 form or provide optimized low-precision kernels. Its measured throughput should therefore not be interpreted as quantized inference speedup.

## Method

For each Mamba2 layer, recurrent inference maintains an SSM state with shape `[B, H, P, N]`, where `P` is the per-head channel dimension and `N` is the state dimension. QMamba intercepts the recurrent `step` path and fake-quantizes this state before the next update.

Supported approaches include:

- **Uniform INT quantization:** per-tensor, per-head, per-channel, or per-state scaling.
- **Decoupled Scale Quantization (DSQ):** factorizes the scale across the channel and state axes.
- **FP8 state quantization:** E4M3 and E5M2, either raw or with tensor/head/channel/state scaling and DSQ-like scaling.
- **Hadamard-enhanced quantization:** rotates the state along `N`, `P`, or both axes, quantizes in the transformed domain, and applies the inverse rotation.
- **SmoothQuant W8A8:** fake-quantizes mixer projections with per-token activations and per-output-channel weights.
- **ESR-lite:** reconstructs Mamba2 mixer blocks while simulating quantized recurrent states.

The Hadamard path uses a normalized Fast Walsh-Hadamard Transform:

```text
state -> Hadamard rotation -> INT/FP8 quantization -> inverse rotation
```

Because the normalized transform is orthogonal and self-inverse, it preserves the represented state while spreading large coordinates across the transformed dimension. The current implementation requires transformed dimensions to be powers of two.

## Repository layout

The project is preserved as a sequence of experimental snapshots. For the latest state-quantization evaluator, start with `qmamba_hadamard_state_addons`.

| Directory | Contents |
|---|---|
| [`qmamba_phase0`](qmamba_phase0/) | FP16 perplexity and generation baselines |
| [`qmamba_phase1_addons`](qmamba_phase1_addons/) | INT state fake quantization and recurrent evaluation |
| [`qmamba_phase2_addons`](qmamba_phase2_addons/) | ESR-lite reconstruction and calibration |
| [`qmamba_phase2_parallel_addons`](qmamba_phase2_parallel_addons/) | Vectorized ESR forward implementation |
| [`qmamba_phase3`](qmamba_phase3/) | SmoothQuant W8A8 experiments |
| [`qmamba_phase4`](qmamba_phase4/) | FP8 recurrent-state quantization |
| [`qmamba_fp8_state_addons`](qmamba_fp8_state_addons/) | Standalone FP8 additions for an earlier phase |
| [`qmamba_hadamard_state_addons`](qmamba_hadamard_state_addons/) | Hadamard-enhanced INT/FP8 evaluation and analysis |

Each snapshot contains its own scripts, outputs, and focused documentation. The folders are not installed as one top-level Python package.

## Setup

A CUDA GPU is strongly recommended because recurrent perplexity evaluation processes one token at a time. The committed full runs used Python 3.10.20, PyTorch 2.4.1 with CUDA 12.1, and an NVIDIA H100 80 GB GPU.

```bash
conda create -n qmamba python=3.10 -y
conda activate qmamba

# Change the PyTorch wheel/index to match your CUDA installation.
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r qmamba_phase0/requirements.txt
pip install causal-conv1d mamba-ssm --no-build-isolation
```

Model weights, tokenizers, and WikiText-2 are downloaded through Hugging Face on first use. To place the cache on a larger disk:

```bash
export HF_HOME=/path/to/huggingface/cache
export HF_DATASETS_CACHE=/path/to/huggingface/datasets
```

## Quick start

Run a short eight-window smoke test from the latest experiment directory:

```bash
cd qmamba_hadamard_state_addons

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
  --out_dir results/hadamard_smoke_test
```

Results are appended to both JSONL and CSV files under the selected output directory. Smoke-test perplexity is useful for checking the installation, but it is not directly comparable to the full-run table above.

To run the predefined INT/FP8/Hadamard sweep:

```bash
SEQ_LEN=512 MAX_WINDOWS=8 OUT_DIR=results/hadamard_sweep \
  bash scripts/sweep_hadamard_state_ppl.sh
```

Set `SEQ_LEN=2048` and `MAX_WINDOWS=0` for full WikiText-2 evaluation. This is a long-running sweep; testing promising configurations with the evaluator directly is usually more practical.

## Reproducing a headline configuration

From `qmamba_hadamard_state_addons`, tensor-scaled FP8 E4M3 with a state-axis Hadamard rotation can be evaluated with:

```bash
PYTHONPATH=$PWD python scripts/eval_ppl_recurrent_hadamard_state.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --dataset wikitext2 \
  --seq_len 2048 \
  --stride 2048 \
  --batch_size 1 \
  --max_windows 0 \
  --state_quant fp8_e4m3_scaled_tensor \
  --hadamard_dims state \
  --hadamard_compute_dtype float32 \
  --out_dir results/hadamard_state
```

Change `--hadamard_dims` to `none` for the 17.33 scaled-FP8 baseline, or use `--state_quant fp8_e4m3` to compare raw FP8 with and without the transform.

## Analysis utilities

The latest snapshot also includes scripts for inspecting quantization error and visualizing layer-wise state outliers:

```bash
cd qmamba_hadamard_state_addons

PYTHONPATH=$PWD python scripts/inspect_hadamard_state_error.py \
  --model state-spaces/mamba2-370m \
  --loader mamba_ssm \
  --dtype float16 \
  --seq_len 256 \
  --step 128 \
  --method fp8_e4m3_scaled_tensor \
  --hadamard_dims state
```

See [`README_HADAMARD_STATE.md`](qmamba_hadamard_state_addons/README_HADAMARD_STATE.md) for all Hadamard options, [`README_FP8_STATE.md`](qmamba_phase4/README_FP8_STATE.md) for FP8 scale granularities, and the phase-specific READMEs for the earlier experiments.

## Citation

If you use this repository, please cite the original Q-Mamba paper:

```bibtex
@inproceedings{tianqi-etal-2025-q,
  title     = {{Q}-Mamba: Towards more efficient Mamba models via post-training quantization},
  author    = {Tianqi, Chen and Chen, Yuanteng and Wang, Peisong and
               Xu, Weixiang and Zhu, Zeyu and Cheng, Jian},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  year      = {2025},
  month     = jul,
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
  pages     = {10594--10610},
  url       = {https://aclanthology.org/2025.findings-acl.551/},
  doi       = {10.18653/v1/2025.findings-acl.551}
}
```

## Limitations

- Quantization, SmoothQuant, and Hadamard operations are reference implementations intended for numerical experiments, not production kernels.
- Reported perplexities depend on the exact tokenizer, package versions, precision, dataset preprocessing, and recurrent evaluation protocol.
- The repository contains experiment snapshots rather than a unified package or CLI; run commands from the relevant phase directory with `PYTHONPATH=$PWD`.
- Full token-by-token evaluation is substantially slower than the fused parallel Mamba2 forward path.
