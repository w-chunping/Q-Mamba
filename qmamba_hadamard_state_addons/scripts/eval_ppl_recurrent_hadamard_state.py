#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from common import append_jsonl, base_record, extract_logits, get_device, load_lm, load_tokenizer, str2dtype, update_csv_from_jsonl
from qmamba.patch.patch_mamba2_hadamard_state import patch_mamba2_step_hadamard_state_quant


def parse_args():
    p = argparse.ArgumentParser(description="Recurrent PPL with Hadamard-enhanced INT/FP8 state quantization.")
    p.add_argument("--model", type=str, default="state-spaces/mamba2-370m")
    p.add_argument("--loader", choices=["mamba_ssm"], default="mamba_ssm")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--dataset", choices=["wikitext2", "text"], default="wikitext2")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--text_file", type=str, default=None)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--stride", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_windows", type=int, default=0)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--checkpoint", type=str, default=None, help="Optional ESR/state_dict checkpoint")

    p.add_argument("--state_quant", type=str, default="none", help="INT: tensor/head/channel/state/dsq. FP8: fp8_e4m3_scaled_tensor, fp8_e4m3_dsq, ...")
    p.add_argument("--state_bits", type=int, default=8, help="Used for INT methods only")
    p.add_argument("--hadamard_dims", type=str, default="none", help="none/state/channel/both or raw dim list like -1")
    p.add_argument("--state_quant_position", choices=["before", "after", "both"], default="before")
    p.add_argument("--state_channel_dim", type=int, default=-2)
    p.add_argument("--state_state_dim", type=int, default=-1)
    p.add_argument("--qrange", choices=["symmetric", "full"], default="symmetric")
    p.add_argument("--channel_stat", choices=["sqrt_mean_abs", "rms", "mean_abs"], default="sqrt_mean_abs")
    p.add_argument("--hadamard_compute_dtype", choices=["float16", "bfloat16", "float32"], default="float32")

    # Optional SmoothQuant compatibility if installed.
    p.add_argument("--sq_scales", type=str, default=None)
    p.add_argument("--sq_alpha", type=float, default=0.5)
    p.add_argument("--sq_target", choices=["mixer", "in_proj", "out_proj", "all"], default="mixer")
    p.add_argument("--act_bits", type=int, default=8)
    p.add_argument("--weight_bits", type=int, default=8)
    p.add_argument("--disable_act_quant", action="store_true")
    p.add_argument("--disable_weight_quant", action="store_true")

    p.add_argument("--out_dir", type=str, default="results/hadamard_state")
    p.add_argument("--log_name", type=str, default="recurrent_ppl_hadamard_state_runs")
    return p.parse_args()


def load_text(args):
    if args.dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
        return "\n\n".join(t for t in ds["text"] if t and t.strip())
    if args.dataset == "text":
        if args.text_file is None:
            raise ValueError("--text_file is required for --dataset text")
        return Path(args.text_file).read_text(encoding="utf-8")
    raise ValueError(args.dataset)


def make_windows(tokens, seq_len, stride, max_windows):
    starts = list(range(0, max(0, tokens.numel() - seq_len - 1), stride))
    if max_windows and max_windows > 0:
        starts = starts[:max_windows]
    return starts


def load_optional_checkpoint(model, path):
    if path is None:
        return 0, 0
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    return len(missing), len(unexpected)


def apply_optional_smoothquant(model, args):
    if args.sq_scales is None:
        return 0
    from qmamba.quant.smoothquant import apply_smoothquant_fake, load_sq_scales
    scales, meta = load_sq_scales(args.sq_scales)
    print(f"Loaded SQ scales from {args.sq_scales}; meta={meta}")
    return apply_smoothquant_fake(
        model,
        scales,
        alpha=args.sq_alpha,
        act_bits=args.act_bits,
        weight_bits=args.weight_bits,
        target=args.sq_target,
        quantize_activation=not args.disable_act_quant,
        quantize_weight=not args.disable_weight_quant,
    )


def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = str2dtype(args.dtype)
    h_dtype = str2dtype(args.hadamard_compute_dtype)
    out_dir = Path(args.out_dir)
    jsonl_path = out_dir / f"{args.log_name}.jsonl"
    csv_path = out_dir / f"{args.log_name}.csv"

    tok = load_tokenizer(args.model, args.tokenizer, args.loader)
    tokens = tok(load_text(args), return_tensors="pt").input_ids.squeeze(0).contiguous()
    starts = make_windows(tokens, args.seq_len, args.stride, args.max_windows)
    print(f"tokens={tokens.numel()}, windows={len(starts)}, seq_len={args.seq_len}")

    model = load_lm(args.model, args.loader, dtype, device)
    missing, unexpected = load_optional_checkpoint(model, args.checkpoint)
    sq_modules = apply_optional_smoothquant(model, args)

    if args.state_quant != "none" or args.hadamard_dims not in ("none", "off", ""):
        model = patch_mamba2_step_hadamard_state_quant(
            model,
            method=args.state_quant,
            bits=args.state_bits,
            hadamard_dims=args.hadamard_dims,
            quant_position=args.state_quant_position,
            channel_dim=args.state_channel_dim,
            state_dim=args.state_state_dim,
            qrange=args.qrange,
            channel_stat=args.channel_stat,
            hadamard_compute_dtype=h_dtype,
        )
    model.eval()

    from mamba_ssm.utils.generation import InferenceParams

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for i in tqdm(range(0, len(starts), args.batch_size), desc="Recurrent PPL windows"):
            ss = starts[i:i + args.batch_size]
            batch = torch.stack([tokens[s:s + args.seq_len + 1] for s in ss], dim=0).to(device)
            bsz = batch.shape[0]
            inference_params = InferenceParams(max_seqlen=args.seq_len + 1, max_batch_size=bsz, seqlen_offset=0)
            for t in range(args.seq_len):
                out = model(batch[:, t:t + 1], inference_params=inference_params, num_last_tokens=1)
                logits = extract_logits(out)[:, -1, :].float()
                labels = batch[:, t + 1]
                loss = F.cross_entropy(logits, labels, reduction="sum")
                total_nll += float(loss.cpu())
                total_tokens += labels.numel()
                inference_params.seqlen_offset += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll) if avg_nll < 80 else float("inf")

    record = base_record("eval_ppl_recurrent_hadamard_state.py")
    record.update({
        "model": args.model,
        "loader": args.loader,
        "dataset": args.dataset,
        "split": args.split,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "num_windows": len(starts),
        "total_tokens": total_tokens,
        "avg_nll": avg_nll,
        "ppl": ppl,
        "elapsed_s": elapsed,
        "tokens_per_s": total_tokens / elapsed,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0,
        "checkpoint": args.checkpoint,
        "state_quant": args.state_quant,
        "state_bits": args.state_bits,
        "hadamard_dims": args.hadamard_dims,
        "state_quant_position": args.state_quant_position,
        "state_channel_dim": args.state_channel_dim,
        "state_state_dim": args.state_state_dim,
        "qrange": args.qrange,
        "channel_stat": args.channel_stat,
        "hadamard_compute_dtype": args.hadamard_compute_dtype,
        "sq_scales": args.sq_scales,
        "sq_modules": sq_modules,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    })
    append_jsonl(jsonl_path, record)
    update_csv_from_jsonl(jsonl_path, csv_path)

    print("\n=== Hadamard State recurrent PPL result ===")
    for k in [
        "model", "dataset", "dtype", "seq_len", "batch_size", "num_windows",
        "state_quant", "state_bits", "hadamard_dims", "state_quant_position", "ppl",
        "avg_nll", "elapsed_s", "tokens_per_s", "peak_allocated_gb", "peak_reserved_gb", "sq_modules",
    ]:
        print(f"{k}: {record[k]}")
    print(f"Logged JSONL: {jsonl_path}")
    print(f"Logged CSV:   {csv_path}")


if __name__ == "__main__":
    main()
