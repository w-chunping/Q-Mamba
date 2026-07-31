#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset

from common import (
    append_jsonl,
    base_record,
    extract_logits,
    get_device,
    load_lm,
    load_tokenizer,
    str2dtype,
    update_csv_from_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase-0 generation benchmark for Mamba/Mamba2 LMs.")
    p.add_argument("--model", type=str, default="state-spaces/mamba2-370m")
    p.add_argument("--loader", choices=["mamba_ssm", "hf"], default="mamba_ssm")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--prompt_len", type=int, default=100)
    p.add_argument("--gen_len", type=int, default=100)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="results/phase0")
    p.add_argument("--log_name", type=str, default="generate_runs")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--cg", action="store_true", help="Try CUDA graph generation for mamba_ssm if supported.")
    return p.parse_args()


def get_prompt_batch(tokenizer, prompt_len: int, batch_size: int) -> torch.Tensor:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
    needed = prompt_len * batch_size
    if ids.numel() < needed + 1:
        reps = (needed + 1 + ids.numel() - 1) // ids.numel()
        ids = ids.repeat(reps)
    chunks = [ids[i * prompt_len : (i + 1) * prompt_len] for i in range(batch_size)]
    return torch.stack(chunks, dim=0).contiguous()


def model_generate(model, input_ids, gen_len: int, loader: str, cg: bool):
    # Try HF-style first. If native mamba-ssm rejects kwargs, fall back to max_length style.
    try:
        return model.generate(
            input_ids=input_ids,
            max_new_tokens=gen_len,
            do_sample=False,
            use_cache=True,
            pad_token_id=None,
        )
    except TypeError:
        kwargs = {
            "input_ids": input_ids,
            "max_length": input_ids.shape[1] + gen_len,
            "top_k": 1,
            "top_p": 0.0,
        }
        if loader == "mamba_ssm":
            kwargs["cg"] = cg
        return model.generate(**kwargs)


def timed_forward(model, input_ids, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids)
        logits = extract_logits(out)
        _ = logits[:, -1, :].argmax(dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def timed_generate(model, input_ids, gen_len, loader, cg, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model_generate(model, input_ids, gen_len, loader, cg)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, out


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    device = get_device(args.device)
    dtype = str2dtype(args.dtype)
    out_dir = Path(args.out_dir)
    jsonl_path = out_dir / f"{args.log_name}.jsonl"
    csv_path = out_dir / f"{args.log_name}.csv"

    tokenizer = load_tokenizer(args.model, args.tokenizer, args.loader)
    model = load_lm(args.model, args.loader, dtype, device)
    input_ids = get_prompt_batch(tokenizer, args.prompt_len, args.batch_size).to(device)

    # Warmup
    for _ in range(args.warmup):
        _ = timed_forward(model, input_ids, device)
        _ = timed_generate(model, input_ids, args.gen_len, args.loader, args.cg, device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    prefill_times = []
    total_times = []
    generated_shape = None
    for _ in range(args.repeat):
        prefill_times.append(timed_forward(model, input_ids, device))
        total_ms, out = timed_generate(model, input_ids, args.gen_len, args.loader, args.cg, device)
        total_times.append(total_ms)
        if hasattr(out, "sequences"):
            generated_shape = tuple(out.sequences.shape)
        elif isinstance(out, torch.Tensor):
            generated_shape = tuple(out.shape)
        else:
            generated_shape = str(type(out))

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0
    prefill_ms = sum(prefill_times) / len(prefill_times)
    total_generate_ms = sum(total_times) / len(total_times)
    approx_decode_ms = max(0.0, total_generate_ms - prefill_ms)
    generated_tokens = args.batch_size * args.gen_len

    record = base_record("benchmark_generate.py")
    record.update({
        "model": args.model,
        "loader": args.loader,
        "tokenizer": args.tokenizer or ("EleutherAI/gpt-neox-20b" if args.loader == "mamba_ssm" else args.model),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "prefill_ms": prefill_ms,
        "total_generate_ms": total_generate_ms,
        "approx_decode_ms": approx_decode_ms,
        "generated_tokens": generated_tokens,
        "decode_tokens_per_s": generated_tokens / (approx_decode_ms / 1000.0) if approx_decode_ms > 0 else 0.0,
        "end_to_end_tokens_per_s": generated_tokens / (total_generate_ms / 1000.0) if total_generate_ms > 0 else 0.0,
        "peak_allocated_gb": peak_allocated,
        "peak_reserved_gb": peak_reserved,
        "generated_shape": generated_shape,
        "allow_tf32": bool(args.allow_tf32),
        "cg": bool(args.cg),
    })
    append_jsonl(jsonl_path, record)
    update_csv_from_jsonl(jsonl_path, csv_path)
    print("\n=== Phase-0 generation benchmark ===")
    for k in ["model", "dtype", "batch_size", "prompt_len", "gen_len", "prefill_ms", "total_generate_ms", "approx_decode_ms", "decode_tokens_per_s", "peak_allocated_gb", "peak_reserved_gb"]:
        print(f"{k}: {record[k]}")
    print(f"Logged JSONL: {jsonl_path}")
    print(f"Logged CSV:   {csv_path}")


if __name__ == "__main__":
    main()
