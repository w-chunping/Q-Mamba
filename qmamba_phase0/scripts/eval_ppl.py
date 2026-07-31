#!/usr/bin/env python3
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

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
    p = argparse.ArgumentParser(description="Phase-0 perplexity evaluator for Mamba/Mamba2 LMs.")
    p.add_argument("--model", type=str, default="state-spaces/mamba2-370m")
    p.add_argument("--loader", choices=["mamba_ssm", "hf"], default="mamba_ssm")
    p.add_argument("--tokenizer", type=str, default=None, help="Usually EleutherAI/gpt-neox-20b for state-spaces Mamba models.")
    p.add_argument("--dataset", choices=["wikitext2", "c4", "text"], default="wikitext2")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--text_file", type=str, default=None)
    p.add_argument("--c4_docs", type=int, default=512, help="Number of C4 validation docs to stream when --dataset c4.")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--stride", type=int, default=2048, help="Use seq_len for non-overlap; smaller stride gives sliding-window PPL.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=0, help="0 = all available tokens.")
    p.add_argument("--max_windows", type=int, default=0, help="0 = all windows.")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--warmup_batches", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="results/phase0")
    p.add_argument("--log_name", type=str, default="ppl_runs")
    p.add_argument("--allow_tf32", action="store_true")
    return p.parse_args()


def load_text(args):
    if args.dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
        return "\n\n".join(t for t in ds["text"] if t and t.strip())
    if args.dataset == "c4":
        # Full C4 validation is large; stream a fixed number of documents for Phase 0.
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for ex in ds:
            txt = ex.get("text", "")
            if txt and txt.strip():
                texts.append(txt)
            if len(texts) >= args.c4_docs:
                break
        return "\n\n".join(texts)
    if args.dataset == "text":
        if not args.text_file:
            raise ValueError("--text_file is required when --dataset text")
        return Path(args.text_file).read_text(encoding="utf-8")
    raise ValueError(args.dataset)


def make_windows(tokens: torch.Tensor, seq_len: int, stride: int, max_windows: int):
    starts = list(range(0, max(0, tokens.numel() - seq_len - 1), stride))
    if max_windows and max_windows > 0:
        starts = starts[:max_windows]
    return starts


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
    text = load_text(args)
    enc = tokenizer(text, return_tensors="pt")
    tokens = enc.input_ids.squeeze(0).contiguous()
    if args.max_tokens and args.max_tokens > 0:
        tokens = tokens[: args.max_tokens]
    starts = make_windows(tokens, args.seq_len, args.stride, args.max_windows)
    if len(starts) == 0:
        raise RuntimeError(f"Not enough tokens ({tokens.numel()}) for seq_len={args.seq_len}")

    model = load_lm(args.model, args.loader, dtype, device)

    # Warmup on the first few windows.
    with torch.inference_mode():
        for s in starts[: args.warmup_batches]:
            batch = tokens[s : s + args.seq_len + 1].unsqueeze(0).to(device)
            _ = model(batch[:, :-1])
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in tqdm(range(0, len(starts), args.batch_size), desc="PPL windows"):
            ss = starts[i : i + args.batch_size]
            batch = torch.stack([tokens[s : s + args.seq_len + 1] for s in ss], dim=0).to(device)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            output = model(input_ids)
            logits = extract_logits(output)
            if logits.shape[1] != labels.shape[1]:
                logits = logits[:, -labels.shape[1] :, :]
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss.detach().cpu())
            total_tokens += int(labels.numel())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    avg_nll = total_nll / max(1, total_tokens)
    ppl = math.exp(avg_nll) if avg_nll < 80 else float("inf")
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0

    record = base_record("eval_ppl.py")
    record.update({
        "model": args.model,
        "loader": args.loader,
        "tokenizer": args.tokenizer or ("EleutherAI/gpt-neox-20b" if args.loader == "mamba_ssm" else args.model),
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
        "tokens_per_s": total_tokens / elapsed if elapsed > 0 else 0.0,
        "peak_allocated_gb": peak_allocated,
        "peak_reserved_gb": peak_reserved,
        "max_tokens": args.max_tokens,
        "max_windows": args.max_windows,
        "allow_tf32": bool(args.allow_tf32),
    })
    append_jsonl(jsonl_path, record)
    update_csv_from_jsonl(jsonl_path, csv_path)
    print("\n=== Phase-0 PPL result ===")
    for k in ["model", "dataset", "dtype", "seq_len", "batch_size", "num_windows", "ppl", "avg_nll", "elapsed_s", "tokens_per_s", "peak_allocated_gb", "peak_reserved_gb"]:
        print(f"{k}: {record[k]}")
    print(f"Logged JSONL: {jsonl_path}")
    print(f"Logged CSV:   {csv_path}")


if __name__ == "__main__":
    main()
