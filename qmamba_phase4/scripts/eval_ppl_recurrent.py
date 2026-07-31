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
    p = argparse.ArgumentParser(description="Recurrent-mode PPL evaluator for Phase-1 state quantization.")
    p.add_argument("--model", type=str, default="state-spaces/mamba2-370m")
    p.add_argument("--loader", choices=["mamba_ssm"], default="mamba_ssm")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--dataset", choices=["wikitext2", "text"], default="wikitext2")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--text_file", type=str, default=None)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--stride", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--max_windows", type=int, default=0)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--state_quant", choices=["none", "tensor", "channel", "state", "dsq"], default="none")
    p.add_argument("--state_bits", type=int, default=4)
    p.add_argument("--state_quant_position", choices=["before", "after", "both"], default="before")
    p.add_argument("--out_dir", type=str, default="results/phase1")
    p.add_argument("--log_name", type=str, default="recurrent_ppl_runs")
    p.add_argument("--allow_tf32", action="store_true")
    return p.parse_args()


def load_text(args):
    if args.dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
        return "\n\n".join(t for t in ds["text"] if t and t.strip())
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
    tokens = tokenizer(text, return_tensors="pt").input_ids.squeeze(0).contiguous()
    if args.max_tokens and args.max_tokens > 0:
        tokens = tokens[: args.max_tokens]
    starts = make_windows(tokens, args.seq_len, args.stride, args.max_windows)
    if len(starts) == 0:
        raise RuntimeError(f"Not enough tokens ({tokens.numel()}) for seq_len={args.seq_len}")

    model = load_lm(args.model, args.loader, dtype, device)

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

    if args.state_quant != "none":
        from qmamba.patch.patch_mamba2_state_quant import patch_mamba2_step_state_quant
        model = patch_mamba2_step_state_quant(
            model,
            method=args.state_quant,
            bits=args.state_bits,
            quant_position=args.state_quant_position,
        )

    try:
        from mamba_ssm.utils.generation import InferenceParams
    except Exception as e:
        raise RuntimeError("Could not import InferenceParams from mamba_ssm.utils.generation") from e

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()

    with torch.inference_mode():
        for i in tqdm(range(0, len(starts), args.batch_size), desc="Recurrent PPL windows"):
            ss = starts[i : i + args.batch_size]
            batch = torch.stack([tokens[s : s + args.seq_len + 1] for s in ss], dim=0).to(device)
            bsz = batch.shape[0]

            inference_params = InferenceParams(
                max_seqlen=args.seq_len + 1,
                max_batch_size=bsz,
                seqlen_offset=0,
            )

            # Predict token t+1 from prefix ending at token t, token-by-token.
            for t in range(args.seq_len):
                input_t = batch[:, t : t + 1]
                label_t = batch[:, t + 1]
                out = model(input_t, inference_params=inference_params, num_last_tokens=1)
                logits = extract_logits(out)
                logits_t = logits[:, -1, :].float()
                loss = F.cross_entropy(logits_t, label_t, reduction="sum")
                total_nll += float(loss.detach().cpu())
                total_tokens += int(label_t.numel())
                inference_params.seqlen_offset += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    avg_nll = total_nll / max(1, total_tokens)
    ppl = math.exp(avg_nll) if avg_nll < 80 else float("inf")
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0

    record = base_record("eval_ppl_recurrent.py")
    record.update({
        "model": args.model,
        "checkpoint": args.checkpoint, # for phase 2 only
        "loader": args.loader,
        "tokenizer": args.tokenizer or "EleutherAI/gpt-neox-20b",
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
        "state_quant": args.state_quant,
        "state_bits": args.state_bits,
        "state_quant_position": args.state_quant_position,
        "recurrent_eval": True,
    })
    append_jsonl(jsonl_path, record)
    update_csv_from_jsonl(jsonl_path, csv_path)

    print("\n=== Phase-1 recurrent PPL result ===")
    for k in ["model", "dataset", "dtype", "seq_len", "batch_size", "num_windows", "state_quant", "state_bits", "state_quant_position", "ppl", "avg_nll", "elapsed_s", "tokens_per_s", "peak_allocated_gb", "peak_reserved_gb"]:
        print(f"{k}: {record[k]}")
    print(f"Logged JSONL: {jsonl_path}")
    print(f"Logged CSV:   {csv_path}")


if __name__ == "__main__":
    main()
