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

    # Optional: dump recurrent SSM states for offline quantization-error analysis.
    # For one-step FP8 / Hadamard+FP8 / DSQ-INT4 error comparison, run this script
    # with --state_quant none --hadamard_dims none, then feed the dump to
    # scripts/analyze_state_quant_error.py.
    p.add_argument("--dump_states", type=str, default=None,
                   help="Optional output .pt path for sampled recurrent SSM states.")
    p.add_argument("--dump_state_every", type=int, default=128,
                   help="Save one state snapshot every N recurrent steps.")
    p.add_argument("--dump_state_max_steps", type=int, default=16,
                   help="Maximum number of state snapshots to save in total.")
    p.add_argument("--dump_state_max_windows", type=int, default=1,
                   help="Only dump states for the first K evaluated batches/windows. Use 0 for no limit.")
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




def maybe_dump_state_snapshot(inference_params, state_snapshots, max_steps: int):
    """
    Extract Mamba/Mamba2 recurrent SSM states from InferenceParams.

    In mamba_ssm, inference_params.key_value_memory_dict usually maps
      layer_idx -> (conv_state, ssm_state)
    where ssm_state has shape like [B, H, P, N].
    """
    if max_steps > 0:
        # Count snapshots by looking at the first non-empty layer.
        for snapshots in state_snapshots.values():
            if len(snapshots) >= max_steps:
                return
            break

    cache = getattr(inference_params, "key_value_memory_dict", None)
    if not cache:
        return

    for layer_idx, layer_cache in sorted(cache.items(), key=lambda kv: kv[0]):
        ssm_state = None

        if isinstance(layer_cache, (tuple, list)) and len(layer_cache) >= 2:
            # Mamba/Mamba2 common format: (conv_state, ssm_state).
            ssm_state = layer_cache[1]
        elif isinstance(layer_cache, dict):
            # More defensive fallback in case your patched code uses named fields.
            for key in ("ssm_state", "state", "states"):
                if key in layer_cache and isinstance(layer_cache[key], torch.Tensor):
                    ssm_state = layer_cache[key]
                    break

        if isinstance(ssm_state, torch.Tensor):
            state_snapshots.setdefault(int(layer_idx), []).append(
                ssm_state.detach().float().cpu()
            )


def save_state_dump(state_snapshots, out_path: str, meta: dict):
    """Save sampled states as {"states": [...], "meta": {...}}."""
    if not state_snapshots:
        print("[state dump] no states were collected; nothing saved")
        return

    layers = []
    layer_indices = []
    for layer_idx in sorted(state_snapshots.keys()):
        snapshots = state_snapshots[layer_idx]
        if len(snapshots) == 0:
            continue
        # [T, B, H, P, N] if each snapshot is [B, H, P, N].
        layers.append(torch.stack(snapshots, dim=0))
        layer_indices.append(layer_idx)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "states": layers,
        "layer_indices": layer_indices,
        "meta": meta,
    }, out_path)
    print(f"[state dump] saved {len(layers)} layers to {out_path}")


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

    state_snapshots = {}
    dump_steps_collected = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(starts), args.batch_size), desc="Recurrent PPL windows"):
            ss = starts[i:i + args.batch_size]
            batch = torch.stack([tokens[s:s + args.seq_len + 1] for s in ss], dim=0).to(device)
            bsz = batch.shape[0]
            inference_params = InferenceParams(max_seqlen=args.seq_len + 1, max_batch_size=bsz, seqlen_offset=0)
            for t in range(args.seq_len):
                out = model(batch[:, t:t + 1], inference_params=inference_params, num_last_tokens=1)

                if args.dump_states is not None:
                    window_batch_idx = i // args.batch_size
                    allow_window = (args.dump_state_max_windows <= 0
                                    or window_batch_idx < args.dump_state_max_windows)
                    allow_step = (args.dump_state_every > 0
                                  and (t % args.dump_state_every == 0)
                                  and (args.dump_state_max_steps <= 0
                                       or dump_steps_collected < args.dump_state_max_steps))
                    if allow_window and allow_step:
                        maybe_dump_state_snapshot(
                            inference_params,
                            state_snapshots,
                            max_steps=args.dump_state_max_steps,
                        )
                        dump_steps_collected += 1

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

    if args.dump_states is not None:
        save_state_dump(
            state_snapshots,
            args.dump_states,
            meta={
                "script": "eval_ppl_recurrent_hadamard_state.py",
                "model": args.model,
                "dataset": args.dataset,
                "split": args.split,
                "dtype": args.dtype,
                "seq_len": args.seq_len,
                "stride": args.stride,
                "batch_size": args.batch_size,
                "num_windows": len(starts),
                "dump_state_every": args.dump_state_every,
                "dump_state_max_steps": args.dump_state_max_steps,
                "dump_state_max_windows": args.dump_state_max_windows,
                "state_quant": args.state_quant,
                "state_bits": args.state_bits,
                "hadamard_dims": args.hadamard_dims,
                "state_quant_position": args.state_quant_position,
            },
        )

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
        "dump_states": args.dump_states,
        "dump_state_every": args.dump_state_every,
        "dump_state_max_steps": args.dump_state_max_steps,
        "dump_state_max_windows": args.dump_state_max_windows,
    })
    append_jsonl(jsonl_path, record)
    update_csv_from_jsonl(jsonl_path, csv_path)

    print("\n=== Hadamard State recurrent PPL result ===")
    for k in [
        "model", "dataset", "dtype", "seq_len", "batch_size", "num_windows",
        "state_quant", "state_bits", "hadamard_dims", "state_quant_position", "ppl",
        "avg_nll", "elapsed_s", "tokens_per_s", "peak_allocated_gb", "peak_reserved_gb", "sq_modules",
        "dump_states",
    ]:
        print(f"{k}: {record[k]}")
    print(f"Logged JSONL: {jsonl_path}")
    print(f"Logged CSV:   {csv_path}")


if __name__ == "__main__":
    main()
