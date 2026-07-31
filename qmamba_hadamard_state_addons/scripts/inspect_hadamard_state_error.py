#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch
from datasets import load_dataset

from common import get_device, load_lm, load_tokenizer, str2dtype
from qmamba.quant.hadamard_state import quantize_state_hadamard, estimate_scale_count


def parse_args():
    p = argparse.ArgumentParser(description="Capture Mamba2 states and report Hadamard-enhanced quantization error.")
    p.add_argument("--model", type=str, default="state-spaces/mamba2-370m")
    p.add_argument("--loader", choices=["mamba_ssm"], default="mamba_ssm")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--step", type=int, default=128)
    p.add_argument("--method", type=str, default="fp8_e4m3_scaled_tensor")
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--hadamard_dims", type=str, default="state")
    p.add_argument("--state_channel_dim", type=int, default=-2)
    p.add_argument("--state_state_dim", type=int, default=-1)
    p.add_argument("--qrange", choices=["symmetric", "full"], default="symmetric")
    p.add_argument("--channel_stat", choices=["sqrt_mean_abs", "rms", "mean_abs"], default="sqrt_mean_abs")
    p.add_argument("--hadamard_compute_dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    p.add_argument("--max_layers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = str2dtype(args.dtype)
    h_dtype = str2dtype(args.hadamard_compute_dtype)

    tok = load_tokenizer(args.model, None, args.loader)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    tokens = tok(text, return_tensors="pt").input_ids[:, : args.seq_len + 1].to(device)
    model = load_lm(args.model, args.loader, dtype, device)
    model.eval()

    from mamba_ssm.utils.generation import InferenceParams

    ip = InferenceParams(max_seqlen=args.seq_len + 1, max_batch_size=1, seqlen_offset=0)
    with torch.no_grad():
        for t in range(args.step):
            _ = model(tokens[:, t:t + 1], inference_params=ip, num_last_tokens=1)
            ip.seqlen_offset += 1

    states = []
    for k, v in ip.key_value_memory_dict.items():
        if isinstance(v, tuple) and len(v) >= 2:
            states.append((k, v[1]))
    if not states:
        raise RuntimeError("No ssm_state found in inference cache.")

    print(f"method={args.method}, bits={args.bits}, hadamard_dims={args.hadamard_dims}")
    print(f"captured {len(states)} layer states at step={args.step}")
    for layer, h in states[: args.max_layers]:
        q = quantize_state_hadamard(
            h,
            method=args.method,
            bits=args.bits,
            hadamard_dims=args.hadamard_dims,
            channel_dim=args.state_channel_dim,
            state_dim=args.state_state_dim,
            qrange=args.qrange,
            channel_stat=args.channel_stat,
            hadamard_compute_dtype=h_dtype,
        )
        err = q.float() - h.float()
        h_rms = h.float().pow(2).mean().sqrt().clamp_min(1e-12)
        rel = err.pow(2).mean().sqrt() / h_rms
        scale_count = estimate_scale_count(tuple(h.shape), args.method, args.hadamard_dims, channel_dim=args.state_channel_dim, state_dim=args.state_state_dim)
        print(
            f"layer={layer} shape={tuple(h.shape)} scales={scale_count} "
            f"h_absmax={float(h.float().abs().max()):.4e} h_rms={float(h_rms):.4e} "
            f"mse={float(err.pow(2).mean()):.4e} rel_rmse={float(rel):.4e} "
            f"q_absmax={float(q.float().abs().max()):.4e} max_abs_err={float(err.abs().max()):.4e}"
        )


if __name__ == "__main__":
    main()
