#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch
from datasets import load_dataset

from common import get_device, load_lm, load_tokenizer, str2dtype
from qmamba.quant.fp8_state import quantize_state_fp8


def parse_args():
    p = argparse.ArgumentParser(description="Capture one recurrent Mamba2 state and report FP8 quantization error statistics.")
    p.add_argument("--model", type=str, default="state-spaces/mamba2-370m")
    p.add_argument("--loader", choices=["mamba_ssm"], default="mamba_ssm")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--step", type=int, default=128)
    p.add_argument("--method", type=str, default="fp8_e4m3_scaled_head")
    p.add_argument("--dataset", choices=["wikitext2"], default="wikitext2")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = str2dtype(args.dtype)
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

    # cache layout usually: ip.key_value_memory_dict[layer_idx] = (conv_state, ssm_state)
    states = []
    for k, v in ip.key_value_memory_dict.items():
        if isinstance(v, tuple) and len(v) >= 2:
            states.append((k, v[1]))
    if not states:
        raise RuntimeError("No ssm_state found in inference cache.")

    print(f"captured {len(states)} layer states at step={args.step}")
    for layer, h in states[:8]:
        q = quantize_state_fp8(h, method=args.method)
        err = (q.float() - h.float())
        rel = err.pow(2).mean().sqrt() / h.float().pow(2).mean().sqrt().clamp_min(1e-12)
        print(
            f"layer={layer} shape={tuple(h.shape)} "
            f"h_absmax={float(h.float().abs().max()):.4e} h_rms={float(h.float().pow(2).mean().sqrt()):.4e} "
            f"mse={float(err.pow(2).mean()):.4e} rel_rmse={float(rel):.4e} "
            f"q_absmax={float(q.float().abs().max()):.4e}"
        )


if __name__ == "__main__":
    main()
