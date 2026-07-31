#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from common import get_device, load_lm, str2dtype
from qmamba.quant.smoothquant import SmoothQuantCalibrator, save_sq_scales


def parse_args():
    p = argparse.ArgumentParser(description="Collect activation max scales for SmoothQuant calibration.")
    p.add_argument("--model", type=str, default="state-spaces/mamba2-370m")
    p.add_argument("--loader", choices=["mamba_ssm"], default="mamba_ssm")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--calib", type=str, required=True, help="Calibration .pt from prepare_calib.py")
    p.add_argument("--checkpoint", type=str, default=None, help="Optional ESR checkpoint/state_dict loaded before collecting scales")
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--target", choices=["mixer", "in_proj", "out_proj", "all"], default="mixer")
    p.add_argument("--out", type=str, default="data/sq/mamba2_370m_sq_scales.pt")
    return p.parse_args()


def load_calib_tokens(path: str, seq_len: int, num_samples: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    tokens = payload["tokens"] if isinstance(payload, dict) and "tokens" in payload else payload
    return tokens[:num_samples, :seq_len].contiguous().long()


def load_optional_checkpoint(model, path: str | None):
    if path is None:
        return
    print(f"Loading checkpoint before SQ calibration: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")


def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = str2dtype(args.dtype)
    tokens = load_calib_tokens(args.calib, args.seq_len, args.num_samples)
    print(f"calib tokens: {tuple(tokens.shape)}")

    model = load_lm(args.model, args.loader, dtype, device)
    load_optional_checkpoint(model, args.checkpoint)
    model.eval()

    with torch.no_grad(), SmoothQuantCalibrator(model, target=args.target) as calib:
        for i in tqdm(range(tokens.shape[0]), desc="Collect SQ scales"):
            input_ids = tokens[i:i+1].to(device)
            _ = model(input_ids)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "calib": args.calib,
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "target": args.target,
        "dtype": args.dtype,
    }
    save_sq_scales(str(out), calib.act_scales, meta=meta)
    print(f"Saved SmoothQuant scales: {out}")
    print(f"num modules with scales: {len(calib.act_scales)}")
    for k in list(calib.act_scales.keys())[:8]:
        v = calib.act_scales[k]
        print(f"  {k}: shape={tuple(v.shape)}, max={float(v.max()):.4e}, mean={float(v.mean()):.4e}")


if __name__ == "__main__":
    main()
