#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from common import get_device, load_lm, str2dtype
from qmamba.esr.trainable import freeze_all, get_layers, get_mixer
from qmamba.esr.mamba2_parallel_esr import mamba2_mixer_parallel_esr_forward

try:
    from qmamba.esr.mamba2_qaware import mamba2_mixer_qaware_forward
except Exception:
    mamba2_mixer_qaware_forward = None


class CaptureInput:
    def __init__(self, module):
        self.x = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.x = inputs[0].detach().clone()

    def close(self):
        self.handle.remove()


def load_calib_tokens(path: str, seq_len: int, num_samples: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    tokens = payload["tokens"] if isinstance(payload, dict) and "tokens" in payload else payload
    return tokens[:num_samples, :seq_len].contiguous().long()


def report(name: str, ref: torch.Tensor, out: torch.Tensor):
    diff = (out.float() - ref.float()).detach()
    print(f"\n{name}")
    print(f"  ref shape: {tuple(ref.shape)}, out shape: {tuple(out.shape)}")
    print(f"  mse:      {F.mse_loss(out.float(), ref.float()).item():.6e}")
    print(f"  mean_abs: {diff.abs().mean().item():.6e}")
    print(f"  max_abs:  {diff.abs().max().item():.6e}")
    print(f"  ref_rms:  {ref.float().pow(2).mean().sqrt().item():.6e}")


def main():
    p = argparse.ArgumentParser(description="Validate parallel ESR Mamba2 mixer forward against original mixer.")
    p.add_argument("--model", type=str, default="state-spaces/mamba2-370m")
    p.add_argument("--loader", choices=["mamba_ssm"], default="mamba_ssm")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--calib", type=str, default="data/calib/wikitext2_8x256.pt")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--scan_chunk_size", type=int, default=0)
    args = p.parse_args()

    device = get_device(args.device)
    dtype = str2dtype(args.dtype)
    tokens = load_calib_tokens(args.calib, args.seq_len, 1).to(device)

    model = load_lm(args.model, args.loader, dtype, device)
    model.eval()
    freeze_all(model)
    mixer = get_mixer(get_layers(model)[args.block])

    cap = CaptureInput(mixer)
    with torch.no_grad():
        _ = model(tokens)
    cap.close()
    if cap.x is None:
        raise RuntimeError("failed to capture mixer input")
    x = cap.x.to(device=device, dtype=dtype).detach().clone()

    with torch.no_grad():
        ref = mixer(x)
        par = mamba2_mixer_parallel_esr_forward(
            mixer, x, state_quant="none", state_bits=4, qmode="none", scan_chunk_size=args.scan_chunk_size
        )
        report("parallel qmode=none vs original mixer", ref, par)

        if mamba2_mixer_qaware_forward is not None:
            loop = mamba2_mixer_qaware_forward(mixer, x, state_quant="none", state_bits=4, qmode="none")
            report("old loop qmode=none vs original mixer", ref, loop)
            report("parallel qmode=none vs old loop", loop, par)

    print("\nIf mse/max_abs are small, use run_esr_mixer_parallel.py for ESR training.")


if __name__ == "__main__":
    main()
