#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import get_device, load_lm, str2dtype
from qmamba.esr.mamba2_qaware import mamba2_mixer_qaware_forward
from qmamba.esr.trainable import (
    count_trainable_params,
    freeze_all,
    get_layers,
    get_mixer,
    set_trainable_for_mixer_esr,
)
from qmamba.esr.sga_delta import attach_sga_delta, merge_sga_delta

def parse_args():
    p = argparse.ArgumentParser(description="Phase-2 ESR-lite mixer reconstruction for Q-Mamba.")
    p.add_argument("--model", type=str, default="state-spaces/mamba2-370m")
    p.add_argument("--loader", choices=["mamba_ssm"], default="mamba_ssm")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--calib", type=str, default="data/calib/wikitext2_128x2048.pt")
    p.add_argument("--num_samples", type=int, default=8, help="Use a small number first. Paper uses 128.")
    p.add_argument("--seq_len", type=int, default=256, help="Use 256 for debugging; paper uses 2048.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--adam_eps", type=float, default=1e-5)
    p.add_argument("--state_quant", choices=["none", "tensor", "channel", "state", "dsq"], default="dsq")
    p.add_argument("--state_bits", type=int, default=4)
    p.add_argument("--qmode", choices=["esr", "recurrent", "none"], default="esr")
    p.add_argument("--train_mode", choices=["sga", "sga_conv", "sga_conv_d", "all_mixer"], default="sga")
    p.add_argument("--block_start", type=int, default=0)
    p.add_argument("--block_end", type=int, default=-1, help="exclusive; -1 means all blocks")
    p.add_argument("--save_every_block", action="store_true")
    p.add_argument("--out_dir", type=str, default="results/phase2")
    p.add_argument("--save_name", type=str, default="mamba2_370m_esr_lite.pt")
    p.add_argument("--log_name", type=str, default="esr_mixer_log.jsonl")
    return p.parse_args()


class CaptureIO:
    def __init__(self, module):
        self.x = None
        self.y = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # Clone to make sure the captured tensors are normal tensors,
        # not views tied to the teacher forward.
        self.x = inputs[0].detach().clone()
        if isinstance(output, tuple):
            self.y = output[0].detach().clone()
        else:
            self.y = output.detach().clone()

    def close(self):
        self.handle.remove()


def load_calib_tokens(path: str, seq_len: int, num_samples: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    tokens = payload["tokens"] if isinstance(payload, dict) and "tokens" in payload else payload
    if tokens.ndim != 2:
        raise ValueError(f"Expected calibration tokens [N, L], got {tuple(tokens.shape)}")
    if tokens.shape[1] < seq_len:
        raise ValueError(f"calib seq len {tokens.shape[1]} < requested {seq_len}")
    tokens = tokens[:num_samples, :seq_len].contiguous().long()
    return tokens


def capture_teacher_mixer_io(teacher, mixer, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cap = CaptureIO(mixer)

    # Important: use no_grad, not inference_mode.
    # inference_mode creates inference tensors that cannot be saved for backward
    # when used as student inputs during ESR optimization.
    with torch.no_grad():
        _ = teacher(input_ids)

    cap.close()
    if cap.x is None or cap.y is None:
        raise RuntimeError("Hook did not capture mixer input/output")
    return cap.x, cap.y


def append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = str2dtype(args.dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / args.log_name

    print("Loading calibration tokens...")
    calib = load_calib_tokens(args.calib, seq_len=args.seq_len, num_samples=args.num_samples)
    print(f"calib shape: {tuple(calib.shape)}")

    print("Loading teacher...")
    teacher = load_lm(args.model, args.loader, dtype, device)
    teacher.eval()
    freeze_all(teacher)

    print("Loading student...")
    student = load_lm(args.model, args.loader, dtype, device)
    student.train()

    teacher_layers = get_layers(teacher)
    student_layers = get_layers(student)
    n_layers = len(student_layers)
    block_end = n_layers if args.block_end < 0 else min(args.block_end, n_layers)

    print(f"Reconstructing blocks [{args.block_start}, {block_end}) out of {n_layers}")
    print(f"qmode={args.qmode}, state_quant={args.state_quant}, bits={args.state_bits}, train_mode={args.train_mode}")

    for block_idx in range(args.block_start, block_end):
        t_block = teacher_layers[block_idx]
        s_block = student_layers[block_idx]
        t_mixer = get_mixer(t_block)
        s_mixer = get_mixer(s_block)

        freeze_all(student)
        trainable = set_trainable_for_mixer_esr(s_mixer, mode=args.train_mode)

        if args.train_mode == "sga":
            for p in s_mixer.parameters():
                p.requires_grad_(False)

            active_delta_params = attach_sga_delta(s_mixer)

            print(f"[Block {block_idx}] active SGA delta params: {active_delta_params}")

            trainable = [p for p in s_mixer.parameters() if p.requires_grad]

        optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps,)

        t0 = time.perf_counter()
        losses = []
        print(f"\n[Block {block_idx}] trainable params in current mixer: {count_trainable_params(s_mixer):,}")

        for ep in range(args.epochs):
            order = torch.randperm(calib.shape[0])
            pbar = tqdm(order.tolist(), desc=f"block {block_idx} epoch {ep+1}/{args.epochs}")
            for sample_idx in pbar:
                input_ids = calib[sample_idx:sample_idx+1].to(device)

                # Teacher full forward gives the target mixer input/output for this block.
                mixer_in, teacher_out = capture_teacher_mixer_io(teacher, t_mixer, input_ids)

                # Student q-aware reference mixer forward.

                student_in = mixer_in.to(device=device, dtype=dtype).detach().clone()

                student_out = mamba2_mixer_qaware_forward(
                    s_mixer,
                    student_in,
                    state_quant=args.state_quant,
                    state_bits=args.state_bits,
                    qmode=args.qmode,
                )
                loss = F.mse_loss(student_out.float(), teacher_out.float())
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()

                loss_val = float(loss.detach().cpu())
                losses.append(loss_val)
                pbar.set_postfix(loss=f"{loss_val:.3e}")

        elapsed = time.perf_counter() - t0
        rec = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "block_idx": block_idx,
            "epochs": args.epochs,
            "num_samples": args.num_samples,
            "seq_len": args.seq_len,
            "train_mode": args.train_mode,
            "qmode": args.qmode,
            "state_quant": args.state_quant,
            "state_bits": args.state_bits,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_mean": sum(losses) / len(losses) if losses else None,
            "elapsed_s": elapsed,
        }
        append_jsonl(log_path, rec)
        print(f"[Block {block_idx}] first loss={rec['loss_first']:.4e}, last loss={rec['loss_last']:.4e}, elapsed={elapsed:.1f}s")

        if args.save_every_block:
            ckpt = out_dir / f"block_{block_idx:03d}_{args.save_name}"
            for block in student.backbone.layers:
                merge_sga_delta(block.mixer)
            torch.save(student.state_dict(), ckpt)
            print(f"Saved checkpoint: {ckpt}")

    final_path = out_dir / args.save_name
    for block in student.backbone.layers:
        merge_sga_delta(block.mixer)
    torch.save(student.state_dict(), final_path)
    print(f"\nSaved final student state_dict: {final_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
