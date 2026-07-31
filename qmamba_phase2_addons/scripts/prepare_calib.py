#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from common import load_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Prepare calibration token segments for Q-Mamba ESR.")
    p.add_argument("--model", type=str, default="state-spaces/mamba2-370m")
    p.add_argument("--loader", type=str, default="mamba_ssm")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--dataset", choices=["wikitext2", "redpajama"], default="wikitext2")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="data/calib/wikitext2_128x2048.pt")
    return p.parse_args()


def load_texts(args):
    if args.dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
        return [t for t in ds["text"] if t and t.strip()]
    if args.dataset == "redpajama":
        # More robust RedPajama sample mirror; avoids the old Together dataset script.
        ds = load_dataset(
            "ZengXiangyu/RedPajama-Data-1T-Sample",
            split=args.split,
            streaming=True,
        )

        texts = []
        for ex in ds:
            txt = ex.get("text", "")
            if isinstance(txt, str) and len(txt.strip()) > 200:
                texts.append(txt)
            if len(texts) >= args.num_samples * 4:
                break
        return texts

    raise ValueError(args.dataset)


def main():
    args = parse_args()
    g = torch.Generator().manual_seed(args.seed)
    tok = load_tokenizer(args.model, args.tokenizer, args.loader)
    texts = load_texts(args)
    joined = "\n\n".join(texts)
    ids = tok(joined, return_tensors="pt").input_ids.squeeze(0).contiguous()
    if ids.numel() < args.seq_len + 1:
        raise RuntimeError(f"Not enough tokens: {ids.numel()} for seq_len={args.seq_len}")

    max_start = ids.numel() - args.seq_len - 1
    starts = torch.randint(0, max_start, (args.num_samples,), generator=g)
    segments = torch.stack([ids[s:s + args.seq_len] for s in starts.tolist()], dim=0).long()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokens": segments,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "model": args.model,
        "tokenizer": args.tokenizer or ("EleutherAI/gpt-neox-20b" if args.loader == "mamba_ssm" else args.model),
    }
    torch.save(payload, out)
    print(f"Saved calibration tokens: {out}")
    print(f"tokens shape: {tuple(segments.shape)}")


if __name__ == "__main__":
    main()
