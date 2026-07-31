import csv
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer


def str2dtype(name: str):
    name = name.lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "cuda_available": True,
        "cuda_device": idx,
        "gpu_name": props.name,
        "gpu_total_mem_gb": round(props.total_memory / 1024**3, 3),
        "torch_cuda": torch.version.cuda,
    }


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_csv_from_jsonl(jsonl_path: str | Path, csv_path: str | Path) -> None:
    jsonl_path = Path(jsonl_path)
    csv_path = Path(csv_path)
    if not jsonl_path.exists():
        return
    rows = []
    keys = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        flat = flatten_dict(rec)
        rows.append(flat)
        keys.update(flat.keys())
    if not rows:
        return
    preferred = [
        "time", "script", "model", "loader", "dataset", "split", "dtype", "seq_len",
        "stride", "batch_size", "num_windows", "total_tokens", "ppl", "avg_nll",
        "elapsed_s", "tokens_per_s", "peak_allocated_gb", "peak_reserved_gb",
        "prompt_len", "gen_len", "generated_tokens", "prefill_ms", "total_generate_ms",
        "approx_decode_ms", "decode_tokens_per_s", "gpu_name",
    ]
    fieldnames = [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def load_tokenizer(model_name: str, tokenizer_name: Optional[str], loader: str):
    # Mamba language models commonly use GPT-NeoX tokenizer. Passing --tokenizer is safest.
    name = tokenizer_name or ("EleutherAI/gpt-neox-20b" if loader == "mamba_ssm" else model_name)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_lm(model_name: str, loader: str, dtype: torch.dtype, device: torch.device):
    if loader == "hf":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        return model
    if loader == "mamba_ssm":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        # Different mamba-ssm versions accept slightly different kwargs; keep fallback.
        try:
            model = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=dtype)
        except TypeError:
            model = MambaLMHeadModel.from_pretrained(model_name)
            model.to(device=device, dtype=dtype if device.type == "cuda" else torch.float32)
        model.eval()
        return model
    raise ValueError(f"Unknown loader: {loader}")


def extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def base_record(script: str) -> Dict[str, Any]:
    rec = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": script,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    rec.update(get_gpu_info())
    return rec
