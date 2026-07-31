#!/usr/bin/env python3
"""
Analyze Mamba recurrent state quantization error.

Supported methods:
  1. FP8 per-tensor fake quant
  2. Hadamard + FP8 per-tensor fake quant
  3. DSQ INT4 fake quant

Expected state dump:
  torch.save({"states": states}, path)

where states can be:
  - list/tuple of tensors, one tensor per layer
  - dict mapping layer name -> tensor
  - single tensor

Each state tensor should have last two dims as:
  [..., P, N]
where:
  P = channel / headdim dimension
  N = state dimension

Examples:
  [B, H, P, N]
  [T, B, H, P, N]
  [num_snapshots, B, H, P, N]
"""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Tuple, List, Any

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qmamba.quant.hadamard_state import quantize_state_hadamard

# -----------------------------
# Basic utilities
# -----------------------------

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_float(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float()


def rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.float() ** 2) + eps)


def percentile_abs(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.detach().float().abs().flatten()
    if flat.numel() == 0:
        return torch.tensor(float("nan"))
    return torch.quantile(flat, q)


# -----------------------------
# Load state dump
# -----------------------------

def extract_tensors(obj: Any, state_key: str = "states") -> Dict[str, torch.Tensor]:
    """
    Robustly extract layer tensors from common torch.save formats.
    """

    if isinstance(obj, torch.Tensor):
        return {"layer0": obj}

    if isinstance(obj, (list, tuple)):
        out = {}
        for i, v in enumerate(obj):
            if isinstance(v, torch.Tensor):
                out[f"layer{i}"] = v
        if len(out) == 0:
            raise ValueError("List/tuple contained no tensors.")
        return out

    if isinstance(obj, dict):
        if state_key in obj:
            return extract_tensors(obj[state_key], state_key=state_key)

        out = {}
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                out[str(k)] = v
            elif isinstance(v, (list, tuple)):
                for i, vv in enumerate(v):
                    if isinstance(vv, torch.Tensor):
                        out[f"{k}_{i}"] = vv
            elif isinstance(v, dict):
                # One level recursive extraction.
                sub = extract_tensors(v, state_key=state_key)
                for kk, vv in sub.items():
                    out[f"{k}_{kk}"] = vv

        if len(out) == 0:
            raise ValueError(
                f"Dict contained no tensors. Keys: {list(obj.keys())}. "
                f"Try setting --state_key if your states are under another key."
            )
        return out

    raise ValueError(f"Unsupported state dump type: {type(obj)}")


def load_states(path: str, state_key: str, device: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    states = extract_tensors(obj, state_key=state_key)

    cleaned = {}
    for name, x in states.items():
        if not isinstance(x, torch.Tensor):
            continue
        if x.ndim < 2:
            continue
        cleaned[name] = x.to(device)

    if len(cleaned) == 0:
        raise ValueError("No valid state tensors found.")

    return cleaned


# -----------------------------
# FP8 fake quantization
# -----------------------------

def fp8_max_finite(fmt: str) -> float:
    fmt = fmt.lower()
    if fmt in ["e4m3", "e4m3fn", "float8_e4m3fn"]:
        return 448.0
    if fmt in ["e5m2", "float8_e5m2"]:
        return 57344.0
    raise ValueError(f"Unknown fp8 format: {fmt}")


def fp8_torch_dtype(fmt: str):
    fmt = fmt.lower()
    if fmt in ["e4m3", "e4m3fn", "float8_e4m3fn"]:
        return getattr(torch, "float8_e4m3fn", None)
    if fmt in ["e5m2", "float8_e5m2"]:
        return getattr(torch, "float8_e5m2", None)
    return None


def approx_float_format_qdq(y: torch.Tensor, fmt: str) -> torch.Tensor:
    """
    Approximate FP8 quant-dequant fallback if torch.float8 casting is unavailable.

    This is not bit-exact, but it is good enough for error trend visualization.
    Prefer native torch.float8 if available.
    """
    fmt = fmt.lower()

    if fmt in ["e4m3", "e4m3fn", "float8_e4m3fn"]:
        exp_bits = 4
        man_bits = 3
        emin = -6
        max_finite = 448.0
    elif fmt in ["e5m2", "float8_e5m2"]:
        exp_bits = 5
        man_bits = 2
        emin = -14
        max_finite = 57344.0
    else:
        raise ValueError(f"Unknown fp8 format: {fmt}")

    del exp_bits

    y = y.float()
    sign = torch.sign(y)
    ay = y.abs().clamp(max=max_finite)

    zero = ay == 0
    min_normal = 2.0 ** emin

    # Normal numbers.
    exponent = torch.floor(torch.log2(torch.clamp(ay, min=min_normal)))
    step = torch.pow(2.0, exponent - man_bits)
    q_normal = torch.round(ay / step) * step

    # Subnormal approximation.
    sub_step = 2.0 ** (emin - man_bits)
    q_sub = torch.round(ay / sub_step) * sub_step

    q = torch.where(ay < min_normal, q_sub, q_normal)
    q = q.clamp(max=max_finite)
    q = torch.where(zero, torch.zeros_like(q), q)
    return sign * q


def fp8_qdq_per_tensor(
    x: torch.Tensor,
    fmt: str = "e4m3",
    eps: float = 1e-12,
    use_native_float8: bool = True,
) -> torch.Tensor:
    """
    Per-tensor scaled FP8 QDQ:
      scale = absmax(x) / fp8_max
      qdq = fp8_round(x / scale) * scale
    """
    x = x.float()
    maxval = fp8_max_finite(fmt)
    absmax = x.abs().max()

    if absmax <= eps:
        return torch.zeros_like(x)

    scale = absmax / maxval
    y = (x / scale).clamp(min=-maxval, max=maxval)

    dtype = fp8_torch_dtype(fmt)
    if use_native_float8 and dtype is not None:
        try:
            qdq = y.to(dtype).to(torch.float32)
        except Exception:
            qdq = approx_float_format_qdq(y, fmt)
    else:
        qdq = approx_float_format_qdq(y, fmt)

    return qdq * scale


# -----------------------------
# Hadamard transform
# -----------------------------

def fwht_orthonormal(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Orthonormal Fast Walsh-Hadamard Transform along dim.
    Requires the selected dimension to be power-of-two.
    """
    dim = dim if dim >= 0 else x.ndim + dim
    n = x.shape[dim]

    if not is_power_of_two(n):
        raise ValueError(f"Hadamard dimension size must be power-of-two, got {n}.")

    y = x.transpose(dim, -1).contiguous()
    orig_shape = y.shape
    y = y.reshape(-1, n)

    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        y = y.reshape(-1, n)
        h *= 2

    y = y / math.sqrt(n)
    y = y.reshape(orig_shape)
    y = y.transpose(dim, -1).contiguous()
    return y


def apply_hadamard(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    mode:
      none
      state    -> last dim N
      channel  -> second last dim P
      both     -> P then N
    """
    mode = mode.lower()
    if mode == "none":
        return x
    if mode == "state":
        return fwht_orthonormal(x, dim=-1)
    if mode == "channel":
        return fwht_orthonormal(x, dim=-2)
    if mode == "both":
        return fwht_orthonormal(fwht_orthonormal(x, dim=-2), dim=-1)
    raise ValueError(f"Unknown Hadamard mode: {mode}")


def hadamard_fp8_qdq(
    x: torch.Tensor,
    fmt: str = "e4m3",
    mode: str = "state",
    eps: float = 1e-12,
    use_native_float8: bool = True,
) -> torch.Tensor:
    """
    Rotate -> tensor FP8 QDQ -> inverse rotate.

    Since orthonormal Hadamard is self-inverse:
      H^{-1} = H
    """
    z = apply_hadamard(x.float(), mode)
    z_q = fp8_qdq_per_tensor(
        z,
        fmt=fmt,
        eps=eps,
        use_native_float8=use_native_float8,
    )
    x_hat = apply_hadamard(z_q, mode)
    return x_hat


# -----------------------------
# DSQ INT fake quantization
# -----------------------------

def dsq_int_qdq(
    x: torch.Tensor,
    bits: int = 4,
    channel_dim: int = -2,
    state_dim: int = -1,
    range_mode: str = "symmetric",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, float]:
    """
    Decoupled Scale Quantization fake quant.

    Expected state shape:
      [..., P, N]

    S_channel[..., p, 1] = sqrt(mean(abs(h[..., p, :]), over N))
    S_state[..., 1, n]   = max_p(abs(h[..., p, n] / S_channel[..., p, 1])) / denom

    range_mode:
      symmetric  -> [-7, 7] for int4, denom=7
      neg8_to_7  -> [-8, 7] for int4, denom=8
    """
    x = x.float()
    channel_dim = channel_dim if channel_dim >= 0 else x.ndim + channel_dim
    state_dim = state_dim if state_dim >= 0 else x.ndim + state_dim

    if range_mode == "symmetric":
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax
        denom = float(qmax)
    elif range_mode in ["neg8_to_7", "twos", "two_sided"]:
        qmax = 2 ** (bits - 1) - 1
        qmin = -(2 ** (bits - 1))
        denom = float(2 ** (bits - 1))
    else:
        raise ValueError(f"Unknown range_mode: {range_mode}")

    s_channel = torch.sqrt(torch.mean(x.abs(), dim=state_dim, keepdim=True) + eps)
    u = x / s_channel

    s_state = torch.amax(u.abs(), dim=channel_dim, keepdim=True) / denom
    s_state = s_state.clamp(min=eps)

    scale = s_channel * s_state
    q = torch.round(x / scale).clamp(qmin, qmax)
    x_hat = q * scale

    sat_rate = ((q <= qmin) | (q >= qmax)).float().mean().item()
    return x_hat, sat_rate


# -----------------------------
# Metrics
# -----------------------------

@torch.no_grad()
def compute_metrics(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    sat_rate: float = float("nan"),
    eps: float = 1e-12,
) -> Dict[str, float]:
    x = x.float()
    x_hat = x_hat.float()
    e = x_hat - x

    mse = torch.mean(e ** 2)
    mae = torch.mean(e.abs())
    max_abs_err = e.abs().max()

    x_rms = rms(x, eps)
    e_rms = rms(e, eps)

    rel_rmse = e_rms / (x_rms + eps)
    nmse = mse / (torch.mean(x ** 2) + eps)
    sqnr_db = 20.0 * torch.log10((x_rms + eps) / (e_rms + eps))

    x_flat = x.flatten()
    y_flat = x_hat.flatten()
    cosine = torch.sum(x_flat * y_flat) / (
        torch.linalg.norm(x_flat) * torch.linalg.norm(y_flat) + eps
    )

    return {
        "numel": float(x.numel()),
        "x_rms": float(x_rms.item()),
        "mse": float(mse.item()),
        "mae": float(mae.item()),
        "max_abs_err": float(max_abs_err.item()),
        "rel_rmse": float(rel_rmse.item()),
        "nmse": float(nmse.item()),
        "sqnr_db": float(sqnr_db.item()),
        "cosine": float(cosine.item()),
        "sat_rate": float(sat_rate),
        "x_absmax": float(x.abs().max().item()),
        "x_p999_abs": float(percentile_abs(x, 0.999).item()),
        "x_max_over_rms": float((x.abs().max() / (x_rms + eps)).item()),
        "x_p999_over_rms": float((percentile_abs(x, 0.999) / (x_rms + eps)).item()),
    }


def error_heatmap_pn(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Return [P, N] relative RMSE map, aggregating over all leading dims.
    """
    x = x.float()
    x_hat = x_hat.float()
    e = x_hat - x

    if x.ndim < 2:
        raise ValueError("Need tensor with last two dims [P, N].")

    reduce_dims = tuple(range(x.ndim - 2))

    err_rms = torch.sqrt(torch.mean(e ** 2, dim=reduce_dims) + eps)
    x_rms = torch.sqrt(torch.mean(x ** 2, dim=reduce_dims) + eps)

    return err_rms / (x_rms + eps)


# -----------------------------
# Plotting
# -----------------------------

def save_csv(rows: List[Dict[str, Any]], path: str):
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def method_sort_key(name: str) -> int:
    order = {
        "fp8": 0,
        "hadamard_fp8": 1,
        "dsq_int": 2,
    }
    return order.get(name, 99)


def plot_layer_metric(rows, metric: str, ylabel: str, out_path: str, logy: bool = False):
    methods = sorted(set(r["method"] for r in rows), key=method_sort_key)

    plt.figure(figsize=(8, 4.5))

    for method in methods:
        sub = [r for r in rows if r["method"] == method]
        sub = sorted(sub, key=lambda r: r["layer_index"])
        xs = [r["layer_index"] for r in sub]
        ys = [r[metric] for r in sub]
        plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=method)

    plt.xlabel("Layer index")
    plt.ylabel(ylabel)
    plt.title(f"Layer-wise {ylabel}")
    plt.grid(True, alpha=0.3)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_summary_bar(summary_rows, metric: str, ylabel: str, out_path: str):
    rows = sorted(summary_rows, key=lambda r: method_sort_key(r["method"]))
    labels = [r["method"] for r in rows]
    vals = [r[metric] for r in rows]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals)
    plt.ylabel(ylabel)
    plt.title(f"Average {ylabel}")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_heatmap(mat: torch.Tensor, title: str, out_path: str, log10: bool = True):
    m = mat.detach().cpu().float()
    if log10:
        m = torch.log10(m + 1e-8)
        color_label = "log10(relative RMSE)"
    else:
        color_label = "relative RMSE"

    plt.figure(figsize=(7, 4.5))
    plt.imshow(m.numpy(), aspect="auto", origin="lower")
    plt.colorbar(label=color_label)
    plt.xlabel("State dimension N")
    plt.ylabel("Channel / headdim P")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_hadamard_outlier(
    layer_names: List[str],
    before_max_rms: List[float],
    after_max_rms: List[float],
    before_p999_rms: List[float],
    after_p999_rms: List[float],
    out_path: str,
):
    xs = list(range(len(layer_names)))

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, before_max_rms, marker="o", linewidth=1.5, markersize=3, label="original max/RMS")
    plt.plot(xs, after_max_rms, marker="o", linewidth=1.5, markersize=3, label="Hadamard max/RMS")
    plt.plot(xs, before_p999_rms, marker="x", linewidth=1.5, markersize=3, label="original p99.9/RMS")
    plt.plot(xs, after_p999_rms, marker="x", linewidth=1.5, markersize=3, label="Hadamard p99.9/RMS")

    plt.xlabel("Layer index")
    plt.ylabel("Outlier ratio")
    plt.title("Outlier concentration before/after Hadamard")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_logabs_histogram(
    x: torch.Tensor,
    z: torch.Tensor,
    out_path: str,
    max_points: int = 300_000,
):
    x_abs = x.detach().float().abs().flatten()
    z_abs = z.detach().float().abs().flatten()

    if x_abs.numel() > max_points:
        idx = torch.randperm(x_abs.numel(), device=x_abs.device)[:max_points]
        x_abs = x_abs[idx]
    if z_abs.numel() > max_points:
        idx = torch.randperm(z_abs.numel(), device=z_abs.device)[:max_points]
        z_abs = z_abs[idx]

    x_log = torch.log10(x_abs.cpu() + 1e-12).numpy()
    z_log = torch.log10(z_abs.cpu() + 1e-12).numpy()

    plt.figure(figsize=(7, 4.5))
    plt.hist(x_log, bins=100, alpha=0.55, density=True, label="original")
    plt.hist(z_log, bins=100, alpha=0.55, density=True, label="Hadamard")
    plt.xlabel("log10(|state value|)")
    plt.ylabel("Density")
    plt.title("State magnitude distribution before/after Hadamard")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


@torch.no_grad()
def qdq_like_recurrent_step(
    x: torch.Tensor,
    method: str,
    bits: int = 8,
    hadamard_dims: str = "none",
    channel_dim: int = -2,
    state_dim: int = -1,
    qrange: str = "symmetric",
    channel_stat: str = "sqrt_mean_abs",
    hadamard_compute_dtype: torch.dtype = torch.float32,
    snapshot_dim: int = 0,
) -> torch.Tensor:
    """
    Apply quantization like recurrent PPL.

    If x is a dumped trajectory [T, B, H, P, N], quantize each x[t]
    independently, matching the real recurrent step behavior.

    If x is already [B, H, P, N], quantize directly.
    """
    x = x.float()

    # Most dumped states are [T, B, H, P, N].
    if x.ndim >= 5:
        ys = []
        for t in range(x.shape[snapshot_dim]):
            xt = x.select(snapshot_dim, t).contiguous()
            yt = quantize_state_hadamard(
                xt,
                method=method,
                bits=bits,
                hadamard_dims=hadamard_dims,
                channel_dim=channel_dim,
                state_dim=state_dim,
                qrange=qrange,
                channel_stat=channel_stat,
                hadamard_compute_dtype=hadamard_compute_dtype,
            )
            ys.append(yt)
        return torch.stack(ys, dim=snapshot_dim)

    return quantize_state_hadamard(
        x,
        method=method,
        bits=bits,
        hadamard_dims=hadamard_dims,
        channel_dim=channel_dim,
        state_dim=state_dim,
        qrange=qrange,
        channel_stat=channel_stat,
        hadamard_compute_dtype=hadamard_compute_dtype,
    )

# -----------------------------
# Main analysis
# -----------------------------

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--states", type=str, required=True, help="Path to state dump .pt")
    parser.add_argument("--state_key", type=str, default="states")
    parser.add_argument("--out_dir", type=str, default="results/state_quant_error")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--fp8_format", type=str, default="e4m3", choices=["e4m3", "e5m2"])
    parser.add_argument("--no_native_float8", action="store_true")

    parser.add_argument("--hadamard_mode", type=str, default="state", choices=["none", "state", "channel", "both"])

    parser.add_argument("--int_bits", type=int, default=4)
    parser.add_argument("--dsq_range", type=str, default="symmetric", choices=["symmetric", "neg8_to_7"])

    parser.add_argument("--heatmap_topk", type=int, default=1)
    parser.add_argument("--hist_max_points", type=int, default=300_000)

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print(f"[load] {args.states}")
    states = load_states(args.states, state_key=args.state_key, device=args.device)

    layer_items = list(states.items())
    layer_items = sorted(layer_items, key=lambda kv: kv[0])

    rows = []
    summary_acc = {}
    saved_qdq = {}

    use_native_float8 = not args.no_native_float8

    print(f"[info] found {len(layer_items)} state tensors")
    for layer_index, (layer_name, x_raw) in enumerate(layer_items):
        x = x_raw.float()

        if x.ndim < 2:
            print(f"[skip] {layer_name}: ndim < 2")
            continue

        print(f"[layer {layer_index}] {layer_name}, shape={tuple(x.shape)}")

        qrange_for_repo = "full" if args.dsq_range in ("full", "neg8_to_7") else "symmetric"

        methods = {}

        # 1. FP8 tensor, matching your PPL method name.
        x_fp8 = qdq_like_recurrent_step(
            x,
            method=f"fp8_{args.fp8_format}_scaled_tensor",
            bits=args.int_bits,
            hadamard_dims="none",
            channel_dim=args.state_channel_dim if hasattr(args, "state_channel_dim") else -2,
            state_dim=args.state_state_dim if hasattr(args, "state_state_dim") else -1,
            qrange=qrange_for_repo,
        )
        methods["fp8"] = (x_fp8, float("nan"))

        # 2. Hadamard + FP8 tensor.
        if args.hadamard_mode != "none":
            x_hfp8 = qdq_like_recurrent_step(
                x,
                method=f"fp8_{args.fp8_format}_scaled_tensor",
                bits=args.int_bits,
                hadamard_dims=args.hadamard_mode,
                channel_dim=args.state_channel_dim if hasattr(args, "state_channel_dim") else -2,
                state_dim=args.state_state_dim if hasattr(args, "state_state_dim") else -1,
                qrange=qrange_for_repo,
            )
            methods["hadamard_fp8"] = (x_hfp8, float("nan"))

        # 3. DSQ INT4.
        x_dsq = qdq_like_recurrent_step(
            x,
            method="dsq",
            bits=args.int_bits,
            hadamard_dims="none",
            channel_dim=args.state_channel_dim if hasattr(args, "state_channel_dim") else -2,
            state_dim=args.state_state_dim if hasattr(args, "state_state_dim") else -1,
            qrange=qrange_for_repo,
            channel_stat=getattr(args, "channel_stat", "sqrt_mean_abs"),
        )
        methods["dsq_int"] = (x_dsq, float("nan"))

        saved_qdq[layer_name] = {
            "x": x.detach().cpu(),
            **{m: y.detach().cpu() for m, (y, _) in methods.items()},
        }

        for method, (x_hat, sat_rate) in methods.items():
            m = compute_metrics(x, x_hat, sat_rate=sat_rate)
            row = {
                "layer_index": layer_index,
                "layer_name": layer_name,
                "method": method,
                "shape": str(tuple(x.shape)),
                **m,
            }
            rows.append(row)

            if method not in summary_acc:
                summary_acc[method] = []
            summary_acc[method].append(row)

    metrics_path = os.path.join(args.out_dir, "metrics_by_layer.csv")
    save_csv(rows, metrics_path)
    print(f"[save] {metrics_path}")

    # Summary CSV.
    summary_rows = []
    for method, rs in summary_acc.items():
        total_numel = sum(r["numel"] for r in rs)
        mean_rel_rmse = sum(r["rel_rmse"] for r in rs) / len(rs)
        mean_sqnr = sum(r["sqnr_db"] for r in rs) / len(rs)
        mean_nmse = sum(r["nmse"] for r in rs) / len(rs)
        mean_cosine = sum(r["cosine"] for r in rs) / len(rs)

        # Weighted by number of elements.
        weighted_mse = sum(r["mse"] * r["numel"] for r in rs) / total_numel
        weighted_x2 = sum((r["x_rms"] ** 2) * r["numel"] for r in rs) / total_numel
        global_rel_rmse = math.sqrt(weighted_mse / max(weighted_x2, 1e-30))
        global_sqnr = -20.0 * math.log10(max(global_rel_rmse, 1e-30))

        summary_rows.append({
            "method": method,
            "num_layers": len(rs),
            "total_numel": total_numel,
            "mean_rel_rmse": mean_rel_rmse,
            "mean_sqnr_db": mean_sqnr,
            "mean_nmse": mean_nmse,
            "mean_cosine": mean_cosine,
            "global_rel_rmse": global_rel_rmse,
            "global_sqnr_db": global_sqnr,
        })

    summary_path = os.path.join(args.out_dir, "metrics_summary.csv")
    save_csv(summary_rows, summary_path)
    print(f"[save] {summary_path}")

    # Layer plots.
    plot_layer_metric(
        rows,
        metric="rel_rmse",
        ylabel="relative RMSE",
        out_path=os.path.join(args.out_dir, "fig_layer_relrmse.png"),
        logy=True,
    )
    plot_layer_metric(
        rows,
        metric="sqnr_db",
        ylabel="SQNR (dB)",
        out_path=os.path.join(args.out_dir, "fig_layer_sqnr.png"),
        logy=False,
    )
    plot_summary_bar(
        summary_rows,
        metric="global_rel_rmse",
        ylabel="global relative RMSE",
        out_path=os.path.join(args.out_dir, "fig_summary_global_relrmse.png"),
    )

    # Hadamard outlier plots.
    if args.hadamard_mode != "none":
        before_max_rms = []
        after_max_rms = []
        before_p999_rms = []
        after_p999_rms = []
        layer_names = []

        for layer_name, x_raw in layer_items:
            x = x_raw.float()
            z = apply_hadamard(x, args.hadamard_mode)

            x_r = rms(x)
            z_r = rms(z)

            before_max_rms.append(float((x.abs().max() / x_r).item()))
            after_max_rms.append(float((z.abs().max() / z_r).item()))
            before_p999_rms.append(float((percentile_abs(x, 0.999) / x_r).item()))
            after_p999_rms.append(float((percentile_abs(z, 0.999) / z_r).item()))
            layer_names.append(layer_name)

        plot_hadamard_outlier(
            layer_names,
            before_max_rms,
            after_max_rms,
            before_p999_rms,
            after_p999_rms,
            out_path=os.path.join(args.out_dir, "fig_outlier_hadamard.png"),
        )

    # Heatmaps for worst layers by relRMSE.
    for method in sorted(summary_acc.keys(), key=method_sort_key):
        method_rows = sorted(
            summary_acc[method],
            key=lambda r: r["rel_rmse"],
            reverse=True,
        )[: args.heatmap_topk]

        for r in method_rows:
            layer_name = r["layer_name"]
            layer_index = r["layer_index"]

            x = saved_qdq[layer_name]["x"]
            x_hat = saved_qdq[layer_name][method]

            heat = error_heatmap_pn(x, x_hat)
            out_path = os.path.join(
                args.out_dir,
                f"heatmap_{method}_layer{layer_index}.png",
            )
            title = f"{method} relative error map, layer {layer_index}"
            save_heatmap(heat, title=title, out_path=out_path, log10=True)

    # Histogram for Hadamard on the worst Hadamard-FP8 layer.
    if args.hadamard_mode != "none" and "hadamard_fp8" in summary_acc:
        worst = sorted(
            summary_acc["hadamard_fp8"],
            key=lambda r: r["rel_rmse"],
            reverse=True,
        )[0]
        layer_name = worst["layer_name"]
        layer_index = worst["layer_index"]

        x = states[layer_name].float()
        z = apply_hadamard(x, args.hadamard_mode)

        plot_logabs_histogram(
            x,
            z,
            out_path=os.path.join(args.out_dir, f"hist_logabs_hadamard_layer{layer_index}.png"),
            max_points=args.hist_max_points,
        )

    print("[done]")
    print(f"Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()