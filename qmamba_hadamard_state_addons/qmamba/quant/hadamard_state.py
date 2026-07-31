from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch


# -----------------------------
# Hadamard / FWHT utilities
# -----------------------------


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def fwht_lastdim(x: torch.Tensor, normalize: bool = True, compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Fast Walsh-Hadamard transform over the last dimension.

    The transform is orthogonal if normalize=True, so inverse transform is itself.
    This is a pure PyTorch implementation for experiments; it is not meant as a
    production latency kernel.
    """
    n = x.shape[-1]
    if not _is_power_of_two(n):
        raise ValueError(f"Hadamard dimension must be a power of 2, got {n}")
    orig_dtype = x.dtype
    orig_shape = x.shape
    y = x.to(compute_dtype).reshape(-1, n).contiguous()
    h = 1
    while h < n:
        # [rows, groups, 2, h]
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    if normalize:
        y = y / math.sqrt(n)
    return y.reshape(orig_shape).to(orig_dtype)


def hadamard_transform_dim(x: torch.Tensor, dim: int, normalize: bool = True, compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    dim = dim % x.ndim
    if dim == x.ndim - 1:
        return fwht_lastdim(x, normalize=normalize, compute_dtype=compute_dtype)
    y = x.movedim(dim, -1).contiguous()
    y = fwht_lastdim(y, normalize=normalize, compute_dtype=compute_dtype)
    return y.movedim(-1, dim).contiguous()


def parse_hadamard_dims(hadamard_dims: str | Sequence[int], channel_dim: int = -2, state_dim: int = -1) -> Tuple[int, ...]:
    """Parse rotation axes.

    For Mamba2 state [B,H,P,N]:
      state   -> N axis
      channel -> P axis
      both    -> P and N axes
    """
    if isinstance(hadamard_dims, str):
        h = hadamard_dims.lower().replace("-", "_")
        if h in ("", "none", "off", "no"):
            return tuple()
        if h in ("state", "n"):
            return (state_dim,)
        if h in ("channel", "p"):
            return (channel_dim,)
        if h in ("both", "channel_state", "state_channel", "pn", "np"):
            return (channel_dim, state_dim)
        # Allow comma-separated raw dims, e.g. "-2,-1".
        try:
            return tuple(int(v.strip()) for v in h.split(",") if v.strip())
        except Exception as e:
            raise ValueError(f"Bad hadamard_dims={hadamard_dims}") from e
    return tuple(int(d) for d in hadamard_dims)


def apply_hadamard(x: torch.Tensor, hadamard_dims: str | Sequence[int] = "state", channel_dim: int = -2, state_dim: int = -1, compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    dims = parse_hadamard_dims(hadamard_dims, channel_dim=channel_dim, state_dim=state_dim)
    y = x
    for d in dims:
        y = hadamard_transform_dim(y, d, normalize=True, compute_dtype=compute_dtype)
    return y


# -----------------------------
# INT fake quantization
# -----------------------------


def _qparams(bits: int, qrange: str = "symmetric") -> Tuple[int, int, int]:
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    qrange = qrange.lower()
    if qrange in ("symmetric", "signed_symmetric", "-qmax_qmax"):
        qmax = 2 ** (bits - 1) - 1
        return -qmax, qmax, qmax
    if qrange in ("full", "signed_full", "int"):
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        # Use qmax for scale denominator; negative side gets one extra code.
        return qmin, qmax, qmax
    raise ValueError(f"Unknown qrange={qrange}")


def _reduce_dims_for_granularity(x: torch.Tensor, granularity: str, channel_dim: int = -2, state_dim: int = -1) -> Tuple[int, ...]:
    g = granularity.lower()
    ndim = x.ndim
    cdim = channel_dim % ndim
    sdim = state_dim % ndim
    if g in ("tensor", "per_tensor"):
        return tuple(range(ndim))
    if g in ("head", "per_head", "batch_head"):
        # For [B,H,P,N], keep [B,H], reduce P,N.
        return (cdim, sdim)
    if g in ("channel", "per_channel"):
        # Keep P, reduce N.
        return (sdim,)
    if g in ("state", "per_state"):
        # Keep N, reduce P.
        return (cdim,)
    raise ValueError(f"Unknown INT granularity={granularity}")


def fake_quant_int_uniform(
    x: torch.Tensor,
    bits: int = 8,
    granularity: str = "tensor",
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
    qrange: str = "symmetric",
) -> torch.Tensor:
    """Symmetric fake INT quantization for Mamba state."""
    orig_dtype = x.dtype
    xf = x.float()
    qmin, qmax, scale_den = _qparams(bits, qrange=qrange)
    reduce_dims = _reduce_dims_for_granularity(xf, granularity, channel_dim=channel_dim, state_dim=state_dim)
    amax = xf.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    scale = amax / float(scale_den)
    q = torch.round(xf / scale).clamp(qmin, qmax)
    return (q * scale).to(orig_dtype)


def fake_quant_int_dsq(
    x: torch.Tensor,
    bits: int = 8,
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
    qrange: str = "symmetric",
    channel_stat: str = "sqrt_mean_abs",
) -> torch.Tensor:
    """DSQ-style fake INT quantization.

    For Mamba2 recurrent state [B,H,P,N]:
      S_channel[..., P, 1] = sqrt(mean(abs(h[..., P, :]))) by default
      S_state[..., 1, N]   = max_P(abs(h / S_channel)) / qmax
      Q(h) = round(h / (S_channel*S_state)) * (S_channel*S_state)
    """
    orig_dtype = x.dtype
    xf = x.float()
    cdim = channel_dim % xf.ndim
    sdim = state_dim % xf.ndim
    qmin, qmax, scale_den = _qparams(bits, qrange=qrange)

    channel_stat = channel_stat.lower()
    if channel_stat in ("sqrt_mean_abs", "paper", "mean_abs_sqrt"):
        s_channel = torch.sqrt(torch.mean(xf.abs(), dim=sdim, keepdim=True).clamp_min(eps))
    elif channel_stat in ("rms", "sqrt_mean_square"):
        s_channel = torch.sqrt(torch.mean(xf.pow(2), dim=sdim, keepdim=True).clamp_min(eps))
    elif channel_stat in ("mean_abs",):
        s_channel = torch.mean(xf.abs(), dim=sdim, keepdim=True).clamp_min(eps)
    else:
        raise ValueError(f"Unknown channel_stat={channel_stat}")

    x_smooth = xf / s_channel.clamp_min(eps)
    s_state = x_smooth.abs().amax(dim=cdim, keepdim=True).clamp_min(eps) / float(scale_den)
    scale = (s_channel * s_state).clamp_min(eps)
    q = torch.round(xf / scale).clamp(qmin, qmax)
    return (q * scale).to(orig_dtype)


def fake_quant_int_state(
    x: torch.Tensor,
    method: str = "tensor",
    bits: int = 8,
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
    qrange: str = "symmetric",
    channel_stat: str = "sqrt_mean_abs",
) -> torch.Tensor:
    m = method.lower()
    if m in ("none", "fp16", "off"):
        return x
    if m in ("tensor", "head", "channel", "state", "per_tensor", "per_head", "per_channel", "per_state"):
        return fake_quant_int_uniform(x, bits=bits, granularity=m, channel_dim=channel_dim, state_dim=state_dim, eps=eps, qrange=qrange)
    if m in ("dsq", "dsq_rms"):
        stat = "rms" if m.endswith("rms") else channel_stat
        return fake_quant_int_dsq(x, bits=bits, channel_dim=channel_dim, state_dim=state_dim, eps=eps, qrange=qrange, channel_stat=stat)
    raise ValueError(f"Unknown INT state quant method={method}")


# -----------------------------
# FP8 fake quantization
# -----------------------------


@dataclass(frozen=True)
class FP8Format:
    name: str
    mant_bits: int
    exp_min: int
    exp_max: int
    max_finite: float


FP8_FORMATS = {
    "e4m3": FP8Format("e4m3", mant_bits=3, exp_min=-6, exp_max=8, max_finite=448.0),
    "e5m2": FP8Format("e5m2", mant_bits=2, exp_min=-14, exp_max=15, max_finite=57344.0),
}


def _canonical_fp8(fmt: str) -> str:
    f = fmt.lower().replace("fp8_", "")
    if f in ("e4m3fn", "float8_e4m3fn"):
        f = "e4m3"
    if f in ("float8_e5m2",):
        f = "e5m2"
    if f not in FP8_FORMATS:
        raise ValueError(f"Unknown FP8 format={fmt}")
    return f


def fake_fp8_core(x: torch.Tensor, fmt: str = "e4m3") -> torch.Tensor:
    spec = FP8_FORMATS[_canonical_fp8(fmt)]
    orig_dtype = x.dtype
    xf = x.float()
    sign = torch.sign(xf)
    ax = xf.abs()
    tiny = torch.finfo(torch.float32).tiny
    ax_safe = ax.clamp_min(tiny).clamp_max(spec.max_finite)
    exp = torch.floor(torch.log2(ax_safe))
    exp = torch.clamp(exp, min=spec.exp_min, max=spec.exp_max)
    step = torch.pow(torch.tensor(2.0, device=x.device, dtype=torch.float32), exp - spec.mant_bits)
    q = torch.round(ax_safe / step) * step
    q = torch.clamp(q, max=spec.max_finite)
    q = torch.where(ax == 0, torch.zeros_like(q), q)
    return (sign * q).to(orig_dtype)


def fake_quant_fp8_scaled(
    x: torch.Tensor,
    fmt: str = "e4m3",
    granularity: str = "tensor",
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    spec = FP8_FORMATS[_canonical_fp8(fmt)]
    g = granularity.lower()
    if g in ("none", "raw", "unscaled"):
        return fake_fp8_core(x, fmt=fmt)
    reduce_dims = _reduce_dims_for_granularity(x, g, channel_dim=channel_dim, state_dim=state_dim)
    xf = x.float()
    amax = xf.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    scale = (amax / spec.max_finite).clamp_min(eps)
    return (fake_fp8_core(xf / scale, fmt=fmt).float() * scale).to(x.dtype)


def fake_quant_fp8_dsq(
    x: torch.Tensor,
    fmt: str = "e4m3",
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
    channel_stat: str = "sqrt_mean_abs",
) -> torch.Tensor:
    spec = FP8_FORMATS[_canonical_fp8(fmt)]
    orig_dtype = x.dtype
    xf = x.float()
    cdim = channel_dim % xf.ndim
    sdim = state_dim % xf.ndim

    if channel_stat.lower() in ("rms", "sqrt_mean_square"):
        s_channel = torch.sqrt(torch.mean(xf.pow(2), dim=sdim, keepdim=True).clamp_min(eps))
    else:
        s_channel = torch.sqrt(torch.mean(xf.abs(), dim=sdim, keepdim=True).clamp_min(eps))
    x_smooth = xf / s_channel.clamp_min(eps)
    s_state = x_smooth.abs().amax(dim=cdim, keepdim=True).clamp_min(eps) / spec.max_finite
    scale = (s_channel * s_state).clamp_min(eps)
    q = fake_fp8_core(xf / scale, fmt=fmt).float() * scale
    return q.to(orig_dtype)


def parse_fp8_method(method: str) -> Tuple[str, str, bool, str]:
    """Return (fmt, granularity, is_dsq, channel_stat)."""
    m = method.lower()
    if not m.startswith("fp8_"):
        raise ValueError(f"Not an FP8 method: {method}")
    parts = m.split("_")
    fmt = _canonical_fp8(parts[1])
    rest = parts[2:]
    if not rest:
        return fmt, "raw", False, "sqrt_mean_abs"
    if rest[0] == "scaled":
        gran = "tensor" if len(rest) == 1 else "_".join(rest[1:])
        return fmt, gran, False, "sqrt_mean_abs"
    if rest[0] in ("tensor", "head", "channel", "state"):
        return fmt, rest[0], False, "sqrt_mean_abs"
    if rest[0] == "dsq":
        stat = "rms" if "rms" in rest[1:] else "sqrt_mean_abs"
        return fmt, "dsq", True, stat
    raise ValueError(f"Bad FP8 method: {method}")


def fake_quant_fp8_state(
    x: torch.Tensor,
    method: str = "fp8_e4m3_scaled_tensor",
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    fmt, gran, is_dsq, stat = parse_fp8_method(method)
    if is_dsq:
        return fake_quant_fp8_dsq(x, fmt=fmt, channel_dim=channel_dim, state_dim=state_dim, eps=eps, channel_stat=stat)
    return fake_quant_fp8_scaled(x, fmt=fmt, granularity=gran, channel_dim=channel_dim, state_dim=state_dim, eps=eps)


# -----------------------------
# Hadamard-enhanced quantization
# -----------------------------


def quantize_state_base(
    x: torch.Tensor,
    method: str,
    bits: int = 8,
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
    qrange: str = "symmetric",
    channel_stat: str = "sqrt_mean_abs",
) -> torch.Tensor:
    m = method.lower()
    if m in ("none", "fp16", "off"):
        return x
    if m.startswith("fp8_"):
        return fake_quant_fp8_state(x, method=m, channel_dim=channel_dim, state_dim=state_dim, eps=eps)
    return fake_quant_int_state(x, method=m, bits=bits, channel_dim=channel_dim, state_dim=state_dim, eps=eps, qrange=qrange, channel_stat=channel_stat)


def quantize_state_hadamard(
    x: torch.Tensor,
    method: str,
    bits: int = 8,
    hadamard_dims: str | Sequence[int] = "state",
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
    qrange: str = "symmetric",
    channel_stat: str = "sqrt_mean_abs",
    hadamard_compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rotate -> quantize -> inverse-rotate state.

    Since normalized Hadamard is self-inverse, the inverse is the same transform.
    This tests whether rotation improves the quantizer/scaler without changing the
    Mamba recurrence itself.
    """
    dims = parse_hadamard_dims(hadamard_dims, channel_dim=channel_dim, state_dim=state_dim)
    if len(dims) == 0:
        return quantize_state_base(x, method=method, bits=bits, channel_dim=channel_dim, state_dim=state_dim, eps=eps, qrange=qrange, channel_stat=channel_stat)
    rotated = x
    for d in dims:
        rotated = hadamard_transform_dim(rotated, d, normalize=True, compute_dtype=hadamard_compute_dtype)
    q_rot = quantize_state_base(rotated, method=method, bits=bits, channel_dim=channel_dim, state_dim=state_dim, eps=eps, qrange=qrange, channel_stat=channel_stat)
    restored = q_rot
    # Inverse. Reverse order is mathematically safest if using multiple axes.
    for d in reversed(dims):
        restored = hadamard_transform_dim(restored, d, normalize=True, compute_dtype=hadamard_compute_dtype)
    return restored.to(x.dtype)


def estimate_scale_count(shape: Sequence[int], method: str, hadamard_dims: str = "none", channel_dim: int = -2, state_dim: int = -1) -> int:
    """Rough scale count for reporting. Hadamard itself does not change scale count."""
    m = method.lower()
    if m in ("none", "fp16", "off"):
        return 0
    if m.startswith("fp8_"):
        fmt, gran, is_dsq, _ = parse_fp8_method(m)
        if gran in ("raw", "none", "unscaled") and not is_dsq:
            return 0
        if is_dsq:
            # DSQ-like: channel factors + state factors.
            ndim = len(shape)
            cdim = channel_dim % ndim
            sdim = state_dim % ndim
            kept_channel = 1
            kept_state = 1
            for i, v in enumerate(shape):
                if i != sdim:
                    kept_channel *= int(v)
                if i != cdim:
                    kept_state *= int(v)
            return kept_channel + kept_state
        reduce_dims = _reduce_dims_for_granularity(torch.empty(tuple(shape)), gran, channel_dim=channel_dim, state_dim=state_dim)
    else:
        if m in ("dsq", "dsq_rms"):
            ndim = len(shape)
            cdim = channel_dim % ndim
            sdim = state_dim % ndim
            kept_channel = 1
            kept_state = 1
            for i, v in enumerate(shape):
                if i != sdim:
                    kept_channel *= int(v)
                if i != cdim:
                    kept_state *= int(v)
            return kept_channel + kept_state
        reduce_dims = _reduce_dims_for_granularity(torch.empty(tuple(shape)), m, channel_dim=channel_dim, state_dim=state_dim)
    kept = 1
    for i, v in enumerate(shape):
        if i not in reduce_dims:
            kept *= int(v)
    return kept
