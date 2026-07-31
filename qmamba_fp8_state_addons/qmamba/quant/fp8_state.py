from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class FP8Format:
    name: str
    mant_bits: int
    exp_min: int
    exp_max: int
    max_finite: float


FP8_FORMATS = {
    # NVIDIA-style finite E4M3. Max finite usually quoted as 448.
    "e4m3": FP8Format("e4m3", mant_bits=3, exp_min=-6, exp_max=8, max_finite=448.0),
    # IEEE-like E5M2. Max finite usually quoted as 57344.
    "e5m2": FP8Format("e5m2", mant_bits=2, exp_min=-14, exp_max=15, max_finite=57344.0),
}


def _canonical_fmt(fmt: str) -> str:
    f = fmt.lower().replace("fp8_", "")
    if f in ("e4m3fn", "float8_e4m3fn"):
        f = "e4m3"
    if f in ("float8_e5m2",):
        f = "e5m2"
    if f not in FP8_FORMATS:
        raise ValueError(f"Unknown FP8 format {fmt}. Expected one of {list(FP8_FORMATS)}")
    return f


def fake_fp8_core(x: torch.Tensor, fmt: str = "e4m3") -> torch.Tensor:
    """
    Approximate FP8 round-to-nearest fake quantization.

    This is a pure PyTorch numerical simulator; it does not store real torch.float8.
    It includes a simple subnormal region by clamping exponent to exp_min and using
    the corresponding subnormal step. This is enough for PPL comparison and keeps
    the evaluator portable across torch versions.
    """
    spec = FP8_FORMATS[_canonical_fmt(fmt)]
    orig_dtype = x.dtype
    xf = x.float()
    sign = torch.sign(xf)
    ax = xf.abs()

    # Avoid log2(0). Zeros are restored by multiplying with sign and rounding to zero.
    tiny = torch.finfo(torch.float32).tiny
    ax_safe = ax.clamp_min(tiny).clamp_max(spec.max_finite)

    exp = torch.floor(torch.log2(ax_safe))
    exp = torch.clamp(exp, min=spec.exp_min, max=spec.exp_max)
    step = torch.pow(torch.tensor(2.0, device=x.device, dtype=torch.float32), exp - spec.mant_bits)

    q = torch.round(ax_safe / step) * step
    q = torch.clamp(q, max=spec.max_finite)
    q = torch.where(ax == 0, torch.zeros_like(q), q)
    return (sign * q).to(orig_dtype)


def _scale_reduce_dims(x: torch.Tensor, granularity: str, channel_dim: int = -2, state_dim: int = -1) -> Tuple[int, ...]:
    """Return dimensions reduced when computing external FP8 scale."""
    g = granularity.lower()
    ndim = x.ndim
    cdim = channel_dim % ndim
    sdim = state_dim % ndim

    if g in ("none", "raw"):
        return tuple()
    if g in ("tensor", "per_tensor"):
        return tuple(range(ndim))
    if g in ("head", "per_head", "batch_head"):
        # For recurrent Mamba2 state [B,H,P,N], reduce P,N and keep B,H.
        return (cdim, sdim)
    if g in ("channel", "per_channel"):
        # Keep channel P, reduce state N.
        return (sdim,)
    if g in ("state", "per_state"):
        # Keep state N, reduce channel P.
        return (cdim,)
    if g in ("batch", "per_batch"):
        return tuple(d for d in range(ndim) if d != 0)
    raise ValueError(f"Unknown FP8 scale granularity: {granularity}")


def fake_fp8_scaled(
    x: torch.Tensor,
    fmt: str = "e4m3",
    granularity: str = "head",
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Externally scaled FP8: q = fp8_round(x / scale) * scale."""
    spec = FP8_FORMATS[_canonical_fmt(fmt)]
    reduce_dims = _scale_reduce_dims(x, granularity, channel_dim=channel_dim, state_dim=state_dim)
    if len(reduce_dims) == 0:
        return fake_fp8_core(x, fmt=fmt)

    xf = x.float()
    amax = xf.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps)
    scale = (amax / spec.max_finite).clamp_min(eps)
    y = xf / scale
    q = fake_fp8_core(y, fmt=fmt).float() * scale
    return q.to(x.dtype)


def fake_fp8_dsq_like(
    x: torch.Tensor,
    fmt: str = "e4m3",
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
    use_rms_channel: bool = False,
) -> torch.Tensor:
    """
    DSQ-like two-factor external scale, then FP8 instead of INT quantization.

    For recurrent Mamba2 state [B,H,P,N]:
      S_channel[..., P, 1] = sqrt(mean(abs(h[..., P, :]))), by default
      S_state[..., 1, N]   = max_P(abs(h / S_channel)) / fp8_max
      q = fp8_round(h / (S_channel * S_state)) * (S_channel * S_state)

    use_rms_channel=True tests an RMS-like alternative.
    """
    spec = FP8_FORMATS[_canonical_fmt(fmt)]
    xf = x.float()
    cdim = channel_dim % xf.ndim
    sdim = state_dim % xf.ndim

    if use_rms_channel:
        s_channel = torch.sqrt(torch.mean(xf.pow(2), dim=sdim, keepdim=True).clamp_min(eps))
    else:
        s_channel = torch.sqrt(torch.mean(xf.abs(), dim=sdim, keepdim=True).clamp_min(eps))
    x_smooth = xf / s_channel.clamp_min(eps)
    s_state = x_smooth.abs().amax(dim=cdim, keepdim=True).clamp_min(eps) / spec.max_finite
    scale = (s_channel * s_state).clamp_min(eps)
    q = fake_fp8_core(x_smooth / s_state.clamp_min(eps), fmt=fmt).float() * scale
    return q.to(x.dtype)


def parse_fp8_method(method: str) -> Tuple[str, str, bool, bool]:
    """
    Parse method string.

    Supported examples:
      fp8_e4m3
      fp8_e5m2
      fp8_e4m3_scaled_tensor
      fp8_e4m3_scaled_head
      fp8_e4m3_scaled_channel
      fp8_e4m3_scaled_state
      fp8_e4m3_dsq
      fp8_e4m3_dsq_rms
    Returns: (fmt, granularity, is_dsq_like, use_rms_channel)
    """
    m = method.lower()
    if not m.startswith("fp8_"):
        raise ValueError(f"Not an FP8 state method: {method}")
    parts = m.split("_")
    if len(parts) < 2:
        raise ValueError(f"Bad FP8 method: {method}")
    fmt = _canonical_fmt(parts[1])
    rest = parts[2:]
    if not rest:
        return fmt, "raw", False, False
    if rest[0] == "scaled":
        gran = "head" if len(rest) == 1 else "_".join(rest[1:])
        return fmt, gran, False, False
    if rest[0] == "dsq":
        use_rms = "rms" in rest[1:]
        return fmt, "dsq", True, use_rms
    raise ValueError(f"Bad FP8 method: {method}")


def quantize_state_fp8(
    x: torch.Tensor,
    method: str = "fp8_e4m3_scaled_head",
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    fmt, gran, is_dsq, use_rms = parse_fp8_method(method)
    if is_dsq:
        return fake_fp8_dsq_like(x, fmt=fmt, channel_dim=channel_dim, state_dim=state_dim, eps=eps, use_rms_channel=use_rms)
    if gran in ("raw", "none"):
        return fake_fp8_core(x, fmt=fmt)
    return fake_fp8_scaled(x, fmt=fmt, granularity=gran, channel_dim=channel_dim, state_dim=state_dim, eps=eps)


def estimate_state_memory_bits(method: str, state_numel: int, shape: Optional[Tuple[int, ...]] = None, channel_dim: int = -2, state_dim: int = -1) -> int:
    """
    Rough theoretical bits for state storage plus external scales.
    Assumes FP8 payload = 8 bits per element, scale stored as FP16 = 16 bits.
    Raw FP8 has no external scale.
    """
    fmt, gran, is_dsq, _ = parse_fp8_method(method)
    bits = 8 * state_numel
    if shape is None:
        return bits
    if is_dsq:
        ndim = len(shape)
        cdim = channel_dim % ndim
        sdim = state_dim % ndim
        kept_channel = 1
        kept_state = 1
        for i, v in enumerate(shape):
            if i != sdim:
                kept_channel *= v
            if i != cdim:
                kept_state *= v
        return bits + 16 * (kept_channel + kept_state)
    reduce_dims = _scale_reduce_dims(torch.empty(shape), gran, channel_dim=channel_dim, state_dim=state_dim)
    if len(reduce_dims) == 0:
        return bits
    kept = 1
    for i, v in enumerate(shape):
        if i not in reduce_dims:
            kept *= v
    return bits + 16 * kept
