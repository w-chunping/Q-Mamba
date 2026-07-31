import torch


def _qmax(bits: int) -> int:
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    return 2 ** (bits - 1) - 1


def _canonical_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else ndim + dim


def _fake_quant_symmetric_float(x_float: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = _qmax(bits)
    scale = scale.clamp_min(1e-8)
    q = torch.round(x_float / scale).clamp(-qmax, qmax)
    return q * scale


def fake_quant_per_tensor(h: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Per state-matrix MinMax fake quant over the last two dims: [..., P, N]."""
    orig_dtype = h.dtype
    x = h.float()
    qmax = _qmax(bits)
    scale = x.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8) / qmax
    return _fake_quant_symmetric_float(x, scale, bits).to(orig_dtype)


def fake_quant_per_channel(h: torch.Tensor, bits: int = 4, channel_dim: int = -2, state_dim: int = -1) -> torch.Tensor:
    """
    MinMax per-channel fake quant.

    For Mamba2 recurrent ssm_state, the expected shape is [B, H, P, N].
    P is channel/headdim, N is state dimension. This uses one scale per P,
    reducing over N only, for each leading [B, H, ...] state matrix.
    """
    orig_dtype = h.dtype
    x = h.float().movedim((channel_dim, state_dim), (-2, -1))  # [..., P, N]
    qmax = _qmax(bits)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax  # [..., P, 1]
    y = _fake_quant_symmetric_float(x, scale, bits)
    return y.movedim((-2, -1), (channel_dim, state_dim)).to(orig_dtype)


def fake_quant_per_state(h: torch.Tensor, bits: int = 4, channel_dim: int = -2, state_dim: int = -1) -> torch.Tensor:
    """MinMax per-state-dimension fake quant, one scale per N, reducing over P."""
    orig_dtype = h.dtype
    x = h.float().movedim((channel_dim, state_dim), (-2, -1))  # [..., P, N]
    qmax = _qmax(bits)
    scale = x.abs().amax(dim=-2, keepdim=True).clamp_min(1e-8) / qmax  # [..., 1, N]
    y = _fake_quant_symmetric_float(x, scale, bits)
    return y.movedim((-2, -1), (channel_dim, state_dim)).to(orig_dtype)


def fake_quant_dsq(
    h: torch.Tensor,
    bits: int = 4,
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Decoupled Scale Quantization (DSQ) fake quant.

    Expected Mamba2 recurrent ssm_state shape: [B, H, P, N].
      P: channel/headdim dimension
      N: state dimension

    S_channel[..., i, 1] = sqrt(mean(abs(h[..., i, :])))
    S_state[..., 1, j]   = max_i(abs(h[..., i, j] / S_channel[..., i, 1])) / qmax
    scale                = S_channel * S_state
    """
    orig_dtype = h.dtype
    x = h.float().movedim((channel_dim, state_dim), (-2, -1))  # [..., P, N]
    qmax = _qmax(bits)

    s_channel = torch.sqrt(x.abs().mean(dim=-1, keepdim=True).clamp_min(eps))  # [..., P, 1]
    x_smooth = x / s_channel
    s_state = x_smooth.abs().amax(dim=-2, keepdim=True).clamp_min(eps) / qmax  # [..., 1, N]

    scale = (s_channel * s_state).clamp_min(eps)
    y = _fake_quant_symmetric_float(x, scale, bits)
    return y.movedim((-2, -1), (channel_dim, state_dim)).to(orig_dtype)


def quantize_state(
    h: torch.Tensor,
    method: str = "none",
    bits: int = 4,
    channel_dim: int = -2,
    state_dim: int = -1,
) -> torch.Tensor:
    method = method.lower()
    if method in {"none", "fp", "fp16", "no"}:
        return h
    if method in {"tensor", "per_tensor"}:
        return fake_quant_per_tensor(h, bits=bits)
    if method in {"channel", "per_channel", "minmax"}:
        return fake_quant_per_channel(h, bits=bits, channel_dim=channel_dim, state_dim=state_dim)
    if method in {"state", "per_state"}:
        return fake_quant_per_state(h, bits=bits, channel_dim=channel_dim, state_dim=state_dim)
    if method == "dsq":
        return fake_quant_dsq(h, bits=bits, channel_dim=channel_dim, state_dim=state_dim)
    raise ValueError(f"Unknown state quantization method: {method}")
