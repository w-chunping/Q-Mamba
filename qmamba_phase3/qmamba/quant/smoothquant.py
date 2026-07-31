from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_EXCLUDE = ("lm_head", "embedding", "embed_tokens")


def _amax_lastdim(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.detach().float().abs().amax(dim=tuple(range(x.ndim - 1))).clamp_min(eps)


def fake_quant_activation_per_token(x: torch.Tensor, bits: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric dynamic per-token activation fake quant over last dim."""
    if bits <= 0:
        return x
    orig_dtype = x.dtype
    qmax = 2 ** (bits - 1) - 1
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
    q = torch.round(xf / scale).clamp(-qmax, qmax)
    return (q * scale).to(orig_dtype)


def fake_quant_weight_per_out_channel(w: torch.Tensor, bits: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric static per-output-channel weight fake quant."""
    if bits <= 0:
        return w
    orig_dtype = w.dtype
    qmax = 2 ** (bits - 1) - 1
    wf = w.float()
    scale = wf.abs().amax(dim=1, keepdim=True).clamp_min(eps) / qmax
    q = torch.round(wf / scale).clamp(-qmax, qmax)
    return (q * scale).to(orig_dtype)


class SmoothQuantFakeLinear(nn.Module):
    """
    Linear layer with SmoothQuant smoothing and W8A8 fake quantization.

    Original y = x @ W.T + b.
    SmoothQuant rewrites y = (x / s) @ (W * s).T + b, then quantizes
    smoothed activations and smoothed weights.
    """
    def __init__(
        self,
        linear: nn.Linear,
        smooth_scale: torch.Tensor,
        act_bits: int = 8,
        weight_bits: int = 8,
        quantize_activation: bool = True,
        quantize_weight: bool = True,
        cache_quant_weight: bool = True,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.act_bits = int(act_bits)
        self.weight_bits = int(weight_bits)
        self.quantize_activation = bool(quantize_activation)
        self.quantize_weight = bool(quantize_weight)
        self.cache_quant_weight = bool(cache_quant_weight)

        dtype = linear.weight.dtype
        device = linear.weight.device
        scale = smooth_scale.detach().to(device=device, dtype=torch.float32).clamp_min(1e-8)
        if scale.numel() != linear.in_features:
            raise ValueError(f"smooth_scale length {scale.numel()} != in_features {linear.in_features}")
        self.register_buffer("smooth_scale", scale, persistent=True)

        # Store already-smoothed weight: W_smooth[:, j] = W[:, j] * s[j]
        w_smooth = linear.weight.detach().float() * scale.view(1, -1)
        self.weight = nn.Parameter(w_smooth.to(dtype=dtype), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        if self.quantize_weight and self.cache_quant_weight:
            wq = fake_quant_weight_per_out_channel(self.weight.detach(), bits=self.weight_bits)
            self.register_buffer("weight_q", wq, persistent=False)
        else:
            self.weight_q = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, smooth_scale: torch.Tensor, **kwargs):
        return cls(linear, smooth_scale, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_smooth = x / s
        scale = self.smooth_scale.to(device=x.device, dtype=torch.float32)
        x_s = (x.float() / scale.view(*([1] * (x.ndim - 1)), -1)).to(dtype=x.dtype)
        if self.quantize_activation:
            x_s = fake_quant_activation_per_token(x_s, bits=self.act_bits)

        if self.quantize_weight:
            if self.cache_quant_weight and self.weight_q is not None and self.weight_q.device == self.weight.device:
                w = self.weight_q.to(dtype=self.weight.dtype)
            else:
                w = fake_quant_weight_per_out_channel(self.weight, bits=self.weight_bits)
        else:
            w = self.weight
        return F.linear(x_s, w, self.bias)


def should_quantize_linear(name: str, module: nn.Module, target: str = "mixer", include: Optional[List[str]] = None, exclude: Optional[List[str]] = None) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    exclude = list(DEFAULT_EXCLUDE) + list(exclude or [])
    if any(pat in name for pat in exclude):
        return False
    if include:
        return any(fnmatch.fnmatch(name, pat) or pat in name for pat in include)
    target = target.lower()
    if target == "all":
        return True
    if target == "mixer":
        return name.endswith("mixer.in_proj") or name.endswith("mixer.out_proj") or ".mixer.in_proj" in name or ".mixer.out_proj" in name
    if target == "in_proj":
        return name.endswith("mixer.in_proj") or ".mixer.in_proj" in name
    if target == "out_proj":
        return name.endswith("mixer.out_proj") or ".mixer.out_proj" in name
    raise ValueError(f"Unknown SmoothQuant target: {target}")


class SmoothQuantCalibrator:
    def __init__(self, model: nn.Module, target: str = "mixer", include: Optional[List[str]] = None, exclude: Optional[List[str]] = None):
        self.model = model
        self.target = target
        self.include = include
        self.exclude = exclude
        self.handles = []
        self.act_scales: Dict[str, torch.Tensor] = {}
        self.target_names: List[str] = []

    def _hook(self, name: str):
        def fn(module, inputs, output):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            cur = _amax_lastdim(x).cpu()
            if name not in self.act_scales:
                self.act_scales[name] = cur
            else:
                self.act_scales[name] = torch.maximum(self.act_scales[name], cur)
        return fn

    def __enter__(self):
        for name, module in self.model.named_modules():
            if should_quantize_linear(name, module, self.target, self.include, self.exclude):
                self.target_names.append(name)
                self.handles.append(module.register_forward_hook(self._hook(name)))
        print(f"[SmoothQuant] registered {len(self.handles)} Linear hooks, target={self.target}")
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def compute_smooth_scale(linear: nn.Linear, act_absmax: torch.Tensor, alpha: float = 0.5, eps: float = 1e-8, min_scale: float = 1e-5, max_scale: float = 1e5) -> torch.Tensor:
    """SmoothQuant scale s_j = max|x_j|^alpha / max|W_j|^(1-alpha)."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    device = linear.weight.device
    act = act_absmax.detach().float().to(device).clamp_min(eps)
    w_abs = linear.weight.detach().float().abs().amax(dim=0).clamp_min(eps)
    s = act.pow(alpha) / w_abs.pow(1.0 - alpha)
    return s.clamp(min_scale, max_scale)


def _get_parent_and_child(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_smoothquant_fake(
    model: nn.Module,
    act_scales: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    act_bits: int = 8,
    weight_bits: int = 8,
    target: str = "mixer",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    quantize_activation: bool = True,
    quantize_weight: bool = True,
    cache_quant_weight: bool = True,
    verbose: bool = True,
) -> int:
    """Replace target nn.Linear modules with SmoothQuantFakeLinear in-place."""
    names = []
    for name, module in model.named_modules():
        if should_quantize_linear(name, module, target, include, exclude):
            names.append(name)

    replaced = 0
    missing = []
    for name in names:
        parent, child = _get_parent_and_child(model, name)
        linear = getattr(parent, child)
        if not isinstance(linear, nn.Linear):
            continue
        if name not in act_scales:
            missing.append(name)
            continue
        s = compute_smooth_scale(linear, act_scales[name], alpha=alpha)
        wrapper = SmoothQuantFakeLinear.from_linear(
            linear,
            s,
            act_bits=act_bits,
            weight_bits=weight_bits,
            quantize_activation=quantize_activation,
            quantize_weight=quantize_weight,
            cache_quant_weight=cache_quant_weight,
        )
        setattr(parent, child, wrapper)
        replaced += 1

    if verbose:
        print(f"[SmoothQuant] replaced {replaced}/{len(names)} target Linear modules with fake W{weight_bits}A{act_bits}, alpha={alpha}")
        if missing:
            print(f"[SmoothQuant] missing activation scales for {len(missing)} modules. First few: {missing[:5]}")
    return replaced


def save_sq_scales(path: str, act_scales: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> None:
    payload = {"act_scales": {k: v.cpu() for k, v in act_scales.items()}, "meta": meta or {}}
    torch.save(payload, path)


def load_sq_scales(path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "act_scales" in payload:
        return payload["act_scales"], payload.get("meta", {})
    if isinstance(payload, dict):
        return payload, {}
    raise ValueError(f"Unsupported SmoothQuant scale file: {path}")
