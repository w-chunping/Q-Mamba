from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


@dataclass
class InProjSlices:
    d_mlp: int
    z0: slice
    x0: slice
    z: slice
    x: slice
    B: slice
    C: slice
    dt: slice
    train_bcdt_rows: torch.Tensor


def get_layers(model) -> list[nn.Module]:
    """Return MambaLMHeadModel backbone layers for common mamba_ssm versions."""
    candidates = [
        ("backbone", "layers"),
        ("model", "layers"),
        ("decoder", "layers"),
    ]
    for parent_name, layer_name in candidates:
        parent = getattr(model, parent_name, None)
        if parent is not None and hasattr(parent, layer_name):
            return list(getattr(parent, layer_name))
    if hasattr(model, "layers"):
        return list(model.layers)
    raise AttributeError("Could not find model layers. Expected model.backbone.layers for mamba_ssm.")


def get_mixer(block: nn.Module) -> nn.Module:
    if hasattr(block, "mixer"):
        return block.mixer
    # Fallback: find first Mamba2 module inside block.
    try:
        from mamba_ssm.modules.mamba2 import Mamba2
        for m in block.modules():
            if isinstance(m, Mamba2):
                return m
    except Exception:
        pass
    raise AttributeError("Could not find Mamba2 mixer in block.")


def mamba2_inproj_slices(mixer: nn.Module) -> InProjSlices:
    """
    Slices for Mamba2.in_proj output in the installed mamba_ssm layout:
      zxbcdt = [z0, x0, z, xBC, dt]
      xBC    = [x, B, C]

    The trainable ESR/SGA rows are B, C, and dt.
    """
    out_dim = mixer.in_proj.weight.shape[0]
    d_ssm = int(mixer.d_ssm)
    ngroups = int(mixer.ngroups)
    d_state = int(mixer.d_state)
    nheads = int(mixer.nheads)
    d_mlp = (out_dim - 2 * d_ssm - 2 * ngroups * d_state - nheads) // 2
    if d_mlp < 0:
        raise ValueError(f"Bad d_mlp={d_mlp}; cannot infer in_proj layout.")

    cur = 0
    z0 = slice(cur, cur + d_mlp); cur += d_mlp
    x0 = slice(cur, cur + d_mlp); cur += d_mlp
    z = slice(cur, cur + d_ssm); cur += d_ssm
    x = slice(cur, cur + d_ssm); cur += d_ssm
    B = slice(cur, cur + ngroups * d_state); cur += ngroups * d_state
    C = slice(cur, cur + ngroups * d_state); cur += ngroups * d_state
    dt = slice(cur, cur + nheads); cur += nheads
    if cur != out_dim:
        raise ValueError(f"Slice inference mismatch: cur={cur}, out_dim={out_dim}")

    rows = torch.cat([
        torch.arange(B.start, B.stop),
        torch.arange(C.start, C.stop),
        torch.arange(dt.start, dt.stop),
    ])
    return InProjSlices(d_mlp=d_mlp, z0=z0, x0=x0, z=z, x=x, B=B, C=C, dt=dt, train_bcdt_rows=rows)


def freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def _register_row_mask(param: torch.nn.Parameter, rows: torch.Tensor) -> None:
    """Register a gradient mask once. Rows not in `rows` get zero gradient."""
    if getattr(param, "_qmamba_has_row_mask", False):
        return
    mask = torch.zeros_like(param)
    if param.ndim == 2:
        mask[rows.to(param.device), :] = 1
    elif param.ndim == 1:
        mask[rows.to(param.device)] = 1
    else:
        raise ValueError(f"Unsupported masked param shape: {param.shape}")

    def hook(grad):
        return grad * mask.to(grad.device)

    param.register_hook(hook)
    param._qmamba_has_row_mask = True


def set_trainable_for_mixer_esr(mixer: nn.Module, mode: str = "sga") -> List[nn.Parameter]:
    """
    Enable trainable params for one Mamba2 mixer.

    mode:
      sga              : B/C/dt rows of in_proj + dt_bias
      sga_conv         : sga + conv1d parameters
      sga_conv_d       : sga + conv1d + D
      all_mixer        : all mixer parameters

    Note: B/C/dt are row slices of one fused in_proj, so we use gradient masks.
    """
    mode = mode.lower()
    freeze_all(mixer)

    params: List[nn.Parameter] = []
    if mode == "all_mixer":
        for p in mixer.parameters():
            p.requires_grad_(True)
            params.append(p)
        return params

    slices = mamba2_inproj_slices(mixer)
    rows = slices.train_bcdt_rows.to(mixer.in_proj.weight.device)

    mixer.in_proj.weight.requires_grad_(True)
    _register_row_mask(mixer.in_proj.weight, rows)
    params.append(mixer.in_proj.weight)

    if getattr(mixer.in_proj, "bias", None) is not None:
        mixer.in_proj.bias.requires_grad_(True)
        _register_row_mask(mixer.in_proj.bias, rows)
        params.append(mixer.in_proj.bias)

    # dt_bias directly controls Delta discretization; include it in SGA.
    if hasattr(mixer, "dt_bias") and isinstance(mixer.dt_bias, torch.nn.Parameter):
        mixer.dt_bias.requires_grad_(True)
        params.append(mixer.dt_bias)

    if mode in {"sga_conv", "sga_conv_d"}:
        if hasattr(mixer, "conv1d"):
            for p in mixer.conv1d.parameters():
                p.requires_grad_(True)
                params.append(p)

    if mode == "sga_conv_d":
        if hasattr(mixer, "D") and isinstance(mixer.D, torch.nn.Parameter):
            mixer.D.requires_grad_(True)
            params.append(mixer.D)

    if mode not in {"sga", "sga_conv", "sga_conv_d"}:
        raise ValueError(f"Unknown trainable mode: {mode}")

    # Remove duplicates while preserving order.
    seen = set()
    uniq = []
    for p in params:
        if id(p) not in seen:
            uniq.append(p)
            seen.add(id(p))
    return uniq


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def describe_mixer_slices(mixer: nn.Module) -> Dict[str, Tuple[int, int]]:
    s = mamba2_inproj_slices(mixer)
    return {
        "z0": (s.z0.start, s.z0.stop),
        "x0": (s.x0.start, s.x0.stop),
        "z": (s.z.start, s.z.stop),
        "x": (s.x.start, s.x.stop),
        "B": (s.B.start, s.B.stop),
        "C": (s.C.start, s.C.stop),
        "dt": (s.dt.start, s.dt.stop),
    }
