from __future__ import annotations

import torch
import torch.nn.functional as F


def get_mamba2_inproj_slices(mixer):
    """
    Return row slices of the fused Mamba2 in_proj output:
        z0, x0, z, x, B, C, dt
    This matches the mamba_ssm Mamba2 layout used by state-spaces/mamba2-*.
    """
    out_features = mixer.in_proj.out_features
    d_ssm = int(mixer.d_ssm)
    ngroups = int(mixer.ngroups)
    d_state = int(mixer.d_state)
    nheads = int(mixer.nheads)

    d_mlp = (out_features - 2 * d_ssm - 2 * ngroups * d_state - nheads) // 2
    if d_mlp < 0:
        raise ValueError(f"Bad d_mlp={d_mlp}; cannot infer Mamba2 in_proj layout")

    cur = 0
    slices = {}
    slices["z0"] = slice(cur, cur + d_mlp); cur += d_mlp
    slices["x0"] = slice(cur, cur + d_mlp); cur += d_mlp
    slices["z"] = slice(cur, cur + d_ssm); cur += d_ssm
    slices["x"] = slice(cur, cur + d_ssm); cur += d_ssm
    b_width = ngroups * d_state
    slices["B"] = slice(cur, cur + b_width); cur += b_width
    slices["C"] = slice(cur, cur + b_width); cur += b_width
    slices["dt"] = slice(cur, cur + nheads); cur += nheads

    if cur != out_features:
        raise ValueError(f"slice mismatch: cur={cur}, out_features={out_features}")
    slices["d_mlp"] = d_mlp
    return slices


def sga_row_indices(mixer, device=None):
    slices = get_mamba2_inproj_slices(mixer)
    rows = []
    for name in ["B", "C", "dt"]:
        s = slices[name]
        rows.extend(range(s.start, s.stop))
    return torch.tensor(rows, dtype=torch.long, device=device)


def has_sga_delta(mixer) -> bool:
    return hasattr(mixer, "qmamba_sga_delta_w") and hasattr(mixer, "qmamba_sga_rows")


def attach_sga_delta(mixer, train_dt_bias: bool = True) -> int:
    """
    Freeze fused in_proj and attach small trainable delta parameters only for B/C/dt rows.

    Instead of optimizing the full in_proj.weight, ESR optimizes:
        zxbcdt[..., B/C/dt] += hidden @ delta_w.T + delta_b

    Returns the number of active delta-weight parameters, excluding dt_bias/bias.
    """
    device = mixer.in_proj.weight.device
    in_features = mixer.in_proj.in_features
    rows = sga_row_indices(mixer, device=device)
    n_rows = rows.numel()

    mixer.in_proj.weight.requires_grad_(False)
    if mixer.in_proj.bias is not None:
        mixer.in_proj.bias.requires_grad_(False)

    # Remove stale delta first if this block was reused.
    cleanup_sga_delta(mixer, keep_merged_weight=True)

    mixer.register_buffer("qmamba_sga_rows", rows, persistent=False)
    mixer.qmamba_sga_delta_w = torch.nn.Parameter(
        torch.zeros(n_rows, in_features, device=device, dtype=torch.float32)
    )

    if mixer.in_proj.bias is not None:
        mixer.qmamba_sga_delta_b = torch.nn.Parameter(
            torch.zeros(n_rows, device=device, dtype=torch.float32)
        )
    else:
        mixer.qmamba_sga_delta_b = None

    if hasattr(mixer, "dt_bias") and isinstance(mixer.dt_bias, torch.nn.Parameter) and train_dt_bias:
        mixer.dt_bias.requires_grad_(True)

    return int(n_rows * in_features)


def in_proj_with_sga_delta(mixer, hidden_states: torch.Tensor) -> torch.Tensor:
    """Frozen fused in_proj plus optional trainable SGA delta rows."""
    zxbcdt = mixer.in_proj(hidden_states)
    if not has_sga_delta(mixer):
        return zxbcdt

    rows = mixer.qmamba_sga_rows.to(hidden_states.device)
    delta = F.linear(
        hidden_states.float(),
        mixer.qmamba_sga_delta_w,
        mixer.qmamba_sga_delta_b,
    ).to(zxbcdt.dtype)

    # clone because zxbcdt may be needed by autograd and we only modify selected rows.
    zxbcdt = zxbcdt.clone()
    zxbcdt[..., rows] = zxbcdt[..., rows] + delta
    return zxbcdt


def merge_sga_delta(mixer, cleanup: bool = True) -> None:
    """Merge trained SGA delta rows back into mixer.in_proj.weight/bias."""
    if not has_sga_delta(mixer):
        return

    rows = mixer.qmamba_sga_rows.to(mixer.in_proj.weight.device)
    with torch.no_grad():
        mixer.in_proj.weight.data[rows] += mixer.qmamba_sga_delta_w.to(mixer.in_proj.weight.dtype)
        if mixer.in_proj.bias is not None and getattr(mixer, "qmamba_sga_delta_b", None) is not None:
            mixer.in_proj.bias.data[rows] += mixer.qmamba_sga_delta_b.to(mixer.in_proj.bias.dtype)

    if cleanup:
        cleanup_sga_delta(mixer, keep_merged_weight=True)


def cleanup_sga_delta(mixer, keep_merged_weight: bool = True) -> None:
    """Remove temporary SGA delta params/buffers so saved checkpoints have no unexpected keys."""
    for name in ["qmamba_sga_delta_w", "qmamba_sga_delta_b", "qmamba_sga_rows"]:
        if hasattr(mixer, name):
            try:
                delattr(mixer, name)
            except Exception:
                pass
