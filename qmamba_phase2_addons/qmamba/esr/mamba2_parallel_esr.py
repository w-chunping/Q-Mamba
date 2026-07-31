from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from qmamba.quant.state_quant import quantize_state
from qmamba.esr.trainable import mamba2_inproj_slices
from qmamba.esr.sga_delta import in_proj_with_sga_delta


def fake_quant_state_ste(h: torch.Tensor, method: str, bits: int) -> torch.Tensor:
    """Fake quant with straight-through estimator for ESR training."""
    if method in {"none", "fp", "fp16", "no"}:
        return h
    q = quantize_state(h, method=method, bits=bits, channel_dim=-2, state_dim=-1)
    return h + (q - h).detach()


def _activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation in {"silu", "swish"}:
        return F.silu(x)
    if activation == "relu":
        return F.relu(x)
    raise ValueError(f"Unsupported activation: {activation}")


def causal_depthwise_conv1d_full(x: torch.Tensor, conv1d, activation: str) -> torch.Tensor:
    """
    Reference causal depthwise conv for Mamba2 xBC.

    x: [B, L, Dconv]
    conv1d.weight: [Dconv, 1, W]
    returns: [B, L, Dconv]
    """
    b, l, d = x.shape
    w = conv1d.weight
    width = w.shape[-1]
    x_ch = rearrange(x, "b l d -> b d l")
    y = F.conv1d(x_ch, w, bias=conv1d.bias, groups=d, padding=width - 1)
    y = y[..., :l]
    y = rearrange(y, "b d l -> b l d")
    return _activation(y, activation).to(dtype=x.dtype)


def parallel_ssm_states(
    dA_log: torch.Tensor,
    dBx: torch.Tensor,
    chunk_size: int = 0,
) -> torch.Tensor:
    """
    Compute full state sequence for the scalar-A Mamba2 recurrence in parallel.

    Recurrence:
        h_t = dA_t * h_{t-1} + dBx_t

    Closed form:
        h_t = sum_{s<=t} exp(prefix_t - prefix_s) * dBx_s,
        prefix_t = sum_{k<=t} log(dA_k)

    Args:
        dA_log: [B, L, H], log dA_t = dt_t * A_h, usually <= 0.
        dBx:    [B, L, H, P, N]
        chunk_size: if >0, compute target timesteps in chunks to reduce [L,L] memory.

    Returns:
        states: [B, L, H, P, N]
    """
    if dA_log.ndim != 3:
        raise ValueError(f"dA_log must be [B,L,H], got {tuple(dA_log.shape)}")
    if dBx.ndim != 5:
        raise ValueError(f"dBx must be [B,L,H,P,N], got {tuple(dBx.shape)}")

    Bsz, L, H = dA_log.shape
    prefix = torch.cumsum(dA_log.float(), dim=1)  # [B,L,H]
    prefix_bhl = prefix.permute(0, 2, 1).contiguous()  # [B,H,L]
    dBx_f = dBx.float()

    if chunk_size is None or chunk_size <= 0 or chunk_size >= L:
        # logw[b,h,t,s] = prefix[b,t,h] - prefix[b,s,h]
        logw = prefix_bhl.unsqueeze(-1) - prefix_bhl.unsqueeze(-2)  # [B,H,L,L]
        mask = torch.tril(torch.ones((L, L), device=dA_log.device, dtype=torch.bool))
        logw = logw.masked_fill(~mask.view(1, 1, L, L), float("-inf"))
        weights = torch.exp(logw)
        states = torch.einsum("bhts,bshpn->bthpn", weights, dBx_f)
        return states

    outs = []
    s_idx = torch.arange(L, device=dA_log.device).view(1, L)
    for t0 in range(0, L, int(chunk_size)):
        t1 = min(t0 + int(chunk_size), L)
        Tc = t1 - t0
        p_t = prefix_bhl[:, :, t0:t1]    # [B,H,Tc]
        p_s = prefix_bhl                 # [B,H,L]
        logw = p_t.unsqueeze(-1) - p_s.unsqueeze(-2)  # [B,H,Tc,L]
        t_idx = torch.arange(t0, t1, device=dA_log.device).view(Tc, 1)
        mask = s_idx <= t_idx  # [Tc,L]
        logw = logw.masked_fill(~mask.view(1, 1, Tc, L), float("-inf"))
        weights = torch.exp(logw)
        states_chunk = torch.einsum("bhts,bshpn->bthpn", weights, dBx_f)
        outs.append(states_chunk)
    return torch.cat(outs, dim=1)


def mamba2_mixer_parallel_esr_forward(
    mixer,
    hidden_states: torch.Tensor,
    state_quant: str = "dsq",
    state_bits: int = 4,
    qmode: str = "esr",
    scan_chunk_size: int = 0,
) -> torch.Tensor:
    """
    Parallel/vectorized q-aware Mamba2 mixer forward for ESR reconstruction.

    qmode:
      none      : full-precision parallel recurrence, useful for validation.
      esr       : one-step ESR approximation:
                  qstate_t = dA_t * Q(fp_state_{t-1}) + dBx_t
      recurrent : true quantized recurrence. This still needs a time loop and is kept only
                  for debugging; use qmode=esr for parallel ESR training.

    This supports ngroups=1 first, which is the layout used by the target LM checkpoints.
    """
    if int(mixer.ngroups) != 1:
        raise NotImplementedError(
            f"parallel ESR forward currently supports ngroups=1 only; got {mixer.ngroups}."
        )
    if qmode not in {"none", "esr", "recurrent"}:
        raise ValueError(f"Unknown qmode: {qmode}")

    dtype = hidden_states.dtype
    bsz, seqlen, _ = hidden_states.shape
    slices = mamba2_inproj_slices(mixer)

    # Use SGA delta when attached; otherwise this is just mixer.in_proj.
    zxbcdt = in_proj_with_sga_delta(mixer, hidden_states)

    z0 = zxbcdt[..., slices.z0]
    x0 = zxbcdt[..., slices.x0]
    z = zxbcdt[..., slices.z]
    xBC = zxbcdt[..., slices.x.start:slices.C.stop]
    dt = zxbcdt[..., slices.dt]

    xBC = causal_depthwise_conv1d_full(xBC, mixer.conv1d, mixer.activation)

    d_ssm = int(mixer.d_ssm)
    d_state = int(mixer.d_state)
    nheads = int(mixer.nheads)
    headdim = int(mixer.headdim)

    x, B, C = torch.split(xBC, [d_ssm, d_state, d_state], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", p=headdim)

    A = -torch.exp(mixer.A_log.float())  # [H]
    D = mixer.D.to(dtype=dtype)
    dt = F.softplus(dt + mixer.dt_bias.to(dtype=dt.dtype))  # [B,L,H]

    # Mamba2 step code uses dBx = dt * B * x.
    dBx = torch.einsum("blh,bln,blhp->blhpn", dt.float(), B.float(), x.float())
    dA_log = dt.float() * A.view(1, 1, nheads)  # [B,L,H]

    fp_states = parallel_ssm_states(dA_log, dBx, chunk_size=scan_chunk_size)  # [B,L,H,P,N]

    if qmode == "none" or state_quant == "none":
        used_states = fp_states
    elif qmode == "esr":
        zero = torch.zeros_like(fp_states[:, :1])
        prev_states = torch.cat([zero, fp_states[:, :-1]], dim=1)
        prev_q = fake_quant_state_ste(prev_states, method=state_quant, bits=state_bits)
        dA = torch.exp(dA_log).view(bsz, seqlen, nheads, 1, 1)
        used_states = prev_q * dA + dBx.float()
    else:
        # Debug-only true recurrent fake quant. This is intentionally not used for optimized ESR.
        q_state = torch.zeros_like(fp_states[:, 0])
        q_states = []
        dA = torch.exp(dA_log)
        for t in range(seqlen):
            prev_q = fake_quant_state_ste(q_state, method=state_quant, bits=state_bits)
            q_state = prev_q * dA[:, t].view(bsz, nheads, 1, 1) + dBx[:, t].float()
            q_states.append(q_state)
        used_states = torch.stack(q_states, dim=1)

    y = torch.einsum("blhpn,bln->blhp", used_states, C.float()).to(dtype)
    y = y + D.view(1, 1, nheads, 1) * x.to(dtype)
    y = rearrange(y, "b l h p -> b l (h p)")

    if not bool(mixer.rmsnorm):
        y = y * _activation(z, mixer.activation)
    else:
        y = mixer.norm(y, z)

    if slices.d_mlp > 0:
        y = torch.cat([F.silu(z0) * x0, y], dim=-1)

    out = mixer.out_proj(y)
    return out
