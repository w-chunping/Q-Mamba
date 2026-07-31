from __future__ import annotations

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


def mamba2_mixer_qaware_forward(
    mixer,
    hidden_states: torch.Tensor,
    state_quant: str = "dsq",
    state_bits: int = 4,
    qmode: str = "esr",
) -> torch.Tensor:
    """
    PyTorch reference forward for a Mamba2 mixer with state fake quantization.

    This is meant for ESR/block reconstruction, not speed.

    qmode:
      esr       : paper-style one-step quantization simulation:
                  qstate_t = dA_t * Q(fp_state_{t-1}) + dBx_t
      recurrent : true quantized recurrence simulation:
                  qstate_t = dA_t * Q(qstate_{t-1}) + dBx_t
      none      : no state quantization, reference recurrent mixer

    Assumption for this first implementation: ngroups == 1, which matches the
    common state-spaces Mamba2 LM checkpoints we are targeting first.
    """
    if int(mixer.ngroups) != 1:
        raise NotImplementedError(
            f"Phase-2 reference qaware forward currently supports ngroups=1 only; got {mixer.ngroups}."
        )
    if qmode not in {"esr", "recurrent", "none"}:
        raise ValueError(f"Unknown qmode: {qmode}")

    dtype = hidden_states.dtype
    bsz, seqlen, _ = hidden_states.shape
    slices = mamba2_inproj_slices(mixer)

    # zxbcdt = mixer.in_proj(hidden_states)  # [B, L, projected]
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
    A = -torch.exp(mixer.A_log.float())  # [H]
    D = mixer.D.to(dtype=dtype)

    dt = F.softplus(dt + mixer.dt_bias.to(dtype=dt.dtype))  # [B, L, H]
    x = rearrange(x, "b l (h p) -> b l h p", p=headdim)

    state = torch.zeros((bsz, nheads, headdim, d_state), device=hidden_states.device, dtype=torch.float32)
    fp_states = []
    q_states = []

    # First compute full-precision states. The ESR approximation uses Q(h_{t-1}),
    # where h is the original full-precision state, not recursively quantized state.
    for t in range(seqlen):
        dt_t = dt[:, t].float()              # [B, H]
        B_t = B[:, t].float()                # [B, N]
        x_t = x[:, t].float()                # [B, H, P]
        dA_t = torch.exp(dt_t * A)           # [B, H]
        dBx_t = torch.einsum("bh,bn,bhp->bhpn", dt_t, B_t, x_t)
        state = state * rearrange(dA_t, "b h -> b h 1 1") + dBx_t
        fp_states.append(state)

    if qmode == "none" or state_quant == "none":
        used_states = fp_states
    elif qmode == "esr":
        zero_state = torch.zeros_like(fp_states[0])
        prev_states = [zero_state] + fp_states[:-1]
        for t in range(seqlen):
            dt_t = dt[:, t].float()
            B_t = B[:, t].float()
            x_t = x[:, t].float()
            dA_t = torch.exp(dt_t * A)
            dBx_t = torch.einsum("bh,bn,bhp->bhpn", dt_t, B_t, x_t)
            prev_q = fake_quant_state_ste(prev_states[t], method=state_quant, bits=state_bits)
            q_state = prev_q * rearrange(dA_t, "b h -> b h 1 1") + dBx_t
            q_states.append(q_state)
        used_states = q_states
    else:  # qmode == recurrent
        q_state = torch.zeros_like(fp_states[0])
        for t in range(seqlen):
            dt_t = dt[:, t].float()
            B_t = B[:, t].float()
            x_t = x[:, t].float()
            dA_t = torch.exp(dt_t * A)
            dBx_t = torch.einsum("bh,bn,bhp->bhpn", dt_t, B_t, x_t)
            prev_q = fake_quant_state_ste(q_state, method=state_quant, bits=state_bits)
            q_state = prev_q * rearrange(dA_t, "b h -> b h 1 1") + dBx_t
            q_states.append(q_state)
        used_states = q_states

    ys = []
    for t, st in enumerate(used_states):
        C_t = C[:, t].float()  # [B, N]
        y = torch.einsum("bhpn,bn->bhp", st, C_t).to(dtype)
        y = y + rearrange(D, "h -> h 1") * x[:, t].to(dtype)
        y = rearrange(y, "b h p -> b (h p)")
        if not bool(mixer.rmsnorm):
            y = y * _activation(z[:, t], mixer.activation)
        ys.append(y)

    y = torch.stack(ys, dim=1)  # [B, L, d_ssm]
    if bool(mixer.rmsnorm):
        # mixer.norm supports [B, L, D] and z [B, L, D] in mamba_ssm versions used here.
        y = mixer.norm(y, z)

    if slices.d_mlp > 0:
        y = torch.cat([F.silu(z0) * x0, y], dim=-1)

    out = mixer.out_proj(y)
    return out
