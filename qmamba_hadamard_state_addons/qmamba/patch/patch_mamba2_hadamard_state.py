from __future__ import annotations

import torch

from qmamba.quant.hadamard_state import quantize_state_hadamard


def patch_mamba2_step_hadamard_state_quant(
    model,
    method: str = "none",
    bits: int = 8,
    hadamard_dims: str = "none",
    quant_position: str = "before",
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
    qrange: str = "symmetric",
    channel_stat: str = "sqrt_mean_abs",
    hadamard_compute_dtype: torch.dtype = torch.float32,
):
    """Patch Mamba2.step to apply Hadamard-enhanced state quantization.

    method can be an INT method: tensor/head/channel/state/dsq
    or an FP8 method: fp8_e4m3_scaled_tensor, fp8_e4m3_dsq, etc.

    hadamard_dims controls rotation before quantization:
      none, state, channel, both, or comma-separated raw dims such as "-1".
    """
    method = method.lower()
    if method in ("none", "fp16", "off") and hadamard_dims.lower() in ("none", "off", ""):
        return model
    if quant_position not in ("before", "after", "both"):
        raise ValueError(f"Bad quant_position={quant_position}")

    from mamba_ssm.modules.mamba2 import Mamba2

    if not hasattr(Mamba2, "_qmamba_orig_step_hadamard_state_quant"):
        Mamba2._qmamba_orig_step_hadamard_state_quant = Mamba2.step

    orig_step = Mamba2._qmamba_orig_step_hadamard_state_quant

    def qstate(ssm_state: torch.Tensor) -> torch.Tensor:
        return quantize_state_hadamard(
            ssm_state,
            method=method,
            bits=bits,
            hadamard_dims=hadamard_dims,
            channel_dim=channel_dim,
            state_dim=state_dim,
            eps=eps,
            qrange=qrange,
            channel_stat=channel_stat,
            hadamard_compute_dtype=hadamard_compute_dtype,
        )

    def patched_step(self, hidden_states, conv_state, ssm_state):
        if quant_position in ("before", "both"):
            ssm_state.copy_(qstate(ssm_state))
        out, conv_state, ssm_state = orig_step(self, hidden_states, conv_state, ssm_state)
        if quant_position in ("after", "both"):
            ssm_state.copy_(qstate(ssm_state))
        return out, conv_state, ssm_state

    Mamba2.step = patched_step
    print(
        f"[HadamardStateQuant] patched Mamba2.step: method={method}, bits={bits}, "
        f"hadamard_dims={hadamard_dims}, position={quant_position}, "
        f"channel_dim={channel_dim}, state_dim={state_dim}, qrange={qrange}"
    )
    return model
