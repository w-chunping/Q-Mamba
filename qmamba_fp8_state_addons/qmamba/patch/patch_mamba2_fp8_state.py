from __future__ import annotations

import torch

from qmamba.quant.fp8_state import quantize_state_fp8


def _fallback_int_quantize(ssm_state, method, bits, channel_dim=-2, state_dim=-1, eps=1e-8):
    # Optional compatibility with your existing DSQ/state_quant implementation.
    from qmamba.quant.state_quant import quantize_state
    try:
        return quantize_state(ssm_state, method=method, bits=bits, channel_dim=channel_dim, state_dim=state_dim, eps=eps)
    except TypeError:
        # Older version may not expose dims.
        return quantize_state(ssm_state, method=method, bits=bits)


def patch_mamba2_step_mixed_state_quant(
    model,
    method: str = "none",
    bits: int = 4,
    quant_position: str = "before",
    channel_dim: int = -2,
    state_dim: int = -1,
    eps: float = 1e-8,
):
    """
    Patch Mamba2.step to fake-quantize recurrent ssm_state.

    Supports:
      none
      tensor/channel/state/dsq       via existing qmamba.quant.state_quant
      fp8_e4m3, fp8_e5m2
      fp8_e4m3_scaled_tensor/head/channel/state
      fp8_e5m2_scaled_tensor/head/channel/state
      fp8_e4m3_dsq, fp8_e5m2_dsq
      fp8_e4m3_dsq_rms, fp8_e5m2_dsq_rms

    quant_position:
      before: quantize cached state before step update/load simulation
      after:  quantize updated state after step/store simulation
      both:   both before and after
    """
    method = method.lower()
    if method == "none":
        return model
    if quant_position not in ("before", "after", "both"):
        raise ValueError(f"Bad quant_position: {quant_position}")

    from mamba_ssm.modules.mamba2 import Mamba2

    if not hasattr(Mamba2, "_qmamba_orig_step_mixed_state_quant"):
        Mamba2._qmamba_orig_step_mixed_state_quant = Mamba2.step

    orig_step = Mamba2._qmamba_orig_step_mixed_state_quant

    def qstate(ssm_state: torch.Tensor) -> torch.Tensor:
        if method.startswith("fp8_"):
            return quantize_state_fp8(ssm_state, method=method, channel_dim=channel_dim, state_dim=state_dim, eps=eps)
        return _fallback_int_quantize(ssm_state, method=method, bits=bits, channel_dim=channel_dim, state_dim=state_dim, eps=eps)

    def patched_step(self, hidden_states, conv_state, ssm_state):
        if quant_position in ("before", "both"):
            ssm_state.copy_(qstate(ssm_state))
        out, conv_state, ssm_state = orig_step(self, hidden_states, conv_state, ssm_state)
        if quant_position in ("after", "both"):
            ssm_state.copy_(qstate(ssm_state))
        return out, conv_state, ssm_state

    Mamba2.step = patched_step
    print(
        f"[FP8/StateQuant] patched Mamba2.step: method={method}, bits={bits}, "
        f"position={quant_position}, channel_dim={channel_dim}, state_dim={state_dim}"
    )
    return model
