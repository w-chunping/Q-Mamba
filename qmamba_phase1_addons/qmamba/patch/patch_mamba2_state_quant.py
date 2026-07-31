import functools
import types

import torch

from qmamba.quant.state_quant import quantize_state


def patch_mamba2_step_state_quant(
    model,
    method: str = "dsq",
    bits: int = 4,
    channel_dim: int = -2,
    state_dim: int = -1,
    quant_position: str = "before",
    verbose: bool = True,
):
    """
    Monkey-patch mamba_ssm.modules.mamba2.Mamba2.step for recurrent state-cache fake quant.

    The installed Mamba2.step signature should be:
        step(self, hidden_states, conv_state, ssm_state)

    For Mamba2, ssm_state shape is [B, H, P, N], where P is headdim/channel
    and N is d_state. DSQ should therefore use channel_dim=-2 and state_dim=-1.

    quant_position:
      before: quantize/dequantize cached state before the next recurrent update.
              This matches h_t^q = A_t Q(h_{t-1}^q) + B_t x_t.
      after:  quantize/dequantize the updated state after the recurrent update.
              This mimics storing the new state in a low-bit cache.
      both:   debugging only; usually unnecessarily harsh.
    """
    if quant_position not in {"before", "after", "both"}:
        raise ValueError(f"Bad quant_position: {quant_position}")

    try:
        from mamba_ssm.modules.mamba2 import Mamba2
    except Exception as e:
        raise RuntimeError("Could not import mamba_ssm.modules.mamba2.Mamba2") from e

    patched = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, Mamba2):
            continue
        if hasattr(module, "_qmamba_original_step"):
            continue

        module._qmamba_original_step = module.step
        module._qmamba_state_quant_cfg = {
            "method": method,
            "bits": int(bits),
            "channel_dim": int(channel_dim),
            "state_dim": int(state_dim),
            "quant_position": quant_position,
            "module_name": module_name,
        }

        @functools.wraps(module.step)
        def wrapped_step(self, hidden_states, conv_state, ssm_state):
            cfg = self._qmamba_state_quant_cfg

            if cfg["method"] != "none" and cfg["quant_position"] in {"before", "both"}:
                # In-place copy keeps generation cache object identity stable.
                ssm_state.copy_(
                    quantize_state(
                        ssm_state,
                        method=cfg["method"],
                        bits=cfg["bits"],
                        channel_dim=cfg["channel_dim"],
                        state_dim=cfg["state_dim"],
                    )
                )

            out, conv_state_out, ssm_state_out = self._qmamba_original_step(hidden_states, conv_state, ssm_state)

            if cfg["method"] != "none" and cfg["quant_position"] in {"after", "both"}:
                ssm_state_out.copy_(
                    quantize_state(
                        ssm_state_out,
                        method=cfg["method"],
                        bits=cfg["bits"],
                        channel_dim=cfg["channel_dim"],
                        state_dim=cfg["state_dim"],
                    )
                )

            return out, conv_state_out, ssm_state_out

        module.step = types.MethodType(wrapped_step, module)
        patched += 1

    if verbose:
        print(f"[Q-Mamba] patched {patched} Mamba2.step modules: method={method}, bits={bits}, position={quant_position}")
    if patched == 0:
        raise RuntimeError("No Mamba2 modules were patched. Check mamba_ssm version / model type.")
    return model
