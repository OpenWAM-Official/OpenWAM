"""Feature-flagged ``torch.compile`` helper for the tri-system MoT loop."""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import torch
from torch import Tensor

from openwam.model.compile_options import torch_compile_kwargs

logger = logging.getLogger(__name__)


def _tensor_signature(tensor: Optional[Tensor]) -> tuple | None:
    if tensor is None:
        return None
    return (tuple(tensor.shape), str(tensor.dtype), str(tensor.device), bool(tensor.requires_grad))


class CompiledTriSystemMoTLoop:
    """Compile the fixed-shape trimodal MoT loop while keeping setup eager."""

    def __init__(self, driver: Any, compile_cfg: Any) -> None:
        self.compile_kwargs = torch_compile_kwargs(compile_cfg, default_mode="reduce-overhead")
        self.driver = driver
        self._compiled_fn = None
        self._compile_disabled = False
        self._signature: tuple | None = None
        self._compile_count = 0

    def can_run(
        self,
        vstate: Any,
        astate: Any,
        ustate: Any,
        *,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> bool:
        if torch.is_grad_enabled():
            return False
        if use_gradient_checkpointing or use_gradient_checkpointing_offload:
            return False
        if getattr(vstate, "vace_hints", None) is not None:
            return False
        if getattr(vstate, "extras", {}).get("use_usp", False):
            return False
        payload = getattr(astate, "payload", None)
        return payload is not None and hasattr(payload, "x_action") and hasattr(ustate, "und_tokens")

    def run(self, vstate: Any, astate: Any, ustate: Any) -> tuple[Any, Any, Any]:
        if self._compile_disabled:
            return self._run_eager(vstate, astate, ustate)

        payload = astate.payload
        s_video = vstate.x.shape[1]
        s_action = payload.x_action.shape[1]
        s_understanding = ustate.und_tokens.shape[1]
        attn_mask = self.driver._build_attention_mask(
            s_video=s_video,
            s_action=s_action,
            s_understanding=s_understanding,
            video_tokens_per_frame=self.driver._video_tokens_per_frame(vstate),
            device=vstate.x.device,
            und_mask=getattr(ustate, "und_mask", None),
        )

        signature = self._signature_for(vstate, payload, ustate, attn_mask)
        compiled_fn = self._compiled_fn
        compiled_now = compiled_fn is None or signature != self._signature
        if compiled_now:
            try:
                compiled_fn = self._compile_for(vstate, astate, ustate)
            except Exception as exc:
                return self._fallback_or_raise_eager_error(exc, vstate, astate, ustate)

        inputs = self._inputs_for(vstate, payload, ustate, attn_mask)
        try:
            new_vx, new_ax, new_ux = compiled_fn(*inputs)
        except Exception as exc:
            return self._fallback_or_raise_eager_error(exc, vstate, astate, ustate)

        if compiled_now:
            self._compiled_fn = compiled_fn
            self._signature = signature
            self._compile_count += 1
            logger.info(
                "tri-system MoT loop torch.compile prepared (count=%d, kwargs=%s, signature=%s)",
                self._compile_count,
                self.compile_kwargs,
                signature,
            )

        vstate.x = new_vx
        payload.x_action = new_ax
        ustate.und_tokens = new_ux
        return vstate, astate, ustate

    def _disable_compile(self, exc: Exception) -> None:
        self._compiled_fn = None
        self._signature = None
        self._compile_disabled = True
        logger.warning("tri-system MoT loop torch.compile failed, permanently falling back to eager: %s", exc)

    def _fallback_or_raise_eager_error(
        self, exc: Exception, vstate: Any, astate: Any, ustate: Any
    ) -> tuple[Any, Any, Any]:
        try:
            eager_vstate, eager_astate, eager_ustate = self._run_eager(vstate, astate, ustate)
        except Exception as eager_exc:
            logger.error(
                "tri-system MoT loop eager fallback failed; original torch.compile exception follows.",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise eager_exc from None
        self._disable_compile(exc)
        return eager_vstate, eager_astate, eager_ustate

    def _run_eager(self, vstate: Any, astate: Any, ustate: Any) -> tuple[Any, Any, Any]:
        return self.driver.run_joint_loop(vstate, astate, ustate)

    def _signature_for(self, vstate: Any, payload: Any, ustate: Any, attn_mask: Optional[Tensor]) -> tuple:
        return (
            _tensor_signature(vstate.x),
            _tensor_signature(payload.x_action),
            _tensor_signature(ustate.und_tokens),
            _tensor_signature(vstate.t_mod),
            _tensor_signature(vstate.freqs),
            _tensor_signature(vstate.context),
            _tensor_signature(vstate.context_mask),
            _tensor_signature(payload.t_mod),
            _tensor_signature(payload.action_freqs),
            _tensor_signature(payload.context),
            _tensor_signature(payload.context_mask),
            _tensor_signature(attn_mask),
            int(self.driver.num_layers),
        )

    def _inputs_for(self, vstate: Any, payload: Any, ustate: Any, attn_mask: Optional[Tensor]) -> list[Any]:
        inputs: list[Any] = [
            vstate.x,
            payload.x_action,
            ustate.und_tokens,
            vstate.t_mod,
            vstate.freqs,
            vstate.context,
            payload.t_mod,
            payload.action_freqs,
            payload.context,
            attn_mask,
        ]
        if vstate.context_mask is not None:
            inputs.append(vstate.context_mask)
        if payload.context_mask is not None:
            inputs.append(payload.context_mask)
        return inputs

    def _compile_for(self, vstate: Any, astate: Any, ustate: Any):
        driver = self.driver
        local_vstate = copy.copy(vstate)
        local_astate = copy.copy(astate)
        local_ustate = copy.copy(ustate)
        local_payload = copy.copy(astate.payload)
        local_astate.payload = local_payload
        has_v_context_mask = vstate.context_mask is not None
        has_a_context_mask = astate.payload.context_mask is not None

        def _loop(
            vx: Tensor,
            ax: Tensor,
            ux: Tensor,
            v_t_mod: Tensor,
            v_freqs: Tensor,
            v_context: Tensor,
            a_t_mod: Tensor,
            a_freqs: Tensor,
            a_context: Tensor,
            attn_mask: Optional[Tensor],
            *mask_tensors: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            mask_idx = 0
            local_vstate.x = vx
            local_vstate.t_mod = v_t_mod
            local_vstate.freqs = v_freqs
            local_vstate.context = v_context
            if has_v_context_mask:
                local_vstate.context_mask = mask_tensors[mask_idx]
                mask_idx += 1
            else:
                local_vstate.context_mask = None

            local_payload.x_action = ax
            local_payload.t_mod = a_t_mod
            local_payload.action_freqs = a_freqs
            local_payload.context = a_context
            if has_a_context_mask:
                local_payload.context_mask = mask_tensors[mask_idx]
            else:
                local_payload.context_mask = None

            local_ustate.und_tokens = ux

            for layer_id in range(driver.num_layers):
                driver._step_impl_for_compile(layer_id, local_vstate, local_astate, local_ustate, attn_mask=attn_mask)
            return local_vstate.x, local_payload.x_action, local_ustate.und_tokens

        return torch.compile(_loop, **self.compile_kwargs)


__all__ = ["CompiledTriSystemMoTLoop"]
