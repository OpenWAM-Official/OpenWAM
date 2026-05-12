"""Feature-flagged ``torch.compile`` helper for cross-attn ActionDiT."""

from __future__ import annotations

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


class CompiledCrossAttnAction:
    """Compile the action-side cross-attn path while keeping Wan/video eager."""

    def __init__(self, action_backbone: Any, compile_cfg: Any) -> None:
        self.action_backbone = action_backbone
        self.compile_kwargs = torch_compile_kwargs(compile_cfg, default_mode="reduce-overhead")
        self._compiled_fn = None
        self._compile_disabled = False
        self._signature: tuple | None = None
        self._compile_count = 0

    def can_run(
        self,
        *,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> bool:
        if torch.is_grad_enabled():
            return False
        if use_gradient_checkpointing or use_gradient_checkpointing_offload:
            return False
        return getattr(self.action_backbone, "variant", None) == "joint_cross_attn"

    def run(
        self,
        action_tokens: Tensor,
        bridges: dict[int, Tensor],
        timestep: Tensor,
        *,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self._compile_disabled:
            return self._run_eager(action_tokens, bridges, timestep, context=context, context_mask=context_mask)

        bridge_tuple = self.action_backbone.bridge_tuple_from_dict(bridges)
        action_freqs = self.action_backbone._get_rope_freqs(action_tokens.shape[1]).to(device=action_tokens.device)
        signature = self._signature_for(action_tokens, bridge_tuple, timestep, action_freqs, context, context_mask)
        compiled_fn = self._compiled_fn
        compiled_now = compiled_fn is None or signature != self._signature
        if compiled_now:
            try:
                compiled_fn = self._compile_for(context is not None, context_mask is not None)
            except Exception as exc:
                return self._fallback_or_raise_eager_error(
                    exc,
                    action_tokens,
                    bridges,
                    timestep,
                    context=context,
                    context_mask=context_mask,
                )

        inputs = self._inputs_for(action_tokens, bridge_tuple, timestep, action_freqs, context, context_mask)
        try:
            action_pred = compiled_fn(*inputs)
        except Exception as exc:
            return self._fallback_or_raise_eager_error(
                exc,
                action_tokens,
                bridges,
                timestep,
                context=context,
                context_mask=context_mask,
            )

        if compiled_now:
            self._compiled_fn = compiled_fn
            self._signature = signature
            self._compile_count += 1
            logger.info(
                "cross-attn ActionDiT torch.compile prepared (count=%d, kwargs=%s, signature=%s)",
                self._compile_count,
                self.compile_kwargs,
                signature,
            )

        return action_pred

    def _disable_compile(self, exc: Exception) -> None:
        self._compiled_fn = None
        self._signature = None
        self._compile_disabled = True
        logger.warning("cross-attn ActionDiT torch.compile failed, permanently falling back to eager: %s", exc)

    def _fallback_or_raise_eager_error(
        self,
        exc: Exception,
        action_tokens: Tensor,
        bridges: dict[int, Tensor],
        timestep: Tensor,
        *,
        context: Optional[Tensor],
        context_mask: Optional[Tensor],
    ) -> Tensor:
        try:
            eager_pred = self._run_eager(action_tokens, bridges, timestep, context=context, context_mask=context_mask)
        except Exception as eager_exc:
            raise eager_exc from None
        self._disable_compile(exc)
        return eager_pred

    def _run_eager(
        self,
        action_tokens: Tensor,
        bridges: dict[int, Tensor],
        timestep: Tensor,
        *,
        context: Optional[Tensor],
        context_mask: Optional[Tensor],
    ) -> Tensor:
        return self.action_backbone(
            action_tokens,
            bridges,
            timestep,
            context=context,
            context_mask=context_mask,
        )

    def _signature_for(
        self,
        action_tokens: Tensor,
        bridge_tuple: tuple[Tensor, ...],
        timestep: Tensor,
        action_freqs: Tensor,
        context: Optional[Tensor],
        context_mask: Optional[Tensor],
    ) -> tuple:
        return (
            _tensor_signature(action_tokens),
            tuple(_tensor_signature(bridge) for bridge in bridge_tuple),
            _tensor_signature(timestep),
            _tensor_signature(action_freqs),
            _tensor_signature(context),
            _tensor_signature(context_mask),
            tuple(self.action_backbone.bridge_layers),
        )

    def _inputs_for(
        self,
        action_tokens: Tensor,
        bridge_tuple: tuple[Tensor, ...],
        timestep: Tensor,
        action_freqs: Tensor,
        context: Optional[Tensor],
        context_mask: Optional[Tensor],
    ) -> list[Any]:
        inputs: list[Any] = [action_tokens, timestep, action_freqs, *bridge_tuple]
        if context is not None:
            inputs.append(context)
        if context_mask is not None:
            inputs.append(context_mask)
        return inputs

    def _compile_for(self, has_context: bool, has_context_mask: bool):
        action_backbone = self.action_backbone
        bridge_count = len(action_backbone.bridge_layers)

        def _action_forward(action_tokens: Tensor, timestep: Tensor, action_freqs: Tensor, *tail: Tensor) -> Tensor:
            bridge_tuple = tail[:bridge_count]
            arg_idx = bridge_count
            context = tail[arg_idx] if has_context else None
            if has_context:
                arg_idx += 1
            context_mask = tail[arg_idx] if has_context_mask else None
            return action_backbone.forward_with_bridge_tuple(
                action_tokens,
                bridge_tuple,
                timestep,
                context=context,
                context_mask=context_mask,
                action_freqs=action_freqs,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )

        return torch.compile(_action_forward, **self.compile_kwargs)


__all__ = ["CompiledCrossAttnAction"]
