"""Feature-flagged ``torch.compile`` helpers for DualSystem IDM inference."""

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


class CompiledIDMVideoLoop:
    """Compile the IDM stage-1 video block loop while keeping prepare/finalize eager."""

    def __init__(self, video_backbone: Any, compile_cfg: Any) -> None:
        self.video_backbone = video_backbone
        self.compile_kwargs = torch_compile_kwargs(compile_cfg, default_mode="reduce-overhead")
        self._compiled_fn = None
        self._compile_disabled = False
        self._signature: tuple | None = None
        self._compile_count = 0

    def can_run(
        self,
        vstate: Any,
        *,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> bool:
        if torch.is_grad_enabled():
            return False
        if not getattr(self.video_backbone, "supports_generic_mot_compile", True):
            return False
        if use_gradient_checkpointing or use_gradient_checkpointing_offload:
            return False
        if getattr(vstate, "vace_hints", None) is not None:
            return False
        extras = getattr(vstate, "extras", {})
        if extras.get("use_usp", False):
            return False
        if extras.get("shared_attention_mask") is not None:
            return False
        if extras.get("tea_cache") is not None:
            return False
        return extras.get("animate_adapter") is None

    def run(self, vstate: Any) -> Any:
        if self._compile_disabled:
            return self._run_eager(vstate)

        signature = self._signature_for(vstate)
        compiled_fn = self._compiled_fn
        compiled_now = compiled_fn is None or signature != self._signature
        if compiled_now:
            try:
                compiled_fn = self._compile_for(vstate)
            except Exception as exc:
                return self._fallback_or_raise_eager_error(exc, vstate)

        inputs = self._inputs_for(vstate)
        try:
            new_vx = compiled_fn(*inputs)
        except Exception as exc:
            return self._fallback_or_raise_eager_error(exc, vstate)

        if compiled_now:
            self._compiled_fn = compiled_fn
            self._signature = signature
            self._compile_count += 1
            logger.info(
                "IDM video loop torch.compile prepared (count=%d, kwargs=%s, signature=%s)",
                self._compile_count,
                self.compile_kwargs,
                signature,
            )

        vstate.x = new_vx
        return vstate

    def _disable_compile(self, exc: Exception) -> None:
        self._compiled_fn = None
        self._signature = None
        self._compile_disabled = True
        logger.warning("IDM video loop torch.compile failed, permanently falling back to eager: %s", exc)

    def _fallback_or_raise_eager_error(self, exc: Exception, vstate: Any) -> Any:
        try:
            eager_vstate = self._run_eager(vstate)
        except Exception as eager_exc:
            logger.error(
                "IDM video loop eager fallback failed; original torch.compile exception follows.",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise eager_exc from None
        self._disable_compile(exc)
        return eager_vstate

    def _run_eager(self, vstate: Any) -> Any:
        for block_id in range(self.video_backbone.num_layers):
            vstate = self.video_backbone.run_block(block_id, vstate)
        return vstate

    def _signature_for(self, vstate: Any) -> tuple:
        return (
            _tensor_signature(vstate.x),
            _tensor_signature(vstate.t_mod),
            _tensor_signature(vstate.freqs),
            _tensor_signature(vstate.context),
            _tensor_signature(vstate.context_mask),
            int(self.video_backbone.num_layers),
        )

    def _inputs_for(self, vstate: Any) -> list[Any]:
        inputs: list[Any] = [
            vstate.x,
            vstate.t_mod,
            vstate.freqs,
            vstate.context,
        ]
        if vstate.context_mask is not None:
            inputs.append(vstate.context_mask)
        return inputs

    def _compile_for(self, vstate: Any):
        video_backbone = self.video_backbone
        local_vstate = copy.copy(vstate)
        has_context_mask = vstate.context_mask is not None

        def _video_loop(
            vx: Tensor,
            v_t_mod: Tensor,
            v_freqs: Tensor,
            v_context: Tensor,
            *tail: Tensor,
        ) -> Tensor:
            state = local_vstate
            state.x = vx
            state.t_mod = v_t_mod
            state.freqs = v_freqs
            state.context = v_context
            state.context_mask = tail[0] if has_context_mask else None
            for block_id in range(video_backbone.num_layers):
                state = video_backbone.run_block(block_id, state)
            return state.x

        return torch.compile(_video_loop, **self.compile_kwargs)


class CompiledIDMActionWithVideoCache:
    """Compile IDM stage-2 action denoising against a frozen video K/V cache."""

    def __init__(self, driver: Any, compile_cfg: Any) -> None:
        self.driver = driver
        self.compile_kwargs = torch_compile_kwargs(compile_cfg, default_mode="reduce-overhead")
        self._compiled_fn = None
        self._compile_disabled = False
        self._signature: tuple | None = None
        self._compile_count = 0

    def can_run(
        self,
        astate: Any,
        *,
        video_kv_cache: list[dict[str, Tensor]],
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> bool:
        if torch.is_grad_enabled():
            return False
        if use_gradient_checkpointing or use_gradient_checkpointing_offload:
            return False
        if getattr(self.driver.ab, "variant", None) != "idm":
            return False
        payload = getattr(astate, "payload", None)
        if payload is None or not hasattr(payload, "x_action"):
            return False
        return len(video_kv_cache) == self.driver.num_layers

    def run(
        self,
        astate: Any,
        *,
        video_kv_cache: list[dict[str, Tensor]],
        video_seq_len: int,
        video_tokens_per_frame: int,
    ) -> Any:
        if self._compile_disabled:
            return self._run_eager(
                astate,
                video_kv_cache=video_kv_cache,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
            )

        payload = astate.payload
        action_mask = self._action_mask(
            payload.x_action,
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
        )
        cache_tensors = self._cache_tensors(video_kv_cache, video_seq_len=video_seq_len)
        signature = self._signature_for(payload, action_mask, cache_tensors, video_seq_len, video_tokens_per_frame)
        compiled_fn = self._compiled_fn
        compiled_now = compiled_fn is None or signature != self._signature
        if compiled_now:
            try:
                compiled_fn = self._compile_for(astate, payload.context_mask is not None)
            except Exception as exc:
                return self._fallback_or_raise_eager_error(
                    exc,
                    astate,
                    video_kv_cache=video_kv_cache,
                    video_seq_len=video_seq_len,
                    video_tokens_per_frame=video_tokens_per_frame,
                )

        inputs = self._inputs_for(payload, action_mask, cache_tensors)
        try:
            new_ax = compiled_fn(*inputs)
        except Exception as exc:
            return self._fallback_or_raise_eager_error(
                exc,
                astate,
                video_kv_cache=video_kv_cache,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
            )

        if compiled_now:
            self._compiled_fn = compiled_fn
            self._signature = signature
            self._compile_count += 1
            logger.info(
                "IDM action-cache torch.compile prepared (count=%d, kwargs=%s, signature=%s)",
                self._compile_count,
                self.compile_kwargs,
                signature,
            )

        payload.x_action = new_ax
        return astate

    def _disable_compile(self, exc: Exception) -> None:
        self._compiled_fn = None
        self._signature = None
        self._compile_disabled = True
        logger.warning("IDM action-cache torch.compile failed, permanently falling back to eager: %s", exc)

    def _fallback_or_raise_eager_error(
        self,
        exc: Exception,
        astate: Any,
        *,
        video_kv_cache: list[dict[str, Tensor]],
        video_seq_len: int,
        video_tokens_per_frame: int,
    ) -> Any:
        try:
            eager_astate = self._run_eager(
                astate,
                video_kv_cache=video_kv_cache,
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
            )
        except Exception as eager_exc:
            logger.error(
                "IDM action-cache eager fallback failed; original torch.compile exception follows.",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise eager_exc from None
        self._disable_compile(exc)
        return eager_astate

    def _run_eager(
        self,
        astate: Any,
        *,
        video_kv_cache: list[dict[str, Tensor]],
        video_seq_len: int,
        video_tokens_per_frame: int,
    ) -> Any:
        return self.driver.run_action_with_video_cache(
            astate,
            video_kv_cache=video_kv_cache,
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
        )

    def _action_mask(
        self,
        x_action: Tensor,
        *,
        video_seq_len: int,
        video_tokens_per_frame: int,
    ) -> Optional[Tensor]:
        s_action = int(x_action.shape[1])
        joint_mask = self.driver._build_attention_mask(
            s_video=int(video_seq_len),
            s_action=s_action,
            video_tokens_per_frame=int(video_tokens_per_frame),
            device=x_action.device,
        )
        return None if joint_mask is None else joint_mask[video_seq_len : video_seq_len + s_action, :]

    def _cache_tensors(self, video_kv_cache: list[dict[str, Tensor]], *, video_seq_len: int) -> tuple[Tensor, ...]:
        if len(video_kv_cache) != self.driver.num_layers:
            raise ValueError(f"video_kv_cache must contain {self.driver.num_layers} layers, got {len(video_kv_cache)}.")
        tensors: list[Tensor] = []
        for layer_id, cache in enumerate(video_kv_cache):
            k_video = cache["k"]
            v_video = cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"video_kv_cache[{layer_id}] seq length mismatch: "
                    f"expected {video_seq_len}, got {k_video.shape[1]} and {v_video.shape[1]}."
                )
            tensors.extend([k_video, v_video])
        return tuple(tensors)

    def _signature_for(
        self,
        payload: Any,
        action_mask: Optional[Tensor],
        cache_tensors: tuple[Tensor, ...],
        video_seq_len: int,
        video_tokens_per_frame: int,
    ) -> tuple:
        return (
            _tensor_signature(payload.x_action),
            _tensor_signature(payload.t_mod),
            _tensor_signature(payload.action_freqs),
            _tensor_signature(payload.context),
            _tensor_signature(payload.context_mask),
            _tensor_signature(action_mask),
            tuple(_tensor_signature(tensor) for tensor in cache_tensors),
            int(video_seq_len),
            int(video_tokens_per_frame),
            int(self.driver.num_layers),
        )

    def _inputs_for(
        self,
        payload: Any,
        action_mask: Optional[Tensor],
        cache_tensors: tuple[Tensor, ...],
    ) -> list[Any]:
        inputs: list[Any] = [
            payload.x_action,
            payload.t_mod,
            payload.action_freqs,
            payload.context,
            action_mask,
            *cache_tensors,
        ]
        if payload.context_mask is not None:
            inputs.append(payload.context_mask)
        return inputs

    def _compile_for(self, astate: Any, has_context_mask: bool):
        driver = self.driver
        ab = driver.ab
        layer_tensor_count = 2 * driver.num_layers
        local_astate = copy.copy(astate)
        local_payload = copy.copy(local_astate.payload)
        local_astate.payload = local_payload

        def _action_loop(
            ax: Tensor,
            a_t_mod: Tensor,
            a_freqs: Tensor,
            a_context: Tensor,
            action_mask: Optional[Tensor],
            *tail: Tensor,
        ) -> Tensor:
            """Run one fixed-shape IDM action loop.

            ``action_mask`` may be ``None`` or a tensor; switching between
            those cases changes the compile signature and recompiles. Normal
            fixed-shape inference stays on one side and reuses the graph.
            """

            cache_tensors = tail[:layer_tensor_count]
            context_mask = tail[layer_tensor_count] if has_context_mask else None
            local_payload.x_action = ax
            local_payload.t_mod = a_t_mod
            local_payload.action_freqs = a_freqs
            local_payload.context = a_context
            local_payload.context_mask = context_mask
            for layer_id in range(driver.num_layers):
                q_a, k_a, v_a, apost = ab.pre_attn_at_layer_for_compile(layer_id, local_astate)
                k_video = cache_tensors[2 * layer_id]
                v_video = cache_tensors[2 * layer_id + 1]
                k_cat = torch.cat([k_video, k_a], dim=1)
                v_cat = torch.cat([v_video, v_a], dim=1)
                mixed_a = driver._mixed_attention(q_a, k_cat, v_cat, action_mask)
                ab.post_attn_at_layer_for_compile(layer_id, local_astate, mixed_a.contiguous(), apost)
            return local_payload.x_action

        return torch.compile(_action_loop, **self.compile_kwargs)


__all__ = ["CompiledIDMActionWithVideoCache", "CompiledIDMVideoLoop"]
