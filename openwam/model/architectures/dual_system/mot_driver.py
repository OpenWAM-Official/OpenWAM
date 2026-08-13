"""DualSystem MoT (Mixture-of-Transformers) driver: per-layer mixed attention
across a video backbone and an action backbone.

Each backbone runs the prefix of one DiT block (``pre_attn_at_layer``) to yield
Q/K/V; the driver concatenates the two modalities along the sequence dimension,
runs a single mixed self-attention with a joint mask, splits the result, and
feeds each slice through that modality's block suffix (``post_attn_at_layer``).
Inspired by FastWAM-Joint's ``MoT.forward`` + ``_build_mot_attention_mask``
(``references/FastWAM/src/fastwam/models/wan22/{mot.py,fastwam_joint.py}``).
The driver owns no parameters — a plain Python class, absent from ``state_dict``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from openwam.model.architectures.utils.common import compute_video_tokens_per_frame
from openwam.model.architectures.utils.mask_modes import (
    ACTION_SEES_VIDEO,
    build_cross_modal_attention_mask,
    set_video_attention_mask_mode,
    validate_attention_mask_mode,
    widen_mask_for_prefix_kv,
)

if TYPE_CHECKING:
    from openwam.model.action_backbone.base import ActionDiTBackbone
    from openwam.model.architectures.base import ActionState
    from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone


class DualSystemMoTDriver:
    """Drives joint self-attention across a video backbone and an action backbone.

    Validates structural compatibility at construction time:

    - ``vb.num_layers == ab.num_layers`` (one joint attention per layer)
    - ``vb.num_heads == ab.num_heads`` and ``vb.head_dim == ab.head_dim``
      (so concatenated Q/K/V can run through a single attention; FastWAM's
      "two experts share the per-head attention space" pattern)

    Hidden dim (``vb.dim`` vs ``ab.dim``) does **not** need to match — each
    backbone owns its own Q/K/V projections that map their residual streams
    into the shared ``num_heads * head_dim`` attention space.

    The cross-modal ``[Sv+Sa, Sv+Sa]`` mask is built once per
    :meth:`run_joint_loop` and reused across layers. ``v↔v`` follows
    ``vb.video_attention_mask_mode``; ``a↔a`` is fully connected; the v↔a
    coupling follows ``attention_mask_mode`` — see :mod:`utils.mask_modes`
    for the four modes (default ``action_sees_video``, the FastWAM-Joint
    layout where video does not see action so video-KV prefill stays correct).
    """

    def __init__(
        self,
        vb: "VideoBackbone",
        ab: "ActionDiTBackbone",
        *,
        mot_checkpoint_mixed_attn: bool = True,
        attention_mask_mode: str = ACTION_SEES_VIDEO,
        video_attention_mask_mode: Optional[str] = None,
    ) -> None:
        if vb.num_layers != ab.num_layers:
            raise ValueError(
                f"DualSystemMoTDriver: video num_layers ({vb.num_layers}) must equal "
                f"action num_layers ({ab.num_layers}) for joint self-attention."
            )
        if vb.num_heads != ab.num_heads:
            raise ValueError(
                f"DualSystemMoTDriver: video num_heads ({vb.num_heads}) must equal "
                f"action num_heads ({ab.num_heads}). Per-head attention layout must match so "
                f"the concatenated Q/K/V can run through a single attention."
            )
        if vb.head_dim != ab.head_dim:
            raise ValueError(
                f"DualSystemMoTDriver: video head_dim ({vb.head_dim}) must equal action head_dim ({ab.head_dim})."
            )
        validate_attention_mask_mode(attention_mask_mode)

        self.vb = vb
        self.ab = ab
        self.num_layers = vb.num_layers
        self.num_heads = vb.num_heads
        self.head_dim = vb.head_dim
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        self.attention_mask_mode = attention_mask_mode

        # Allow the architecture / config to override the video v↔v sub-mode.
        # When None we defer to whatever ``vb.video_attention_mask_mode`` reports.
        set_video_attention_mask_mode(vb, video_attention_mask_mode)

    # ------------------------------------------------------------------
    # Mixed attention
    # ------------------------------------------------------------------

    def _build_attention_mask(
        self,
        s_video: int,
        s_action: int,
        video_tokens_per_frame: int,
        *,
        device: torch.device,
    ) -> Tensor:
        """Build the ``[Sv+Sa, Sv+Sa]`` cross-modal bool mask for the configured mode.

        Delegates to :func:`build_cross_modal_attention_mask`. ``v↔v`` follows
        ``vb.video_attention_mask_mode``; ``a↔a`` is fully connected; ``v↔a``
        follows ``attention_mask_mode`` (see :mod:`utils.mask_modes`).
        """
        return build_cross_modal_attention_mask(
            self.vb,
            s_video=s_video,
            s_action=s_action,
            video_tokens_per_frame=video_tokens_per_frame,
            mode=self.attention_mask_mode,
            device=device,
        )

    def _mixed_attention(
        self,
        q_cat: Tensor,
        k_cat: Tensor,
        v_cat: Tensor,
        attn_mask: Optional[Tensor],
    ) -> Tensor:
        """Single mixed self-attention over the concatenated [v, a] sequence.

        Inputs are ``(B, S, H*D)`` (the layout produced by ``pre_attn_at_layer``
        in both backbones). Output is the same layout. SDPA is used so an
        optional bool ``attn_mask`` (True = keep) can be honored.
        """
        n = self.num_heads
        q = rearrange(q_cat, "b s (n d) -> b n s d", n=n)
        k = rearrange(k_cat, "b s (n d) -> b n s d", n=n)
        v = rearrange(v_cat, "b s (n d) -> b n s d", n=n)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return rearrange(out, "b n s d -> b s (n d)", n=n)

    # ------------------------------------------------------------------
    # Per-layer step
    # ------------------------------------------------------------------

    def _video_tokens_per_frame(self, vstate: "BlockLoopState") -> int:
        """Tokens per video frame, derived from the spatial dims in vstate.

        Delegates to :func:`compute_video_tokens_per_frame`, which returns
        ``h * w`` from the spatial dims populated on ``vstate``.
        """
        return compute_video_tokens_per_frame(vstate, "DualSystemMoTDriver")

    def step(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionState",
        attn_mask: Optional[Tensor] = None,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple["BlockLoopState", "ActionState"]:
        """Run one mixed-attention layer.

        Pulls Q/K/V from each backbone, concatenates along sequence,
        runs mixed attention (optionally checkpointed), splits the
        result, and feeds each slice through its post-attention suffix.

        ``attn_mask`` is the joint ``[Sv+Sa, Sv+Sa]`` mask built once at
        :meth:`run_joint_loop`. When ``None``, the driver falls back to
        per-step construction (used by direct ``step()`` callers in tests).

        When ``use_gradient_checkpointing`` is enabled (and the action
        backbone is in training mode), the entire pre→mixed→post triplet
        runs under :func:`torch.utils.checkpoint.checkpoint`, mirroring
        what the other architectures get from ``vb.run_block``. The inner
        per-layer ``mot_checkpoint_mixed_attn`` is suppressed in that mode
        to avoid nested-checkpoint waste — the outer wrapper already
        recomputes mixed attention.
        """
        if use_gradient_checkpointing and self.ab.training:
            return self._step_checkpointed(
                layer_id, vstate, astate, attn_mask=attn_mask, offload=use_gradient_checkpointing_offload
            )
        return self._step_impl(layer_id, vstate, astate, attn_mask=attn_mask)

    def _step_impl(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionState",
        attn_mask: Optional[Tensor] = None,
        *,
        suppress_inner_attn_ckpt: bool = False,
    ) -> Tuple["BlockLoopState", "ActionState"]:
        """Unwrapped per-layer body. See :meth:`step` for the public entry point.

        INVARIANT (consumed by :meth:`_step_checkpointed`): this method must
        only reassign the two layer-varying tensor fields ``vstate.hidden_states`` and
        ``astate.payload.x_action``. It MUST NOT mutate any layer-invariant
        field of ``vstate`` / ``astate`` in place — concretely, do not append
        to ``vace_hints``, write into ``extras``, or mutate ``context`` /
        ``freqs`` / ``t_mod``. ``_step_checkpointed`` runs this body inside
        ``torch.utils.checkpoint`` against shallow copies of the state
        objects; only ``x`` / ``x_action`` are isolated, everything else is
        shared by reference. In-place writes there would be re-applied on
        the backward recompute and silently corrupt the outer state without
        any test detecting it.
        """
        vb = self.vb
        ab = self.ab

        q_v, k_v, v_v, vpost = vb.pre_attn_at_layer(layer_id, vstate)
        q_a, k_a, v_a, apost = ab.pre_attn_at_layer(layer_id, astate)

        if q_v.dtype != q_a.dtype:
            raise RuntimeError(
                f"DualSystemMoTDriver: dtype mismatch at layer {layer_id} "
                f"(video={q_v.dtype}, action={q_a.dtype}). Both backbones "
                "must produce attention inputs in matching dtype."
            )
        if q_v.device != q_a.device:
            raise RuntimeError(
                f"DualSystemMoTDriver: device mismatch at layer {layer_id} (video={q_v.device}, action={q_a.device})."
            )

        s_video = q_v.shape[1]
        s_action = q_a.shape[1]
        q_cat = torch.cat([q_v, q_a], dim=1)
        k_cat = torch.cat([k_v, k_a], dim=1)
        v_cat = torch.cat([v_v, v_a], dim=1)
        # Contract: `run_joint_loop` pre-builds ``attn_mask`` once per forward and
        # passes it in for every layer. _step_impl does NOT rebuild the mask
        # itself; any direct caller must pass the same pre-built mask.

        if self.mot_checkpoint_mixed_attn and ab.training and not suppress_inner_attn_ckpt:
            mixed = torch.utils.checkpoint.checkpoint(
                self._mixed_attention, q_cat, k_cat, v_cat, attn_mask, use_reentrant=False
            )
        else:
            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attn_mask)

        attn_v, attn_a = mixed.split([s_video, s_action], dim=1)
        vstate = vb.post_attn_at_layer(layer_id, vstate, attn_v.contiguous(), vpost)
        astate = ab.post_attn_at_layer(layer_id, astate, attn_a.contiguous(), apost)
        return vstate, astate

    def _step_checkpointed(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionState",
        *,
        attn_mask: Optional[Tensor],
        offload: bool,
    ) -> Tuple["BlockLoopState", "ActionState"]:
        """Run :meth:`_step_impl` under ``torch.utils.checkpoint``.

        ``_step_impl`` mutates ``vstate.hidden_states`` and ``astate.payload.x_action``
        as it walks the block, which is fine on forward but lethal on
        backward: ``torch.utils.checkpoint`` re-runs the closure during
        recompute, and that second mutation would clobber the post-forward
        value held by the outer state objects, leaving them pointing at a
        recomputed activation from an earlier layer once backward unwinds.

        We avoid that by giving the closure shallow copies of the state
        containers (``copy.copy`` on the dataclass / payload — same field
        references, fresh wrappers). Layer-invariant fields like
        ``context`` / ``freqs`` / ``t_mod`` / ``vace_hints`` / ``extras``
        are still shared by reference (cheap), but the two tensor fields
        the body writes (``x`` / ``x_action``) live on the local copies so
        recompute never touches the outer references the caller and the
        autograd graph hold. After the checkpoint returns we propagate the
        new tensor values onto the outer ``vstate`` / ``astate`` for the
        next layer's iteration in ``run_joint_loop``.
        """
        outer_payload = astate.payload

        def _run(vx: Tensor, ax: Tensor) -> Tuple[Tensor, Tensor]:
            local_vstate = copy.copy(vstate)
            local_astate = copy.copy(astate)
            local_payload = copy.copy(outer_payload)
            local_astate.payload = local_payload
            local_vstate.hidden_states = vx
            local_payload.x_action = ax
            self._step_impl(layer_id, local_vstate, local_astate, attn_mask=attn_mask, suppress_inner_attn_ckpt=True)
            return local_vstate.hidden_states, local_payload.x_action

        vx0 = vstate.hidden_states
        ax0 = outer_payload.x_action

        if offload:
            with torch.autograd.graph.save_on_cpu():
                new_vx, new_ax = torch.utils.checkpoint.checkpoint(_run, vx0, ax0, use_reentrant=False)
        else:
            new_vx, new_ax = torch.utils.checkpoint.checkpoint(_run, vx0, ax0, use_reentrant=False)

        vstate.hidden_states = new_vx
        outer_payload.x_action = new_ax
        return vstate, astate

    def run_joint_loop(
        self,
        vstate: "BlockLoopState",
        astate: "ActionState",
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple["BlockLoopState", "ActionState"]:
        """Run the full per-layer loop over both modalities.

        Builds the joint attention mask once from the initial shapes and
        reuses it across all layers (token counts and tokens-per-frame are
        layer-invariant during one forward). The two ``use_gradient_*``
        flags are forwarded to :meth:`step` so each layer may opt into
        step-level activation checkpointing — see the class docstring of
        :meth:`step` for memory/compute trade-offs.
        """
        # Resolve sequence shapes from the backbone-populated f/h/w fields.
        # ``vstate.hidden_states.shape[1]`` is identical to ``f*tokens_per_frame`` for
        # backbones that carry a 3D ``(B, S, D)`` state (Wan), but for
        # backbones whose ``state.hidden_states`` is natively 5D ``(B, T, H, W, D)``
        # (CosmosPredict25) ``shape[1]`` is just ``T`` — wrong. Going through f and
        # the shared ``compute_video_tokens_per_frame`` helper is the only
        # formulation that works for both layouts.
        s_video = int(vstate.grid_frames) * self._video_tokens_per_frame(vstate)
        payload = astate.payload
        if payload is None or not hasattr(payload, "x_action"):
            raise RuntimeError(
                "DualSystemMoTDriver: astate.payload must expose `x_action` (populated by ActionDiT.prepare_state)."
            )
        s_action = payload.x_action.shape[1]

        attn_mask = self._build_attention_mask(
            s_video=s_video,
            s_action=s_action,
            video_tokens_per_frame=self._video_tokens_per_frame(vstate),
            device=vstate.hidden_states.device,
        )

        # Backbones may prepend prefix K/V tokens (keys without matching query
        # rows — e.g. Cosmos3's cached und text stream). ``_step_impl`` needs no
        # change: it splits the attention output by query lengths, and SDPA
        # handles the resulting rectangular mask.
        attn_mask = widen_mask_for_prefix_kv(attn_mask, vstate)

        for layer_id in range(self.num_layers):
            vstate, astate = self.step(
                layer_id,
                vstate,
                astate,
                attn_mask=attn_mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        return vstate, astate


__all__ = ["DualSystemMoTDriver"]
