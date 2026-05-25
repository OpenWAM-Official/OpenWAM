"""MoTJointDriver: per-layer mixed attention across video and action streams.

Inspired by FastWAM-Joint's ``MoT.forward`` + ``FastWAMJoint._build_mot_attention_mask``
(``references/FastWAM/src/fastwam/models/wan22/mot.py`` and
``fastwam_joint.py:29-49``). Each backbone runs the prefix of one of its
DiT blocks (``pre_attn_at_layer``) which yields Q/K/V; the driver
concatenates the two modalities along the sequence dimension, runs a
single mixed self-attention with a ``[Sv+Sa, Sv+Sa]`` joint mask, splits
the result back, and feeds each slice through that modality's block
suffix (``post_attn_at_layer``).

The driver owns no parameters — it is a plain Python class and does not
appear in ``state_dict``.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from openwam.model.architectures._mot_utils import compute_video_tokens_per_frame

if TYPE_CHECKING:
    from openwam.model.action_backbone.backbone import ActionBackbone
    from openwam.model.base import ActionState
    from openwam.model.video_backbone.adapter import BlockLoopState, VideoBackbone

logger = logging.getLogger(__name__)


_VALID_ATTENTION_MASK_MODES = ("bidirectional", "joint")


class MoTJointDriver:
    """Drives joint self-attention across a video backbone and an action backbone.

    Validates structural compatibility at construction time:

    - ``vb.num_layers == ab.num_layers`` (one joint attention per layer)
    - ``vb.num_heads == ab.num_heads`` and ``vb.head_dim == ab.head_dim``
      (so concatenated Q/K/V can run through a single attention; FastWAM's
      "two experts share the per-head attention space" pattern)

    Hidden dim (``vb.dim`` vs ``ab.dim``) does **not** need to match — each
    backbone owns its own Q/K/V projections that map their residual streams
    into the shared ``num_heads * head_dim`` attention space.

    Two ``attention_mask_mode`` values are supported:

    - ``bidirectional``: pass ``None`` to SDPA — full v↔a coupling, fastest
      kernel selection. Useful for ablations and tests.
    - ``joint``: build the FastWAM-Joint mask
      ([fastwam_joint.py:29-49](references/FastWAM/src/fastwam/models/wan22/fastwam_joint.py#L29)),
      a ``[Sv+Sa, Sv+Sa]`` bool layout where:

        - ``v↔v`` is determined by ``vb.video_attention_mask_mode``
          (``bidirectional`` / ``per_frame_causal`` / ``first_frame_causal``);
        - ``a↔a`` is fully connected;
        - ``a→v`` is fully connected (action queries see all video keys);
        - ``v→a`` is **off** (video queries do not see action keys —
          this is what makes optional video-KV prefill correct).

      The mask is built once per :meth:`run_joint_loop` from the shapes in
      ``vstate`` and reused across layers.
    """

    def __init__(
        self,
        vb: "VideoBackbone",
        ab: "ActionBackbone",
        *,
        mot_checkpoint_mixed_attn: bool = True,
        attention_mask_mode: str = "joint",
        video_attention_mask_mode: Optional[str] = None,
    ) -> None:
        if vb.num_layers != ab.num_layers:
            raise ValueError(
                f"MoTJointDriver: video num_layers ({vb.num_layers}) must equal "
                f"action num_layers ({ab.num_layers}) for joint self-attention."
            )
        if vb.num_heads != ab.num_heads:
            raise ValueError(
                f"MoTJointDriver: video num_heads ({vb.num_heads}) must equal "
                f"action num_heads ({ab.num_heads}). Per-head attention layout must match so "
                f"the concatenated Q/K/V can run through a single attention."
            )
        if vb.head_dim != ab.head_dim:
            raise ValueError(
                f"MoTJointDriver: video head_dim ({vb.head_dim}) must equal action head_dim ({ab.head_dim})."
            )
        if attention_mask_mode not in _VALID_ATTENTION_MASK_MODES:
            raise ValueError(
                f"MoTJointDriver: unknown attention_mask_mode '{attention_mask_mode}'. "
                f"Choose from: {_VALID_ATTENTION_MASK_MODES}."
            )

        self.vb = vb
        self.ab = ab
        self.num_layers = vb.num_layers
        self.num_heads = vb.num_heads
        self.head_dim = vb.head_dim
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        self.attention_mask_mode = attention_mask_mode

        # Allow the architecture / config to override the video v↔v sub-mode.
        # When None we defer to whatever ``vb.video_attention_mask_mode`` reports.
        if video_attention_mask_mode is not None:
            try:
                vb.video_attention_mask_mode = video_attention_mask_mode  # type: ignore[misc]
            except AttributeError:
                logger.warning(
                    "video_attention_mask_mode='%s' supplied to MoTJointDriver but "
                    "%s does not expose a settable property; falling back to %s.",
                    video_attention_mask_mode,
                    type(vb).__name__,
                    vb.video_attention_mask_mode,
                )

    # ------------------------------------------------------------------
    # Mixed attention
    # ------------------------------------------------------------------

    def _build_joint_mask(
        self,
        s_video: int,
        s_action: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        """Build the FastWAM-Joint ``[Sv+Sa, Sv+Sa]`` bool attention mask.

        Layout (rows = queries, cols = keys; ``True`` means "attend to"):

        - ``[:Sv, :Sv]`` = ``vb.build_video_to_video_mask(...)``
        - ``[Sv:, Sv:]`` = True  (action↔action)
        - ``[Sv:, :Sv]`` = True  (action queries → all video keys)
        - ``[:Sv, Sv:]`` = False (video queries → no action keys)
        """
        total = s_video + s_action
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:s_video, :s_video] = self.vb.build_video_to_video_mask(
            video_seq_len=s_video,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[s_video:, s_video:] = True
        mask[s_video:, :s_video] = True
        # ``mask[:s_video, s_video:]`` stays False — video doesn't see action.
        return mask

    def _build_attention_mask(
        self,
        s_video: int,
        s_action: int,
        video_tokens_per_frame: int,
        *,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Return the SDPA attn_mask for the configured mode.

        ``bidirectional`` returns ``None`` so SDPA picks the fastest fused
        kernel (semantically equivalent to a fully-True mask).
        """
        if self.attention_mask_mode == "bidirectional":
            return None
        if self.attention_mask_mode == "joint":
            return self._build_joint_mask(
                s_video=s_video,
                s_action=s_action,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        raise RuntimeError(f"unhandled attention_mask_mode '{self.attention_mask_mode}'")

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
        return compute_video_tokens_per_frame(vstate, "MoTJointDriver")

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
        only reassign the two layer-varying tensor fields ``vstate.x`` and
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
                f"MoTJointDriver: dtype mismatch at layer {layer_id} "
                f"(video={q_v.dtype}, action={q_a.dtype}). Both backbones "
                "must produce attention inputs in matching dtype."
            )
        if q_v.device != q_a.device:
            raise RuntimeError(
                f"MoTJointDriver: device mismatch at layer {layer_id} (video={q_v.device}, action={q_a.device})."
            )

        s_video = q_v.shape[1]
        s_action = q_a.shape[1]
        q_cat = torch.cat([q_v, q_a], dim=1)
        k_cat = torch.cat([k_v, k_a], dim=1)
        v_cat = torch.cat([v_v, v_a], dim=1)
        # Contract: `run_joint_loop` pre-builds ``attn_mask`` once per forward and
        # passes it in for every layer. ``attn_mask=None`` is the SDPA "no mask"
        # signal — legitimate only when ``attention_mask_mode='bidirectional'``.
        # _step_impl does NOT rebuild the mask itself; any direct caller must follow
        # the same contract.

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

    def _step_impl_for_compile(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionState",
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple["BlockLoopState", "ActionState"]:
        """Compile-oriented per-layer body with tuple post-attention state."""
        vb = self.vb
        ab = self.ab

        v_pre = getattr(vb, "pre_attn_at_layer_for_compile", vb.pre_attn_at_layer)
        a_pre = getattr(ab, "pre_attn_at_layer_for_compile", ab.pre_attn_at_layer)
        v_post = getattr(vb, "post_attn_at_layer_for_compile", vb.post_attn_at_layer)
        a_post = getattr(ab, "post_attn_at_layer_for_compile", ab.post_attn_at_layer)

        q_v, k_v, v_v, vpost = v_pre(layer_id, vstate)
        q_a, k_a, v_a, apost = a_pre(layer_id, astate)

        if q_v.dtype != q_a.dtype:
            raise RuntimeError(
                f"MoTJointDriver: dtype mismatch at layer {layer_id} "
                f"(video={q_v.dtype}, action={q_a.dtype}). Both backbones "
                "must produce attention inputs in matching dtype."
            )
        if q_v.device != q_a.device:
            raise RuntimeError(
                f"MoTJointDriver: device mismatch at layer {layer_id} (video={q_v.device}, action={q_a.device})."
            )

        s_video = q_v.shape[1]
        s_action = q_a.shape[1]
        q_cat = torch.cat([q_v, q_a], dim=1)
        k_cat = torch.cat([k_v, k_a], dim=1)
        v_cat = torch.cat([v_v, v_a], dim=1)
        # Contract: same as ``_step_impl`` — caller pre-builds ``attn_mask``.

        mixed = self._mixed_attention(q_cat, k_cat, v_cat, attn_mask)
        attn_v, attn_a = mixed.split([s_video, s_action], dim=1)
        vstate = v_post(layer_id, vstate, attn_v.contiguous(), vpost)
        astate = a_post(layer_id, astate, attn_a.contiguous(), apost)
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

        ``_step_impl`` mutates ``vstate.x`` and ``astate.payload.x_action``
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
            local_vstate.x = vx
            local_payload.x_action = ax
            self._step_impl(layer_id, local_vstate, local_astate, attn_mask=attn_mask, suppress_inner_attn_ckpt=True)
            return local_vstate.x, local_payload.x_action

        vx0 = vstate.x
        ax0 = outer_payload.x_action

        if offload:
            with torch.autograd.graph.save_on_cpu():
                new_vx, new_ax = torch.utils.checkpoint.checkpoint(_run, vx0, ax0, use_reentrant=False)
        else:
            new_vx, new_ax = torch.utils.checkpoint.checkpoint(_run, vx0, ax0, use_reentrant=False)

        vstate.x = new_vx
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
        # ``vstate.x.shape[1]`` is identical to ``f*tokens_per_frame`` for
        # backbones that carry a 3D ``(B, S, D)`` state (Wan), but for
        # backbones whose ``state.x`` is natively 5D ``(B, T, H, W, D)``
        # (Cosmos25) ``shape[1]`` is just ``T`` — wrong. Going through f and
        # the shared ``compute_video_tokens_per_frame`` helper is the only
        # formulation that works for both layouts.
        s_video = int(vstate.f) * self._video_tokens_per_frame(vstate)
        payload = astate.payload
        if payload is None or not hasattr(payload, "x_action"):
            raise RuntimeError(
                "MoTJointDriver: astate.payload must expose `x_action` (populated by ActionDiT.prepare_state)."
            )
        s_action = payload.x_action.shape[1]

        attn_mask = self._build_attention_mask(
            s_video=s_video,
            s_action=s_action,
            video_tokens_per_frame=self._video_tokens_per_frame(vstate),
            device=vstate.x.device,
        )

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


__all__ = ["MoTJointDriver"]
