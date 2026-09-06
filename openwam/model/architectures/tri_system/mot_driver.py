"""TriSystem MoT (Mixture-of-Transformers) driver: per-layer mixed attention
across video, action, and understanding (Motus-style) streams.

Each backbone runs the prefix of one DiT block (``pre_attn_at_layer``) to yield
Q/K/V; the driver concatenates the three modalities along the sequence
dimension, runs a single mixed self-attention with a trimodal joint mask, splits
the result, and feeds each slice through that modality's block suffix
(``post_attn_at_layer``). The driver owns no parameters — a plain Python class,
absent from ``state_dict``.
"""

# Source: https://github.com/thu-ml/Motus.
# Upstream revision: UNKNOWN (the original internal import did not record a commit SHA).
# License: Apache-2.0; see the repository-level LICENSE (Apache-2.0).
# Modified by OpenWAM contributors: the Motus MoT layer loop was adapted to
# OpenWAM backbones, masks, padding, checkpointing, and state containers.

from __future__ import annotations

import copy
from contextlib import nullcontext
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
    from openwam.model.architectures.tri_system.und_expert import UnderstandingExpert, UnderstandingState
    from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone


class TriSystemMoTDriver:
    """Drive one Motus layer loop across video, action, and understanding streams.

    The trimodal ``[video, action, understanding]`` mask is built once per
    :meth:`run_joint_loop`. ``v↔v`` follows ``vb.video_attention_mask_mode``;
    ``a↔a`` is fully connected; understanding is a read-only tail (everyone
    attends to it, it attends only to itself); the v↔a coupling follows
    ``attention_mask_mode`` — see :mod:`utils.mask_modes` for the four modes
    (default ``action_sees_video``: video does not see action, action sees all video).
    """

    def __init__(
        self,
        vb: "VideoBackbone",
        ab: "ActionDiTBackbone",
        ub: "UnderstandingExpert",
        *,
        mot_checkpoint_mixed_attn: bool = True,
        attention_mask_mode: str = ACTION_SEES_VIDEO,
        video_attention_mask_mode: Optional[str] = None,
    ) -> None:
        if vb.num_layers != ab.num_layers:
            raise ValueError(
                f"TriSystemMoTDriver: video num_layers ({vb.num_layers}) must equal "
                f"action num_layers ({ab.num_layers})."
            )
        if vb.num_layers != ub.num_layers:
            raise ValueError(
                f"TriSystemMoTDriver: video num_layers ({vb.num_layers}) must equal "
                f"understanding num_layers ({ub.num_layers})."
            )
        if vb.num_heads != ab.num_heads or vb.num_heads != ub.num_heads:
            raise ValueError(
                "TriSystemMoTDriver: video/action/understanding num_heads must match "
                f"(video={vb.num_heads}, action={ab.num_heads}, understanding={ub.num_heads})."
            )
        if vb.head_dim != ab.head_dim or vb.head_dim != ub.head_dim:
            raise ValueError(
                "TriSystemMoTDriver: video/action/understanding head_dim must match "
                f"(video={vb.head_dim}, action={ab.head_dim}, understanding={ub.head_dim})."
            )
        validate_attention_mask_mode(attention_mask_mode)

        self.vb = vb
        self.ab = ab
        self.ub = ub
        self.num_layers = vb.num_layers
        self.num_heads = vb.num_heads
        self.head_dim = vb.head_dim
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        self.attention_mask_mode = attention_mask_mode

        set_video_attention_mask_mode(vb, video_attention_mask_mode)

    @staticmethod
    def _get_action_tokens(astate: "ActionState") -> Tensor:
        payload = astate.payload
        if hasattr(payload, "x_action"):
            return payload.x_action
        if hasattr(payload, "action_tokens"):
            return payload.action_tokens
        raise RuntimeError(
            "TriSystemMoTDriver: action payload must expose `x_action` "
            "(shared ActionDiT) or `action_tokens` (Motus-style tri action expert)."
        )

    @staticmethod
    def _set_action_tokens(astate: "ActionState", value: Tensor) -> None:
        payload = astate.payload
        if hasattr(payload, "x_action"):
            payload.x_action = value
            return
        if hasattr(payload, "action_tokens"):
            payload.action_tokens = value
            return
        raise RuntimeError(
            "TriSystemMoTDriver: action payload must expose `x_action` "
            "or `action_tokens` before checkpointed execution."
        )

    def _apply_und_padding_mask(
        self,
        base_mask: Tensor,
        und_mask: Tensor,
        s_video: int,
        s_action: int,
        s_understanding: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        """Apply per-batch und padding to the joint mask.

        ``und_mask`` is ``[B, Su]`` bool (True = valid VLM token). Returns
        ``[B, 1, S, S]`` bool mask suitable for ``F.scaled_dot_product_attention``.

        Symmetric blocking: padded und positions are masked out both as KEY columns
        (so v/a/u queries cannot read them) and as QUERY rows (so they do not consume
        FLOPs). To avoid the SDPA NaN-row behavior when a query row is fully masked,
        each padded und QUERY row keeps a self-attending diagonal element to itself.
        These rows do not contribute to any supervised loss (only video + action are
        supervised) so the self-attention output is discarded.
        """
        if und_mask.ndim != 2 or und_mask.shape[1] != s_understanding:
            raise ValueError(f"und_mask must be [B, Su={s_understanding}] bool, got shape {tuple(und_mask.shape)}")
        B = und_mask.shape[0]
        total = s_video + s_action + s_understanding
        u_start = s_video + s_action

        mask = base_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, total, total).contiguous()
        # Caller (_build_attention_mask) guards via `und_mask.all()` — when we reach here
        # there is guaranteed to be at least one padded position, so no need to short-circuit.
        pad = ~und_mask.to(device=device, dtype=torch.bool)  # [B, Su]

        # KEY column block: any query → padded und key = False.
        mask[..., u_start:].masked_fill_(pad[:, None, None, :], False)
        # QUERY row block: padded und query → everything = False.
        mask[:, :, u_start:, :].masked_fill_(pad[:, None, :, None], False)
        # Restore a self-attending diagonal on padded und rows so SDPA does not
        # produce NaN from all-False rows. The resulting output is unused by loss.
        diag_idx = torch.arange(s_understanding, device=device)
        # mask[b, 0, u_start + i, u_start + i] = True where pad[b, i]
        mask[:, 0, u_start + diag_idx, u_start + diag_idx] = mask[:, 0, u_start + diag_idx, u_start + diag_idx] | pad
        return mask

    def _build_attention_mask(
        self,
        s_video: int,
        s_action: int,
        s_understanding: int,
        video_tokens_per_frame: int,
        *,
        device: torch.device,
        und_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Build the trimodal mask: video + action + understanding read-only tail.

        Assumes no reference-latent prefix on ``vstate.hidden_states``. Wan2.2-TI2V-5B
        never populates ``reference_latents``; if a future variant adds one,
        ``first_frame_causal`` would misalign (first tokens become reference).
        """
        base = build_cross_modal_attention_mask(
            self.vb,
            s_video=s_video,
            s_action=s_action,
            video_tokens_per_frame=video_tokens_per_frame,
            mode=self.attention_mask_mode,
            device=device,
            n_readonly_tail=s_understanding,
        )
        if und_mask is None or bool(und_mask.all()):
            return base
        return self._apply_und_padding_mask(
            base,
            und_mask,
            s_video=s_video,
            s_action=s_action,
            s_understanding=s_understanding,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

    def _mixed_attention(
        self,
        q_cat: Tensor,
        k_cat: Tensor,
        v_cat: Tensor,
        attn_mask: Optional[Tensor],
    ) -> Tensor:
        """Wan-compatible mixed self-attention over the concatenated trimodal sequence."""
        n = self.num_heads
        q = rearrange(q_cat, "b s (n d) -> b n s d", n=n)
        k = rearrange(k_cat, "b s (n d) -> b n s d", n=n)
        v = rearrange(v_cat, "b s (n d) -> b n s d", n=n)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return rearrange(out, "b n s d -> b s (n d)", n=n)

    def _check_compatible(self, layer_id: int, *tensors: Tensor) -> None:
        first = tensors[0]
        for tensor in tensors[1:]:
            if tensor.dtype != first.dtype:
                raise RuntimeError(
                    f"TriSystemMoTDriver: dtype mismatch at layer {layer_id} ({first.dtype} vs {tensor.dtype})."
                )
            if tensor.device != first.device:
                raise RuntimeError(
                    f"TriSystemMoTDriver: device mismatch at layer {layer_id} ({first.device} vs {tensor.device})."
                )

    def _video_tokens_per_frame(self, vstate: "BlockLoopState") -> int:
        return compute_video_tokens_per_frame(vstate, "TriSystemMoTDriver")

    def step(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionState",
        ustate: "UnderstandingState",
        attn_mask: Optional[Tensor] = None,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple["BlockLoopState", "ActionState", "UnderstandingState"]:
        if use_gradient_checkpointing and self.ab.training:
            return self._step_checkpointed(
                layer_id,
                vstate,
                astate,
                ustate,
                attn_mask=attn_mask,
                offload=use_gradient_checkpointing_offload,
            )
        return self._step_impl(layer_id, vstate, astate, ustate, attn_mask=attn_mask)

    def _step_impl(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionState",
        ustate: "UnderstandingState",
        attn_mask: Optional[Tensor] = None,
        *,
        suppress_inner_attn_ckpt: bool = False,
    ) -> Tuple["BlockLoopState", "ActionState", "UnderstandingState"]:
        q_v, k_v, v_v, vpost = self.vb.pre_attn_at_layer(layer_id, vstate)
        q_a, k_a, v_a, apost = self.ab.pre_attn_at_layer(layer_id, astate)
        q_u, k_u, v_u, upost = self.ub.pre_attn_at_layer(layer_id, ustate)

        self._check_compatible(layer_id, q_v, k_v, v_v, q_a, k_a, v_a, q_u, k_u, v_u)

        s_video = q_v.shape[1]
        s_action = q_a.shape[1]
        s_understanding = q_u.shape[1]

        q_cat = torch.cat([q_v, q_a, q_u], dim=1)
        k_cat = torch.cat([k_v, k_a, k_u], dim=1)
        v_cat = torch.cat([v_v, v_a, v_u], dim=1)
        # Contract: `run_joint_loop` pre-builds ``attn_mask`` once per forward and
        # passes it in for every layer. _step_impl does NOT rebuild the mask itself;
        # any direct caller must pass the same pre-built mask.
        if self.mot_checkpoint_mixed_attn and self.ab.training and not suppress_inner_attn_ckpt:
            mixed = torch.utils.checkpoint.checkpoint(
                self._mixed_attention, q_cat, k_cat, v_cat, attn_mask, use_reentrant=False
            )
        else:
            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attn_mask)

        attn_v, attn_a, attn_u = mixed.split([s_video, s_action, s_understanding], dim=1)
        vstate = self.vb.post_attn_at_layer(layer_id, vstate, attn_v.contiguous(), vpost)
        astate = self.ab.post_attn_at_layer(layer_id, astate, attn_a.contiguous(), apost)
        ustate = self.ub.post_attn_at_layer(layer_id, ustate, attn_u.contiguous(), upost)
        return vstate, astate, ustate

    def _step_checkpointed(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionState",
        ustate: "UnderstandingState",
        *,
        attn_mask: Optional[Tensor],
        offload: bool,
    ) -> Tuple["BlockLoopState", "ActionState", "UnderstandingState"]:
        outer_payload = astate.payload

        def _run(vx: Tensor, ax: Tensor, ux: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            local_vstate = copy.copy(vstate)
            local_astate = copy.copy(astate)
            local_ustate = copy.copy(ustate)
            local_payload = copy.copy(outer_payload)
            local_astate.payload = local_payload
            local_vstate.hidden_states = vx
            self._set_action_tokens(local_astate, ax)
            local_ustate.und_tokens = ux
            self._step_impl(
                layer_id,
                local_vstate,
                local_astate,
                local_ustate,
                attn_mask=attn_mask,
                suppress_inner_attn_ckpt=True,
            )
            return local_vstate.hidden_states, self._get_action_tokens(local_astate), local_ustate.und_tokens

        vx0 = vstate.hidden_states
        ax0 = self._get_action_tokens(astate)
        ux0 = ustate.und_tokens

        cm = torch.autograd.graph.save_on_cpu() if offload else nullcontext()
        with cm:
            new_vx, new_ax, new_ux = torch.utils.checkpoint.checkpoint(_run, vx0, ax0, ux0, use_reentrant=False)

        vstate.hidden_states = new_vx
        self._set_action_tokens(astate, new_ax)
        ustate.und_tokens = new_ux
        return vstate, astate, ustate

    def run_joint_loop_for_compile(
        self,
        vstate: "BlockLoopState",
        astate: "ActionState",
        ustate: "UnderstandingState",
        *,
        attn_mask: Optional[Tensor],
    ) -> Tuple["BlockLoopState", "ActionState", "UnderstandingState"]:
        """Compile-friendly trimodal loop with a prebuilt attention mask.

        This eval-only path deliberately avoids the Python ``step(layer_id, ...)``
        wrapper and the tensor-to-Python padding-mask branch in
        ``_build_attention_mask``. The architecture builds the mask outside the
        compiled boundary, then this method keeps the hot path on per-layer
        tensor/tuple pre/post helpers.
        """
        # Prefix K/V columns (e.g. Cosmos3's cached und text stream) are
        # represented by a rectangular key axis.  Widen the trimodal mask once
        # before entering the compiled loop; the per-layer Q/K/V concat then
        # remains unchanged.
        if attn_mask is not None:
            attn_mask = widen_mask_for_prefix_kv(attn_mask, vstate)
        pre_v = getattr(self.vb, "pre_attn_at_layer_for_compile", self.vb.pre_attn_at_layer)
        post_v = getattr(self.vb, "post_attn_at_layer_for_compile", self.vb.post_attn_at_layer)
        for layer_id in range(self.num_layers):
            q_v, k_v, v_v, vpost = pre_v(layer_id, vstate)
            q_a, k_a, v_a, apost = self.ab.pre_attn_at_layer_for_compile(layer_id, astate)
            q_u, k_u, v_u, upost = self.ub.pre_attn_at_layer_for_compile(layer_id, ustate)

            self._check_compatible(layer_id, q_v, k_v, v_v, q_a, k_a, v_a, q_u, k_u, v_u)

            s_video = q_v.shape[1]
            s_action = q_a.shape[1]
            s_understanding = q_u.shape[1]

            q_cat = torch.cat([q_v, q_a, q_u], dim=1)
            k_cat = torch.cat([k_v, k_a, k_u], dim=1)
            v_cat = torch.cat([v_v, v_a, v_u], dim=1)
            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attn_mask)

            attn_v, attn_a, attn_u = mixed.split([s_video, s_action, s_understanding], dim=1)
            vstate = post_v(layer_id, vstate, attn_v.contiguous(), vpost)
            astate = self.ab.post_attn_at_layer_for_compile(layer_id, astate, attn_a.contiguous(), apost)
            ustate = self.ub.post_attn_at_layer_for_compile(layer_id, ustate, attn_u.contiguous(), upost)
        return vstate, astate, ustate

    def run_joint_loop(
        self,
        vstate: "BlockLoopState",
        astate: "ActionState",
        ustate: "UnderstandingState",
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple["BlockLoopState", "ActionState", "UnderstandingState"]:
        # Resolve sequence shapes from the backbone-populated f/h/w fields.
        # ``vstate.hidden_states.shape[1]`` is identical to ``f*tokens_per_frame`` for
        # backbones that carry a 3D ``(B, S, D)`` state (Wan), but for
        # backbones whose ``state.hidden_states`` is natively 5D ``(B, T, H, W, D)``
        # (CosmosPredict25) ``shape[1]`` is just ``T`` — wrong. Going through f and
        # the shared ``compute_video_tokens_per_frame`` helper is the only
        # formulation that works for both layouts.
        s_video = int(vstate.grid_frames) * self._video_tokens_per_frame(vstate)
        s_action = self._get_action_tokens(astate).shape[1]
        s_understanding = ustate.und_tokens.shape[1]
        attn_mask = self._build_attention_mask(
            s_video=s_video,
            s_action=s_action,
            s_understanding=s_understanding,
            video_tokens_per_frame=self._video_tokens_per_frame(vstate),
            device=vstate.hidden_states.device,
            und_mask=getattr(ustate, "und_mask", None),
        )
        # Cosmos3 prepends cached understanding K/V to the video keys.  The
        # shared mask describes query×query streams, so prepend those key-only
        # columns before every layer consumes it.
        attn_mask = widen_mask_for_prefix_kv(attn_mask, vstate)

        last_layer = self.num_layers - 1
        for layer_id in range(self.num_layers):
            use_ckpt = use_gradient_checkpointing and layer_id != last_layer
            vstate, astate, ustate = self.step(
                layer_id,
                vstate,
                astate,
                ustate,
                attn_mask=attn_mask,
                use_gradient_checkpointing=use_ckpt,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        return vstate, astate, ustate


__all__ = ["TriSystemMoTDriver"]
