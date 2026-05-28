"""Tri-system MoT driver for Motus-style trimodal joint self-attention.

The driver owns no parameters. It coordinates the video Wan block split,
the tri-system action expert split, and the understanding expert split.
Each stream projects into Wan's shared per-head attention space; the
driver concatenates Q/K/V as ``[video, action, understanding]``, runs one
mixed self-attention, then routes each slice back through that stream's
post-attention/FFN suffix.
"""

from __future__ import annotations

import copy
import logging
from contextlib import nullcontext
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
    from openwam.model.vlm_backbone.qwen3_vl.und_expert import UnderstandingExpert, UnderstandingState


logger = logging.getLogger(__name__)

_VALID_ATTENTION_MASK_MODES = ("bidirectional", "joint")


class TriSystemMoTDriver:
    """Drive one Motus layer loop across video, action, and understanding streams.

    ``attention_mask_mode='joint'`` builds a trimodal mask with layout
    ``[video, action, understanding]`` where video queries cannot attend to
    action keys, action queries remain fully connected, and understanding
    queries attend only to understanding keys. ``bidirectional`` keeps the
    fully-connected trimodal attention.
    """

    def __init__(
        self,
        vb: "VideoBackbone",
        ab: "ActionBackbone",
        ub: "UnderstandingExpert",
        *,
        mot_checkpoint_mixed_attn: bool = True,
        attention_mask_mode: str = "joint",
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
        if attention_mask_mode not in _VALID_ATTENTION_MASK_MODES:
            raise ValueError(
                f"TriSystemMoTDriver: unknown attention_mask_mode '{attention_mask_mode}'. "
                f"Choose from: {_VALID_ATTENTION_MASK_MODES}."
            )

        self.vb = vb
        self.ab = ab
        self.ub = ub
        self.num_layers = vb.num_layers
        self.num_heads = vb.num_heads
        self.head_dim = vb.head_dim
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
        self.attention_mask_mode = attention_mask_mode

        if video_attention_mask_mode is not None:
            try:
                vb.video_attention_mask_mode = video_attention_mask_mode  # type: ignore[misc]
            except AttributeError:
                logger.warning(
                    "video_attention_mask_mode='%s' supplied to TriSystemMoTDriver but "
                    "%s does not expose a settable property; falling back to %s.",
                    video_attention_mask_mode,
                    type(vb).__name__,
                    vb.video_attention_mask_mode,
                )

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

    def _build_joint_mask(
        self,
        s_video: int,
        s_action: int,
        s_understanding: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        """Build the trimodal ``[Sv+Sa+Su, Sv+Sa+Su]`` bool attention mask.

        Layout (rows = queries, cols = keys; ``True`` means "attend to"):

        - ``v→v`` = ``vb.build_video_to_video_mask(...)``
        - ``v→a`` = False
        - ``v→u`` = True
        - action query rows are fully connected.
        - ``u→v`` / ``u→a`` = False and ``u→u`` = True.

        Assumes ``vstate.x`` has no reference-latent prefix prepended. Wan2.2-TI2V-5B
        never populates ``reference_latents`` (model config has no ``has_ref_conv``;
        first-frame conditioning uses the ``y`` VAE-embedding path instead). If a
        future Wan variant adds a reference prefix, ``first_frame_causal`` will
        misalign — the first ``tokens_per_frame`` tokens will be reference rather
        than the real first frame.
        """
        total = s_video + s_action + s_understanding
        mask = torch.ones((total, total), dtype=torch.bool, device=device)
        mask[:s_video, :s_video] = self.vb.build_video_to_video_mask(
            video_seq_len=s_video,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[:s_video, s_video : s_video + s_action] = False
        u_start = s_video + s_action
        mask[u_start:, :u_start] = False
        return mask

    def _apply_und_padding_mask(
        self,
        base_mask: Optional[Tensor],
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

        if base_mask is None:
            # bidirectional mode — start from all-True 2D base.
            base_2d = torch.ones((total, total), dtype=torch.bool, device=device)
        else:
            base_2d = base_mask

        mask = base_2d.unsqueeze(0).unsqueeze(0).expand(B, 1, total, total).contiguous()
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
    ) -> Optional[Tensor]:
        base: Optional[Tensor]
        if self.attention_mask_mode == "bidirectional":
            base = None
        elif self.attention_mask_mode == "joint":
            base = self._build_joint_mask(
                s_video=s_video,
                s_action=s_action,
                s_understanding=s_understanding,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        else:
            raise RuntimeError(f"unhandled attention_mask_mode '{self.attention_mask_mode}'")

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
        # passes it in for every layer. ``attn_mask=None`` is the SDPA "no mask"
        # signal — legitimate only when ``attention_mask_mode='bidirectional'`` and
        # no und padding is present. _step_impl does NOT rebuild the mask itself;
        # any direct caller must follow the same contract.
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

    def _step_impl_for_compile(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionState",
        ustate: "UnderstandingState",
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple["BlockLoopState", "ActionState", "UnderstandingState"]:
        """Compile-oriented per-layer body with tuple post-attention state."""
        v_pre = getattr(self.vb, "pre_attn_at_layer_for_compile", self.vb.pre_attn_at_layer)
        a_pre = getattr(self.ab, "pre_attn_at_layer_for_compile", self.ab.pre_attn_at_layer)
        u_pre = getattr(self.ub, "pre_attn_at_layer_for_compile", self.ub.pre_attn_at_layer)
        v_post = getattr(self.vb, "post_attn_at_layer_for_compile", self.vb.post_attn_at_layer)
        a_post = getattr(self.ab, "post_attn_at_layer_for_compile", self.ab.post_attn_at_layer)
        u_post = getattr(self.ub, "post_attn_at_layer_for_compile", self.ub.post_attn_at_layer)

        q_v, k_v, v_v, vpost = v_pre(layer_id, vstate)
        q_a, k_a, v_a, apost = a_pre(layer_id, astate)
        q_u, k_u, v_u, upost = u_pre(layer_id, ustate)

        self._check_compatible(layer_id, q_v, k_v, v_v, q_a, k_a, v_a, q_u, k_u, v_u)

        s_video = q_v.shape[1]
        s_action = q_a.shape[1]
        s_understanding = q_u.shape[1]

        q_cat = torch.cat([q_v, q_a, q_u], dim=1)
        k_cat = torch.cat([k_v, k_a, k_u], dim=1)
        v_cat = torch.cat([v_v, v_a, v_u], dim=1)

        mixed = self._mixed_attention(q_cat, k_cat, v_cat, attn_mask)
        attn_v, attn_a, attn_u = mixed.split([s_video, s_action, s_understanding], dim=1)
        vstate = v_post(layer_id, vstate, attn_v.contiguous(), vpost)
        astate = a_post(layer_id, astate, attn_a.contiguous(), apost)
        ustate = u_post(layer_id, ustate, attn_u.contiguous(), upost)
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
            local_vstate.x = vx
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
            return local_vstate.x, self._get_action_tokens(local_astate), local_ustate.und_tokens

        vx0 = vstate.x
        ax0 = self._get_action_tokens(astate)
        ux0 = ustate.und_tokens

        cm = torch.autograd.graph.save_on_cpu() if offload else nullcontext()
        with cm:
            new_vx, new_ax, new_ux = torch.utils.checkpoint.checkpoint(_run, vx0, ax0, ux0, use_reentrant=False)

        vstate.x = new_vx
        self._set_action_tokens(astate, new_ax)
        ustate.und_tokens = new_ux
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
        # ``vstate.x.shape[1]`` is identical to ``f*tokens_per_frame`` for
        # backbones that carry a 3D ``(B, S, D)`` state (Wan), but for
        # backbones whose ``state.x`` is natively 5D ``(B, T, H, W, D)``
        # (Cosmos25) ``shape[1]`` is just ``T`` — wrong. Going through f and
        # the shared ``compute_video_tokens_per_frame`` helper is the only
        # formulation that works for both layouts.
        s_video = int(vstate.f) * self._video_tokens_per_frame(vstate)
        s_action = self._get_action_tokens(astate).shape[1]
        s_understanding = ustate.und_tokens.shape[1]
        attn_mask = self._build_attention_mask(
            s_video=s_video,
            s_action=s_action,
            s_understanding=s_understanding,
            video_tokens_per_frame=self._video_tokens_per_frame(vstate),
            device=vstate.x.device,
            und_mask=getattr(ustate, "und_mask", None),
        )

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
