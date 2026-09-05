"""DualSystem joint cross-attention architecture.

Bridge-collection mode: the video DiT runs to completion; hidden states
at the configured ``bridge_layers`` are captured along the way and feed
a separate ActionDiT via cross-attention. ``detach_bridge=True`` blocks
action gradients from flowing back into the video DiT.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import _cfg_get, register_architecture
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.compile_options import (
    compile_enabled,
    cross_attn_compile_cfg,
    section_enabled,
    torch_compile_kwargs,
)

logger = logging.getLogger(__name__)


def _cross_attn_options(cfg) -> dict:
    return {"detach_bridge": bool(_cfg_get(cfg, "detach_bridge", False))}


@register_architecture(
    "dual_system_cross_attn",
    status="supported",
    note="DualSystem joint cross-attention: bridge-collection plan with separate ActionDiT.",
    framework="dual_system",
    variant="joint_cross_attn",
    options_from_cfg=_cross_attn_options,
)
class DualSystemCrossAttnArchitecture(BaseWAMArchitecture):
    """DualSystem with bridge cross-attention.

    Action processing happens **after** the video DiT block loop completes:
    captured per-block bridge features feed the ActionDiT's cross-attention
    layers. The video DiT is not aware of the action stream.
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._detach_bridge: bool = False
        self._compiled_action_forward: Optional[Callable[..., Tensor]] = None
        if cfg is None:
            return
        if self.video_backbone is not None:
            # Auto-fill action-side geometry from the loaded video backbone,
            # mirroring ``joint_self_attn``. cross_attn does NOT require
            # num_heads / head_dim parity with the video backbone, but
            # defaulting to vb geometry keeps the two variants
            # apples-to-apples and removes the YAML coupling where every
            # backbone change had to be echoed in ``action_backbone``.
            # YAML/CLI overrides still win for ablations.
            cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
            cfg.setdefault("num_heads", self.video_backbone.num_heads)
            cfg.setdefault("attn_head_dim", self.video_backbone.head_dim)
        bl = resolve_bridge_layers(cfg)
        video_dim = self._resolve_video_dim(cfg)
        # Default the action-side raw context width to the loaded backbone's
        # text_dim (Wan T5-XXL=4096, Cosmos-Predict2.5=1024); explicit cfg/CLI
        # still wins. Falls back to 4096 when the backbone doesn't expose it.
        _vb_text_dim = getattr(self.video_backbone, "text_dim", None)
        text_dim = int(self._cfg_get(cfg, "text_dim", _vb_text_dim or 4096))
        self._init_proprio_context(cfg, text_dim=text_dim)
        self._detach_bridge = bool(cfg.get("detach_bridge", False))

        # Mirror joint_self_attn's heterogeneous-hidden support: when
        # ``attn_head_dim`` is supplied explicitly, the action residual hidden
        # dim (``dim``) is allowed to differ from ``num_heads * attn_head_dim``
        # — Q/K/V project across the gap. Falls back to ``dim // num_heads``
        # for back-compat with older same-width cross_attn configs.
        action_dim_hidden = int(cfg.get("dim", 768))
        # Hard-coded fallback (num_heads=12) is only hit when video_backbone is
        # None at __init__ AND cfg has no ``num_heads``. The setdefault block
        # above fills cfg from vb when vb is attached, and current mock-backbone
        # tests all pass num_heads explicitly, so this fallback is effectively
        # unreachable today; it stays as a last-resort default.
        num_heads = int(cfg.get("num_heads", 12))
        attn_head_dim = cfg.get("attn_head_dim")
        if attn_head_dim is not None:
            attn_head_dim = int(attn_head_dim)

        self.action_backbone = ActionDiT(
            action_dim=int(cfg.get("action_dim", 20)),
            dim=action_dim_hidden,
            ffn_dim=int(cfg.get("ffn_dim", 3072)),
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="joint_cross_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            shift_action=cfg.get("shift_action"),
        )

    @property
    def detach_bridge(self) -> bool:
        return self._detach_bridge

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Compile the tensor-only action cross-attention helper when requested."""

        super().apply_compile_optimizations(compile_cfg)
        self._compiled_action_forward = None
        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        section = cross_attn_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("cross-attn action compile disabled by config; running eager.")
            return
        if self.action_backbone is None:
            logger.warning("cross-attn action compile requested but action_backbone is None; running eager.")
            return

        kwargs = torch_compile_kwargs(section, default_mode="reduce-overhead")
        try:
            self._compiled_action_forward = torch.compile(self.action_backbone.forward_with_bridge_tuple, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive setup fallback
            self._compiled_action_forward = None
            logger.warning("cross-attn action torch.compile setup failed; running eager: %s", exc)
            return
        logger.info("Enabled cross-attn action compile with torch.compile kwargs=%s", kwargs)

    def _predict_actions_from_bridges(
        self,
        noisy_actions: Tensor,
        bridges: dict[int, Tensor],
        action_timestep: Tensor,
        *,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tensor:
        """Predict actions from collected bridge tensors, using compile when active."""

        ab = self.action_backbone
        if ab is None:
            raise RuntimeError("action_backbone is None — cannot predict cross-attn actions.")
        compiled_forward = getattr(self, "_compiled_action_forward", None)
        if compiled_forward is not None and not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
            bridge_tuple = ab.bridge_tuple_from_dict(bridges)
            try:
                torch.compiler.cudagraph_mark_step_begin()
                return compiled_forward(
                    noisy_actions,
                    bridge_tuple,
                    action_timestep,
                    context=context,
                    context_mask=context_mask,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                )
            except Exception as exc:
                self._compiled_action_forward = None
                logger.warning("cross-attn compiled action forward failed; falling back to eager: %s", exc)

        return ab(
            noisy_actions,
            bridges,
            action_timestep,
            context=context,
            context_mask=context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — pass pipe= to build_architecture or "
                "architecture.__init__ to enable forward()."
            )

        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        # Same 4D + clean-prefix-aligned t_mod opt-in as the joint_self_attn /
        # single_system / IDM forwards. TI2V fires its own branch first so
        # these kwargs are inert there. VACE and I2V do NOT emit
        # ``first_frame_latents`` (VACE routes via ``vace_context``, I2V via
        # the ``y`` channel), so ``zero_clean_prefix_t_mod`` is structurally
        # inert for them — kept on for symmetry with joint_self_attn so the
        # MoT driver gets a 4D ``t_mod``.
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        if noisy_actions is None or ab is None:
            for block_id in range(vb.num_layers):
                vstate = vb.run_block(block_id, vstate)
            return vb.finalize(vstate), None

        bridge_set = frozenset(ab.bridge_layers)
        bridges: dict[int, Tensor] = {}
        detach_bridge = self._detach_bridge

        for block_id in range(vb.num_layers):
            vstate = vb.run_block(block_id, vstate)
            if block_id in bridge_set:
                bridge = vstate.hidden_states
                if bridge.ndim == 5:
                    # Cosmos lays out hidden state as (B, T, H, W, D); flatten
                    # the spatial axes into a single token axis so the action
                    # backbone's cross-attn sees the Wan-compatible 3D shape.
                    B5, T5, H5, W5, D5 = bridge.shape
                    bridge = bridge.reshape(B5, T5 * H5 * W5, D5)
                bridges[block_id] = bridge.detach() if detach_bridge else bridge

        video_pred = vb.finalize(vstate)
        if not bridges:
            return video_pred, None

        action_pred = self._predict_actions_from_bridges(
            noisy_actions,
            bridges,
            action_timestep,
            context=action_context,
            context_mask=action_context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return video_pred, action_pred


__all__ = ["DualSystemCrossAttnArchitecture"]
