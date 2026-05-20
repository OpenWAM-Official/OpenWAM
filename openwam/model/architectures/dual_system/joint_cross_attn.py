"""DualSystem joint cross-attention architecture.

Bridge-collection mode: the video DiT runs to completion; hidden states
at the configured ``bridge_layers`` are captured along the way and feed
a separate ActionDiT via cross-attention. ``detach_bridge=True`` blocks
action gradients from flowing back into the video DiT.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.joint_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.cross_attn_compile import CompiledCrossAttnAction
from openwam.model.architectures.registry import register_architecture
from openwam.model.compile_options import compile_mode, cross_attn_compile_cfg, section_enabled
from openwam.model.registry import _cfg_get
from openwam.utils import resolve_bridge_layers


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
        self._compiled_cross_attn_action: CompiledCrossAttnAction | None = None
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
        text_dim = self._resolve_text_dim(cfg)
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
        )

    @property
    def detach_bridge(self) -> bool:
        return self._detach_bridge

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Apply the cross-attention compile mode through an action-side helper."""

        mode = compile_mode(compile_cfg, default="none", strict=True)
        if mode != "auto":
            super().apply_compile_optimizations(compile_cfg)
            self._compiled_cross_attn_action = None
            return

        cross_attn_cfg = cross_attn_compile_cfg(compile_cfg)
        if section_enabled(cross_attn_cfg, default=False) and self.action_backbone is not None:
            self._compiled_cross_attn_action = CompiledCrossAttnAction(self.action_backbone, cross_attn_cfg)
        else:
            self._compiled_cross_attn_action = None

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio_state: Optional[Tensor] = None,
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

        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio_state)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        # Same 4D + clean-prefix-aligned t_mod opt-in as the joint_self_attn /
        # shared_backbone / IDM forwards. TI2V fires its own branch first so
        # these kwargs are inert there. For VACE the broadcast path now zeros
        # the first frame's t_mod to match the latent-side clean-ref
        # replacement, removing an existing data/t_mod mismatch. I2V has no
        # ``first_frame_latents`` so ``zero_clean_prefix_t_mod`` is a no-op.
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

        bridge_set = ab.bridge_layers_set
        bridges: dict[int, Tensor] = {}
        detach_bridge = self._detach_bridge

        for block_id in range(vb.num_layers):
            vstate = vb.run_block(block_id, vstate)
            if block_id in bridge_set:
                bridge = vstate.x
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

        compiled_action = self._compiled_cross_attn_action
        if compiled_action is not None and compiled_action.can_run(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        ):
            action_pred = compiled_action.run(
                noisy_actions,
                bridges,
                action_timestep,
                context=action_context,
                context_mask=action_context_mask,
            )
        else:
            action_pred = ab(
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
