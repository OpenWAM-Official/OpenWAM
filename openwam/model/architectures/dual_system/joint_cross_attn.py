"""DualSystem joint cross-attention architecture.

Bridge-collection mode: the video DiT runs to completion; hidden states
at the configured ``bridge_layers`` are captured along the way and feed
a separate ActionDiT via cross-attention. ``detach_bridge=True`` blocks
action gradients from flowing back into the video DiT.
"""

from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor

from openwam.model.action_backbone.dualsystem_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import register_architecture
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
        if cfg is None:
            return
        if self.video_backbone is not None:
            cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
        bl = resolve_bridge_layers(cfg)
        video_dim = self._resolve_video_dim(cfg)
        self._detach_bridge = bool(cfg.get("detach_bridge", False))

        # Mirror joint_self_attn's heterogeneous-hidden support: when
        # ``attn_head_dim`` is supplied explicitly, the action residual hidden
        # dim (``dim``) is allowed to differ from ``num_heads * attn_head_dim``
        # — Q/K/V project across the gap. Falls back to ``dim // num_heads``
        # for back-compat with older same-width cross_attn configs.
        action_dim_hidden = int(cfg.get("dim", 768))
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
            use_proprioception=bool(cfg.get("use_proprioception", False)),
            state_dim=int(cfg.get("state_dim") or 0),
            proprio_fusion=cfg.get("proprio_fusion", "channel_concat"),
            num_state_tokens=int(cfg.get("num_state_tokens", 4)),
        )

    @property
    def detach_bridge(self) -> bool:
        return self._detach_bridge

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
                bridges[block_id] = vstate.x.detach() if detach_bridge else vstate.x

        video_pred = vb.finalize(vstate)
        if not bridges:
            return video_pred, None

        action_pred = ab(
            noisy_actions,
            bridges,
            action_timestep,
            proprio_state=proprio_state,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return video_pred, action_pred


__all__ = ["DualSystemCrossAttnArchitecture"]
