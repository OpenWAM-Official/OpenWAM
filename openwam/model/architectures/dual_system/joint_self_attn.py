"""DualSystem joint self-attention architecture.

Interleaved mode: ActionDiT blocks fire **inside** the video DiT block
loop. At each configured bridge layer, the video hidden state is
projected down to action dim, accumulated into a running video stream,
the joint ActionDiT block runs over (action_tokens, video_proj), and
the updated video stream is back-projected and added as a residual to
the video hidden state.

The architecture is a thin composer: it instantiates an ActionDiT (which
is itself an ActionBackbone subclass) configured for the joint_self_attn
variant and the unified forward in ``BaseWAMArchitecture`` drives it.
"""

from openwam.model.action_backbone.action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.utils import resolve_bridge_layers


@register_architecture(
    "dual_system_self_attn",
    status="supported",
    note="DualSystem joint self-attention: interleaved ActionDiT blocks inside the video DiT loop.",
    framework="dual_system",
    variant="joint_self_attn",
)
class DualSystemSelfAttnArchitecture(BaseWAMArchitecture):
    """DualSystem with joint self-attention (MMDiT-style dual-stream)."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            return
        if self.video_backbone is not None:
            cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
        bl = resolve_bridge_layers(cfg)
        video_dim = self._resolve_video_dim(cfg)
        self.action_backbone = ActionDiT(
            action_dim=int(cfg.get("action_dim", 20)),
            dim=int(cfg.get("dim", 768)),
            ffn_dim=int(cfg.get("ffn_dim", 3072)),
            num_heads=int(cfg.get("num_heads", 12)),
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="joint_self_attn",
            detach_bridge=bool(cfg.get("detach_bridge", False)),
            use_proprioception=bool(cfg.get("use_proprioception", False)),
            state_dim=int(cfg.get("state_dim") or 0),
            proprio_fusion=cfg.get("proprio_fusion", "channel_concat"),
            num_state_tokens=int(cfg.get("num_state_tokens", 4)),
        )

    @property
    def detach_bridge(self) -> bool:
        return bool(self.action_backbone.detach_bridge) if self.action_backbone is not None else False


__all__ = ["DualSystemSelfAttnArchitecture"]
