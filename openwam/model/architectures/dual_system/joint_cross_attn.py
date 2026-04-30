"""DualSystem joint cross-attention architecture.

Bridge-collection mode: video DiT runs first, hidden states at the
configured ``bridge_layers`` are captured, then a separate ActionDiT
processes the action stream conditioned on those features via cross
attention. ``detach_bridge=True`` blocks action gradients from flowing
back into the video DiT.

The architecture is a thin composer: it instantiates an ActionDiT (which
is itself an ActionBackbone subclass) and the unified forward in
``BaseWAMArchitecture`` drives it via the standard block-loop adapter.
"""

from openwam.model.action_backbone.action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.utils import resolve_bridge_layers


def _cross_attn_options(cfg) -> dict:
    return {"detach_bridge": bool(_cfg_get(cfg, "detach_bridge", False))}


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


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
            variant="joint_cross_attn",
            detach_bridge=bool(cfg.get("detach_bridge", False)),
            use_proprioception=bool(cfg.get("use_proprioception", False)),
            state_dim=int(cfg.get("state_dim") or 0),
            proprio_fusion=cfg.get("proprio_fusion", "channel_concat"),
            num_state_tokens=int(cfg.get("num_state_tokens", 4)),
        )

    @property
    def detach_bridge(self) -> bool:
        return bool(self.action_backbone.detach_bridge) if self.action_backbone is not None else False


__all__ = ["DualSystemCrossAttnArchitecture"]
