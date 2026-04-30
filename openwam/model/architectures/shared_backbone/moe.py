"""SharedBackbone MoE architecture.

Action tokens are concatenated to the video token sequence and ride
through the shared video DiT blocks (same Q/K/V projections, same
self-attention). At configured ``expert_layers`` the action tokens
receive an extra expert FFN correction for modality-specific capacity.

The architecture is a thin composer: action processing logic lives in
``MoEExpertDiT`` (an ActionBackbone subclass); ``BaseWAMArchitecture.forward``
drives both backbones via the unified block-loop adapter interface.
"""

from openwam.model.action_backbone.moe_dit import MoEExpertDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.utils import resolve_bridge_layers


@register_architecture(
    "shared_backbone_moe",
    status="supported",
    note="SharedBackbone MoE: action tokens share the video DiT with expert FFN at configured layers.",
    framework="shared_backbone",
    variant="moe",
)
class SharedBackboneMoEArchitecture(BaseWAMArchitecture):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            return
        vb = self.video_backbone
        video_dim = self._resolve_video_dim(cfg)
        num_layers = vb.num_layers if vb is not None else int(cfg.get("num_dit_layers", 30))
        bl = resolve_bridge_layers(cfg, num_layers=num_layers)

        self.action_backbone = MoEExpertDiT(
            action_dim=int(cfg.get("action_dim", 20)),
            video_dim=video_dim,
            expert_ffn_dim=int(cfg.get("expert_ffn_dim", 14336)),
            num_experts=int(cfg.get("num_experts", len(bl))),
            expert_layers=bl,
        )


__all__ = ["SharedBackboneMoEArchitecture"]
