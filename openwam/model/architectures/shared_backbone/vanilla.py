"""SharedBackbone Vanilla architecture.

Action tokens are concatenated to the video token sequence and ride
through every video DiT block as part of the shared sequence. Unlike
the MoE variant there are no expert FFN corrections — the raw shared
backbone learns the modality boundary itself. Output is taken from
the trailing action segment of the final hidden state via a small MLP.

The architecture is a thin composer: action processing logic lives in
``SharedVanillaActionBackbone``; ``BaseWAMArchitecture.forward`` drives
both backbones via the unified block-loop adapter interface.
"""

from openwam.model.action_backbone.shared_vanilla import SharedVanillaActionBackbone
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import register_architecture


@register_architecture(
    "shared_backbone_vanilla",
    status="supported",
    note="SharedBackbone vanilla: action tokens share the video DiT with no expert FFN.",
    framework="shared_backbone",
    variant="vanilla",  # Also serves as default for framework="shared_backbone" when variant is unset
)
class SharedBackboneVanillaArchitecture(BaseWAMArchitecture):
    """SharedBackbone vanilla: video DiT processes both video and action tokens."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            return
        action_dim = int(cfg.get("action_dim", 20))
        video_dim = self._resolve_video_dim(cfg)
        max_action_len = int(cfg.get("max_action_len", 512))
        self.action_backbone = SharedVanillaActionBackbone(
            action_dim=action_dim,
            video_dim=video_dim,
            max_action_len=max_action_len,
        )


__all__ = ["SharedBackboneVanillaArchitecture"]
