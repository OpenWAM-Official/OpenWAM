"""V-JEPA 2.1 video encoder + its vendored ViT.

``encoder`` holds :class:`VJEPA21VideoEncoder`; ``loader`` builds the ViT from a
manifest; ``_vendor`` is the Apache-2.0 ViT code lifted from
facebookresearch/vjepa2 (branch ``vjepa2_1``) so this subsystem no longer needs
the ``third_party/vjepa2`` submodule. Re-exporting the encoder here triggers its
registry registration on ``encoder.vjepa2_1`` import.
"""

from openwam.model.video_backbone.encoder.vjepa2_1.encoder import VJEPA21VideoEncoder

__all__ = ["VJEPA21VideoEncoder"]
