"""Wan backbone variant strategy.

Each Wan family member (I2V / TI2V / VACE / plain) differs only in how the
first-frame / control signal is turned into conditioning. Those differences are
encapsulated in :class:`WanVariant` subclasses so ``WanBase`` carries
no ``if has_image_input / _is_ti2v / _has_vace`` branches in its hot paths —
the variant is resolved once at construction via :func:`detect` and every
variant-specific decision is delegated to ``self._variant``.

Variants are stateless strategy objects; their methods take the backbone
``bb`` and call its component helpers (``_build_vace_pixel_inputs`` /
``_build_i2v_y`` / ...).
"""

from __future__ import annotations

from openwam.model.video_backbone.wan.variants.base import WanVariant
from openwam.model.video_backbone.wan.variants.i2v import I2VVariant
from openwam.model.video_backbone.wan.variants.ti2v import TI2VVariant
from openwam.model.video_backbone.wan.variants.vace import VACEVariant


def detect(dit, vace) -> WanVariant:
    """Resolve the Wan variant from loaded components (not config names).

    Order matters — the three families are mutually exclusive:
      - VACE: ``vace`` module present (wan21_vace_1_3b).
      - TI2V: DiT fuses the VAE embedding + separated timestep (wan22_ti2v_5b).
      - I2V:  DiT takes an image input via clip_feature + y (wan21_i2v_14b_480p).
      - plain: none of the above (no first-frame conditioning).
    """
    if vace is not None:
        return VACEVariant()
    if bool(getattr(dit, "fuse_vae_embedding_in_latents", False)):
        return TI2VVariant()
    if bool(getattr(dit, "has_image_input", False)):
        return I2VVariant()
    return WanVariant()


__all__ = ["WanVariant", "I2VVariant", "TI2VVariant", "VACEVariant", "detect"]
