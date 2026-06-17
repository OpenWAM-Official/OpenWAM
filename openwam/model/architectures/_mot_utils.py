"""Shared helpers for MoT (Mixture-of-Transformers) driver implementations.

Centralizes utilities used by both ``dual_system.mot_driver.MoTJointDriver``
(and its IDM subclass) and ``tri_system.mot_driver.TriSystemMoTDriver``, so the
two drivers don't drift on equivalent computations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openwam.model.video_backbone.base import BlockLoopState


def compute_video_tokens_per_frame(vstate: "BlockLoopState", driver_name: str) -> int:
    """Derive video tokens-per-frame from the spatial dims populated on ``vstate``.

    Used by every MoT driver to build the v↔v block of the joint attention mask.
    The video backbone's ``prepare()`` must populate ``grid_height`` and
    ``grid_width`` on the ``BlockLoopState``; if not, raise a clear error
    attributable to the calling driver via ``driver_name``.
    """
    h = int(getattr(vstate, "grid_height", 0))
    w = int(getattr(vstate, "grid_width", 0))
    if h <= 0 or w <= 0:
        raise ValueError(
            f"{driver_name}: cannot derive video_tokens_per_frame from vstate "
            f"(grid_height={h}, grid_width={w}). The video backbone's prepare() must populate them."
        )
    return h * w


__all__ = ["compute_video_tokens_per_frame"]
