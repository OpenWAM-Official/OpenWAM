"""Video backbone module for OpenWAM.

Provides the Wan video generation pipeline and utilities for extracting
video features used by the action model.

The underlying implementation lives in the diffsynth subpackage
(derived from DiffSynth-Studio, Apache 2.0 license).
"""

from openwam.model.video_backbone.diffsynth.pipelines.wan_video import WanVideoPipeline

__all__ = ["WanVideoPipeline"]
