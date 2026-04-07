"""Thin re-export of WanVideoPipeline from third_party.

WanVideoPipeline is too deeply coupled to diffsynth internals (~1615 LOC,
15+ model class dependencies) to extract into open_wam. This module
provides a single import point so the rest of open_wam never imports
from third_party directly — making the diffsynth boundary explicit
and easy to swap later.
"""

from third_party.diffsynth.pipelines.wan_video import WanVideoPipeline

__all__ = ["WanVideoPipeline"]
