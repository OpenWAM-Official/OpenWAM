"""Typed input bag for ``VideoBackbone.prepare_inputs_for_inference``.

Replaces the conditional ``prep_kwargs`` dict-building previously sitting
in ``BaseWAMArchitecture.generate`` (and a parallel copy in
``dual_system/idm.py``) for forwarding CFG / cache knobs to the backbone.
The dict pattern was load-bearing for one specific quirk: Wan's adapter
has a narrow signature with no ``**kwargs`` catch-all, so threading
``cfg_scale=2.0`` through it would have ``TypeError``'d at runtime. The
work-around was to only set the keys when they were non-default.

A typed dataclass makes the contract explicit:

- Every field has an unambiguous default that matches the no-op
  inference path (``cfg_scale=1.0``, no caches, no CFG).
- Adapters declare which fields they accept by reading them from the
  ``InferenceInputs`` argument; unknown fields are simply unused
  (vs. ``TypeError``).
- Callers build one object instead of conditionally splicing a dict —
  no more "did I forget to set cfg_merge when cfg_scale > 1?" footguns.

The dataclass is ``frozen`` so adapters cannot accidentally mutate the
caller's request mid-pipeline; if an adapter needs to derive new fields
(e.g. ``T_lat`` from ``num_frames``), it writes into the returned
``inputs_shared`` dict, not back into the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from torch import Tensor


@dataclass(frozen=True)
class InferenceInputs:
    """All knobs ``prepare_inputs_for_inference`` may consume.

    Mirrors the kwarg list previously threaded through
    ``BaseWAMArchitecture.generate -> vb.prepare_inputs_for_inference``.
    Backbones consume what they understand and ignore the rest; the
    ``cfg_scale`` validator stays at the architecture layer (see
    ``BaseWAMArchitecture.generate``) so misconfiguration fails before
    the backbone is touched.
    """

    prompt: str
    vace_video: Any = None
    first_frame_image: Any = None
    pre_encoded_text: Optional[Tensor] = None
    uncond_pre_encoded_text: Optional[Tensor] = None
    num_frames: int = 49
    height: int = 480
    width: int = 832
    seed: int = 42
    num_inference_steps: int = 50
    shift: float = 5.0
    tiled: bool = True
    tile_size: Optional[tuple] = None
    tile_stride: Optional[tuple] = None
    vace_cache: Optional[dict] = None
    prompt_embed_cache: Optional[dict] = None
    cfg_scale: float = 1.0
    cfg_merge: bool = False


__all__ = ["InferenceInputs"]
