"""Locate real model weights for GPU tests.

Each asset is resolved in this order:

1. its ``OPENWAM_<ASSET>`` environment variable,
2. the legacy variable names some tests used before the names were unified,
3. the directory ``scripts/download_assets/`` writes to by default.

So after downloading with the repo scripts nothing needs to be exported; point
the variable elsewhere only for weights stored outside ``assets/``. Tests skip
(or are deselected via ``-m "not gpu"``) when the resolved path does not exist.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VIDEO_BACKBONES = PROJECT_ROOT / "assets" / "video_backbone_ckpt"

# canonical env var -> (legacy env vars, default location written by the downloader)
_ASSETS: dict[str, tuple[tuple[str, ...], Path]] = {
    "OPENWAM_WAN22_TI2V_5B": (("WAN22_TI2V_5B", "WAN_TI2V_5B_PATH"), _VIDEO_BACKBONES / "Wan2.2-TI2V-5B"),
    "OPENWAM_WAN21_VACE_1_3B": (("WAN21_VACE_1_3B",), _VIDEO_BACKBONES / "Wan2.1-VACE-1.3B"),
    "OPENWAM_COSMOS25_2B": (("COSMOS25_ASSET_PATH",), _VIDEO_BACKBONES / "Cosmos-Predict2.5-2B"),
    "OPENWAM_COSMOS_REASON1_7B": (("REASON1_ASSET_PATH",), _VIDEO_BACKBONES / "Cosmos-Reason1-7B"),
    "OPENWAM_COSMOS3_EDGE": (("COSMOS3_EDGE_ASSET_PATH",), _VIDEO_BACKBONES / "Cosmos3-Edge"),
}


def asset_path(name: str) -> str:
    """Return the on-disk location for ``name`` (a key of ``_ASSETS``)."""
    legacy, default = _ASSETS[name]
    for var in (name, *legacy):
        value = os.environ.get(var)
        if value:
            return value
    return str(default)
