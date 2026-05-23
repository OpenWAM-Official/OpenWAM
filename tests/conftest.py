"""Shared test fixtures and sys.path setup."""

import sys
from pathlib import Path

# Ensure project root and third-party packages are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
# SANA upstream uses top-level ``from diffusion.model...`` imports — its
# ``diffusion/`` package lives at ``third_party/Sana/diffusion/``, so the
# parent ``third_party/Sana`` must be on sys.path (not just ``third_party``).
SANA_ROOT = THIRD_PARTY / "Sana"
# facebookresearch/vjepa2 imports its modules as ``app.vjepa_2_1.*``
# (the repo root *is* a package). Adding ``third_party/vjepa2`` itself
# (not just ``third_party/``) makes those imports resolve to the
# submodule's ``app/`` directory.
VJEPA2_ROOT = THIRD_PARTY / "vjepa2"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))
if SANA_ROOT.exists() and str(SANA_ROOT) not in sys.path:
    sys.path.insert(0, str(SANA_ROOT))
if VJEPA2_ROOT.is_dir() and str(VJEPA2_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA2_ROOT))

# Trigger the OpenWAM SANA package's mmcv 1.x → mmengine compatibility shim
# before any test module's top-level ``import diffusion...`` probe runs.
# Wrapped so a missing SANA install / missing openwam SANA subpackage does not
# block CPU-only test collection.
try:  # noqa: SIM105
    import openwam.model.video_backbone.sana  # noqa: F401
except Exception:
    pass


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: test requires GPU")
    config.addinivalue_line("markers", "data: test requires RoboTwin data")
