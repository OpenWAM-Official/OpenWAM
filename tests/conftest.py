"""Shared test fixtures and sys.path setup."""

import sys
from pathlib import Path

# Ensure project root and third-party packages are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
# facebookresearch/vjepa2 imports its modules as ``app.vjepa_2_1.*``
# (the repo root *is* a package). Adding ``third_party/vjepa2`` itself
# (not just ``third_party/``) makes those imports resolve to the
# submodule's ``app/`` directory.
VJEPA2_ROOT = THIRD_PARTY / "vjepa2"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))
if VJEPA2_ROOT.is_dir() and str(VJEPA2_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA2_ROOT))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: test requires GPU")
    config.addinivalue_line("markers", "data: test requires RoboTwin data")
