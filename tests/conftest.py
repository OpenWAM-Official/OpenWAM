"""Shared test fixtures and sys.path setup."""

import sys
from pathlib import Path

import pytest

# Ensure project root, legacy WAM dir, and third-party packages are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAM_DIR = PROJECT_ROOT / "examples" / "wanvideo" / "wam"
THIRD_PARTY = PROJECT_ROOT / "third_party"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WAM_DIR) not in sys.path:
    sys.path.insert(0, str(WAM_DIR))
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: test requires GPU")
    config.addinivalue_line("markers", "data: test requires RoboTwin data")
