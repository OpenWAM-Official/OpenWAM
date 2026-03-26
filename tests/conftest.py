"""Shared test fixtures and sys.path setup."""

import sys
from pathlib import Path

import pytest

# Ensure project root and legacy WAM dir are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAM_DIR = PROJECT_ROOT / "examples" / "wanvideo" / "wam"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WAM_DIR) not in sys.path:
    sys.path.insert(0, str(WAM_DIR))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: test requires GPU")
    config.addinivalue_line("markers", "data: test requires RoboTwin data")
