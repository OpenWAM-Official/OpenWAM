"""
OpenWAM: A Modular Open-Source Library for Systematic WAM Training, Inference and Deployment.

Open World-Action Model — uses video diffusion as a world model and jointly
generates future video frames + robot actions via flow matching.
"""

import sys
from pathlib import Path

__version__ = "0.1.0"

# Make third_party packages (diffsynth) importable as top-level modules.
# This allows legacy code to continue using `from diffsynth...` imports
# after the move to third_party/diffsynth/.
_THIRD_PARTY = str(Path(__file__).resolve().parent.parent / "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)
