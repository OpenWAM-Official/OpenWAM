"""Shared test fixtures and sys.path setup."""

import sys
from pathlib import Path

import pytest

# Ensure project root and third-party packages are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: test requires GPU")
    config.addinivalue_line("markers", "data: test requires RoboTwin data")


class _StubReason1Encoder:
    """Signature-compatible stand-in for ``Reason1LiveTextEncoder`` so GPU
    tests don't pay the 16 GB Qwen load. Returns pre-projection
    ``(B, 512, 100352)`` like the real encoder."""

    def __init__(self, ckpt_path, *, dtype=None, device=None):
        import torch

        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.device = torch.device(device) if device is not None else torch.device("cpu")

    def __call__(self, prompts):
        import torch

        if isinstance(prompts, str):
            prompts = [prompts]
        return torch.randn(len(prompts), 512, 100352, dtype=self.dtype, device=self.device)

    def to(self, *, dtype=None, device=None):
        import torch

        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = torch.device(device)
        return self


@pytest.fixture
def stub_reason1(monkeypatch):
    monkeypatch.setattr(
        "openwam.model.video_backbone.cosmos_predict25.text_encoder.Reason1LiveTextEncoder",
        _StubReason1Encoder,
    )
