"""BaseWAMArchitecture.copy_deploy_artifacts dispatches to every backbone.

Used by the trainer so different backbones (Wan tokenizer, Cosmos
tokenizer+processor, future ones) can ship their own deploy artifacts
without the trainer importing them directly.
"""

from __future__ import annotations

import torch.nn as nn

from openwam.model.architectures.architecture_base import BaseWAMArchitecture


class _RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, object]] = []

    def copy_deploy_artifacts(self, output_dir, cfg):
        self.calls.append((output_dir, cfg))


class _ConcreteArch(BaseWAMArchitecture):
    """Concrete BaseWAMArchitecture so we can exercise the dispatcher in isolation."""

    def forward(self, *args, **kwargs):  # pragma: no cover - never invoked
        raise NotImplementedError


def test_dispatch_to_all_backbones(tmp_path):
    arch = _ConcreteArch(cfg=None)
    vb = _RecordingBackbone()
    ab = _RecordingBackbone()
    arch.video_backbone = vb
    arch.action_backbone = ab

    cfg = {"model": {"video_backbone": {"model_path": str(tmp_path)}}}
    arch.copy_deploy_artifacts(str(tmp_path), cfg)

    assert vb.calls == [(str(tmp_path), cfg)]
    assert ab.calls == [(str(tmp_path), cfg)]


def test_backbone_without_hook_is_skipped(tmp_path):
    arch = _ConcreteArch(cfg=None)
    arch.video_backbone = nn.Linear(1, 1)  # no copy_deploy_artifacts attribute
    arch.action_backbone = _RecordingBackbone()

    arch.copy_deploy_artifacts(str(tmp_path), cfg={})
    assert arch.action_backbone.calls == [(str(tmp_path), {})]
