"""BaseWAMArchitecture.save_assets_for_deployment dispatches save_deploy_assets
to every backbone that implements it.

Lets different backbones (Wan component specs + tokenizer, future ones) ship
their own deploy assets without the trainer importing them directly.
"""

from __future__ import annotations

import torch.nn as nn

from openwam.model.architectures.architecture_base import BaseWAMArchitecture


class _RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, object]] = []

    def save_deploy_assets(self, output_dir, cfg):
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
    arch.save_assets_for_deployment(str(tmp_path), cfg)

    assert vb.calls == [(str(tmp_path), cfg)]
    assert ab.calls == [(str(tmp_path), cfg)]


def test_backbone_without_hook_is_skipped(tmp_path):
    arch = _ConcreteArch(cfg=None)
    arch.video_backbone = nn.Linear(1, 1)  # no save_deploy_assets attribute
    arch.action_backbone = _RecordingBackbone()

    arch.save_assets_for_deployment(str(tmp_path), cfg={})
    assert arch.action_backbone.calls == [(str(tmp_path), {})]
