"""BaseWAMArchitecture.save_assets_for_deployment dispatches save_deploy_assets
to every backbone unconditionally — each backbone base declares the hook
(default no-op), so there is no hasattr probing.

Lets different backbones (Wan component specs + tokenizer, future ones) ship
their own deploy assets without the trainer importing them directly.
"""

from __future__ import annotations

import torch.nn as nn

from openwam.model.architectures.base import BaseWAMArchitecture


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


def test_all_backbone_bases_declare_hook():
    """Direction-A contract: every backbone base declares save_deploy_assets on
    itself (default no-op), so the dispatcher calls it unconditionally rather than
    probing with hasattr. ``vars`` (not hasattr) catches a root that forgot it."""
    from openwam.model.action_backbone.base import ActionDiTBackbone, SharedActionBackbone
    from openwam.model.video_backbone.base import VideoBackbone
    from openwam.model.vlm_backbone.base import VlmBackbone

    for base in (VideoBackbone, SharedActionBackbone, ActionDiTBackbone, VlmBackbone):
        assert "save_deploy_assets" in vars(base), f"{base.__name__} must declare the no-op hook"
