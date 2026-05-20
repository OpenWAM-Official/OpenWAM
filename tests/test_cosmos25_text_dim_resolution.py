"""Auto-resolve text_dim from VideoBackbone.context_dim.

Backbones whose text encoder differs from Wan's T5-XXL (4096) report their
embedding dim via :attr:`VideoBackbone.context_dim`. Architectures that read
``cfg.text_dim`` must fall back to that value when the config omits it, and
must keep honoring an explicit ``cfg.text_dim`` when it is set.
"""

from __future__ import annotations

from openwam.model.architectures.base import BaseWAMArchitecture


class _CtxStub:
    """Minimal video backbone stand-in for `_resolve_text_dim` unit testing."""

    def __init__(self, context_dim):
        self.context_dim = context_dim


class _ConcreteArch(BaseWAMArchitecture):
    """Concrete BaseWAMArchitecture subclass — `forward` is never called by these tests."""

    def forward(self, *args, **kwargs):  # pragma: no cover - never invoked
        raise NotImplementedError


def _make_arch(context_dim=None):
    arch = _ConcreteArch(cfg=None)
    arch.video_backbone = _CtxStub(context_dim) if context_dim is not None else None
    return arch


def test_explicit_text_dim_wins_over_backbone():
    arch = _make_arch(context_dim=2048)
    assert arch._resolve_text_dim({"text_dim": 4096}) == 4096


def test_fallback_to_backbone_context_dim():
    arch = _make_arch(context_dim=2048)
    assert arch._resolve_text_dim({}) == 2048


def test_default_when_neither_cfg_nor_backbone_provides():
    arch = _make_arch(context_dim=None)
    assert arch._resolve_text_dim({}) == 4096


def test_default_when_backbone_context_dim_is_none():
    arch = _make_arch(context_dim=None)
    arch.video_backbone = _CtxStub(None)
    assert arch._resolve_text_dim({}) == 4096


def test_custom_default_overrides_4096():
    arch = _make_arch(context_dim=None)
    assert arch._resolve_text_dim({}, default=3072) == 3072
