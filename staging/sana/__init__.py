"""SANA-Video backbone integration for OpenWAM.

This sub-package wires NVlabs/SANA into the OpenWAM ``VideoBackbone`` ABC.
Public entry point is :class:`SanaVideoBackbone`; the registry is set up in
``openwam.model.video_backbone.__init__`` so config-driven instantiation works
via ``build_video_backbone("sana_video_2b", cfg)``.

Phase 0 scope (this commit): scaffolding only — the adapter loads
``SanaMSVideo`` from ``third_party/Sana``, runs the standard
``prepare → run_block × N → finalize`` lifecycle, and exposes
``pre_attn_at_layer`` returning the dual-track (rotated + unrotated) Q/K
that a future ``SanaMoTJointDriver`` will need. No MoT integration yet —
see ``plans/sana_mot_integration_plan.md`` for the broader roadmap.
"""

# SANA's upstream Python tree is rooted at ``third_party/Sana/`` and uses
# top-level ``from diffusion.model.*`` imports (the ``diffusion/`` package
# lives at ``third_party/Sana/diffusion/``). Tests handle this in
# ``tests/conftest.py``; for production callers (``scripts/train.py`` /
# ``scripts/deploy.py`` / direct library users) we add ``third_party/Sana``
# to ``sys.path`` here so the SANA adapter is self-contained — no
# ``PYTHONPATH`` knob required. Idempotent (won't double-insert) and
# guarded on the submodule actually being present.
import sys as _sys
from pathlib import Path as _Path

_SANA_ROOT = _Path(__file__).resolve().parents[4] / "third_party" / "Sana"
if _SANA_ROOT.is_dir() and str(_SANA_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SANA_ROOT))

# Compat shim: SANA's ``diffusion.model.builder`` does ``from mmcv import Registry``
# (the mmcv 1.x API). mmcv 2.x moved ``Registry`` into ``mmengine``, and the
# CUDA-built ``mmcv==1.7.2`` SANA's pyproject pins doesn't install cleanly under
# torch 2.7 + CUDA 12.8. We install ``mmcv-lite`` + ``mmengine`` instead and
# back-fill the missing symbol here so the SANA import chain loads cleanly.
# Must run BEFORE any ``diffusion.*`` import — kept in this package's __init__
# so it fires the first time anything touches the OpenWAM SANA adapter.
try:
    import types as _types

    import mmcv as _mmcv

    if not hasattr(_mmcv, "Registry"):
        from mmengine import Registry as _MMEngineRegistry

        _mmcv.Registry = _MMEngineRegistry  # type: ignore[attr-defined]
    if not hasattr(_mmcv, "build_from_cfg"):
        from mmengine.registry import build_from_cfg as _mmengine_build_from_cfg

        _mmcv.build_from_cfg = _mmengine_build_from_cfg  # type: ignore[attr-defined]

    # mmcv 1.x exposed ``mmcv.utils.logging.logger_initialized`` (a global dict
    # tracking which loggers have been set up). mmengine handles logger
    # bookkeeping differently and doesn't expose this name. SANA's
    # ``diffusion/utils/logger.py`` only uses it as a presence-check dict, so
    # an empty dict is a behaviour-preserving stand-in.
    if "mmcv.utils.logging" not in _sys.modules:
        _mmcv_utils = getattr(_mmcv, "utils", None)
        if _mmcv_utils is None:
            _mmcv_utils = _types.ModuleType("mmcv.utils")
            _sys.modules["mmcv.utils"] = _mmcv_utils
            _mmcv.utils = _mmcv_utils  # type: ignore[attr-defined]
        _mmcv_utils_logging = _types.ModuleType("mmcv.utils.logging")
        _mmcv_utils_logging.logger_initialized = {}  # type: ignore[attr-defined]
        _sys.modules["mmcv.utils.logging"] = _mmcv_utils_logging
        _mmcv_utils.logging = _mmcv_utils_logging  # type: ignore[attr-defined]

    # mmcv 1.x ``mmcv.runner`` is gone in mmcv 2.x; the few symbols SANA's
    # import chain touches at module-import time (``get_dist_info``) live in
    # mmengine. We stub the module so dist_utils.py loads; runtime-only
    # symbols (``build_optimizer``, ``OPTIMIZER_BUILDERS`` etc.) are not
    # exercised by ``sana_multi_scale_video`` and remain unresolved on
    # purpose — accessing them will fail with a clear AttributeError.
    if "mmcv.runner" not in _sys.modules:
        from mmengine.dist import get_dist_info as _get_dist_info

        _mmcv_runner = _types.ModuleType("mmcv.runner")
        _mmcv_runner.get_dist_info = _get_dist_info  # type: ignore[attr-defined]
        _sys.modules["mmcv.runner"] = _mmcv_runner
        _mmcv.runner = _mmcv_runner  # type: ignore[attr-defined]
except Exception:
    # If neither mmcv nor mmengine is installed, the SANA import will fail
    # downstream with a clearer error than whatever we'd raise here.
    pass

# noqa: E402 — these imports intentionally follow the sys.path injection
# (sana adapter / blocks_split chain into ``diffusion.*`` at import time,
# which requires ``third_party/Sana`` on sys.path) and the mmcv→mmengine
# compatibility shim above.
from openwam.model.video_backbone.sana.adapter import SanaVideoBackbone  # noqa: E402
from openwam.model.video_backbone.sana.blocks_split import SanaMSVideoSplit  # noqa: E402

__all__ = ["SanaVideoBackbone", "SanaMSVideoSplit"]
