"""Attention implementation auto-selection for ActionDiT.

Automatically selects the most efficient available attention backend,
following the priority: Flash Attention 3 > Flash Attention 2 >
Sage Attention > xFormers > PyTorch native SDPA.

Override via the ``WAM_ATTENTION_IMPL`` environment variable:
    flash3, flash2, sage, xformers, sdpa

The video DiT already routes through diffsynth's attention system;
this module only affects ActionDiT and other open_wam-owned models.
"""

import logging
import os
from typing import Callable

import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

_ATTENTION_FN: Callable | None = None


def _try_flash_attn_3() -> Callable | None:
    try:
        from flash_attn_interface import flash_attn_func as flash3_fn  # type: ignore

        def _flash3(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            # flash_attn_3 expects (B, S, H, D), we receive (B, H, S, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash3_fn(q, k, v)
            return out.transpose(1, 2)

        return _flash3
    except ImportError:
        return None


def _try_flash_attn_2() -> Callable | None:
    try:
        from flash_attn import flash_attn_func  # type: ignore

        def _flash2(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash_attn_func(q, k, v)
            return out.transpose(1, 2)

        return _flash2
    except ImportError:
        return None


def _try_sage_attention() -> Callable | None:
    try:
        from sageattention import sageattn  # type: ignore

        def _sage(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return sageattn(q, k, v)

        return _sage
    except ImportError:
        return None


def _try_xformers() -> Callable | None:
    try:
        from xformers.ops import memory_efficient_attention  # type: ignore

        def _xformers(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            # xformers expects (B, S, H, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = memory_efficient_attention(q, k, v)
            return out.transpose(1, 2)

        return _xformers
    except ImportError:
        return None


def _sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return F.scaled_dot_product_attention(q, k, v)


_BACKEND_MAP = {
    "flash3": _try_flash_attn_3,
    "flash2": _try_flash_attn_2,
    "sage": _try_sage_attention,
    "xformers": _try_xformers,
    "sdpa": lambda: _sdpa,
}

_AUTO_PRIORITY = ["flash3", "flash2", "sage", "xformers", "sdpa"]


def get_attention_fn() -> Callable:
    """Return the best available attention function.

    Signature: ``(q, k, v) -> out`` where tensors are ``(B, H, S, D)``.
    """
    global _ATTENTION_FN
    if _ATTENTION_FN is not None:
        return _ATTENTION_FN

    override = os.environ.get("WAM_ATTENTION_IMPL", "").strip().lower()

    if override:
        if override not in _BACKEND_MAP:
            raise ValueError(f"Unknown WAM_ATTENTION_IMPL='{override}'. Choose from: {list(_BACKEND_MAP.keys())}")
        fn = _BACKEND_MAP[override]()
        if fn is None:
            logger.warning(
                "WAM_ATTENTION_IMPL='%s' requested but not available, falling back to auto-detect",
                override,
            )
        else:
            logger.info("Using attention backend: %s (explicit)", override)
            _ATTENTION_FN = fn
            return _ATTENTION_FN

    for name in _AUTO_PRIORITY:
        fn = _BACKEND_MAP[name]()
        if fn is not None:
            logger.info("Using attention backend: %s (auto-detected)", name)
            _ATTENTION_FN = fn
            return _ATTENTION_FN

    # Should never reach here since sdpa always works
    _ATTENTION_FN = _sdpa
    return _ATTENTION_FN
