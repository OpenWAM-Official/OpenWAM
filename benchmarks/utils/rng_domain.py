"""Isolate benchmark-owned legacy RNG state from same-process policy code."""

from __future__ import annotations

import base64
import random
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

_STATE_LOCK = threading.RLock()


def encode_numpy_state(state: tuple[Any, ...]) -> dict[str, Any]:
    if not isinstance(state, tuple) or len(state) != 5:
        raise ValueError("NumPy legacy RNG state must be a five-item tuple")
    bit_generator, keys, position, has_gauss, cached_gaussian = state
    array = np.asarray(keys)
    if bit_generator != "MT19937" or array.dtype != np.dtype("uint32") or array.shape != (624,):
        raise ValueError("only the NumPy MT19937 legacy RNG state is supported")
    if isinstance(position, bool) or not isinstance(position, (int, np.integer)) or not 0 <= int(position) <= 624:
        raise ValueError("invalid NumPy MT19937 position")
    if int(has_gauss) not in (0, 1):
        raise ValueError("invalid NumPy Gaussian-cache flag")
    canonical = np.ascontiguousarray(array.astype("<u4", copy=False))
    return {
        "kind": "numpy.random.RandomState",
        "bit_generator": "MT19937",
        "keys": {
            "dtype": "<u4",
            "shape": [624],
            "data_b64": base64.b64encode(canonical.tobytes(order="C")).decode("ascii"),
        },
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def decode_numpy_state(payload: dict[str, Any]) -> tuple[Any, ...]:
    if not isinstance(payload, dict):
        raise TypeError("serialized NumPy RNG state must be an object")
    if payload.get("kind") != "numpy.random.RandomState" or payload.get("bit_generator") != "MT19937":
        raise ValueError("unsupported NumPy RNG-state generator")
    keys = payload.get("keys")
    if not isinstance(keys, dict) or keys.get("dtype") != "<u4" or keys.get("shape") != [624]:
        raise ValueError("invalid NumPy MT19937 key metadata")
    try:
        raw = base64.b64decode(keys["data_b64"], validate=True)
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid NumPy MT19937 key payload") from exc
    if len(raw) != 624 * 4:
        raise ValueError("invalid NumPy MT19937 key payload length")
    array = np.frombuffer(raw, dtype="<u4").astype(np.uint32, copy=True)
    position = payload.get("position")
    has_gauss = payload.get("has_gauss")
    cached = payload.get("cached_gaussian")
    if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position <= 624:
        raise ValueError("invalid NumPy MT19937 position")
    if isinstance(has_gauss, bool) or has_gauss not in (0, 1):
        raise ValueError("invalid NumPy Gaussian-cache flag")
    if isinstance(cached, bool) or not isinstance(cached, (int, float)) or not np.isfinite(cached):
        raise ValueError("invalid NumPy Gaussian-cache value")
    return "MT19937", array, position, has_gauss, float(cached)


def _torch_module():
    try:
        import torch
    except ImportError:
        return None
    return torch


def seeded_global_state(seed: int) -> dict[str, Any]:
    """Create a complete private Python/NumPy/Torch RNG domain."""

    with _STATE_LOCK:
        outside = capture_global_state()
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch = _torch_module()
            if torch is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            return capture_global_state()
        finally:
            restore_global_state(outside)


def capture_global_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    torch = _torch_module()
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state().clone()
        if torch.cuda.is_available():
            state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_global_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch = _torch_module()
    if torch is not None and "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])


class GlobalRngDomain:
    """A movable RNG cursor activated only around its owner's calls.

    The lock makes swapping module-global RNG state safe between serialized
    callers.  It intentionally does not support concurrent calls inside one
    process; use process isolation for that topology.
    """

    def __init__(self, state: dict[str, Any]):
        self._state = state

    @classmethod
    def from_seed(cls, seed: int) -> "GlobalRngDomain":
        return cls(seeded_global_state(seed))

    @classmethod
    def from_numpy_state(cls, numpy_state: tuple[Any, ...]) -> "GlobalRngDomain":
        state = seeded_global_state(0)
        state["numpy"] = numpy_state
        return cls(state)

    @contextmanager
    def activate(self) -> Iterator[None]:
        with _STATE_LOCK:
            outside = capture_global_state()
            restore_global_state(self._state)
            try:
                yield
            finally:
                self._state = capture_global_state()
                restore_global_state(outside)

    @property
    def numpy_state(self) -> tuple[Any, ...]:
        return self._state["numpy"]
