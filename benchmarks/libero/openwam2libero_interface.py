"""LIBERO eval adapter for the OpenWAM Policy Server.

The adapter intentionally lives in the benchmark client environment and talks to
an already-running OpenWAM WebSocket server. The model, checkpoint, preprocessing
and action denormalization stay server-side.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Iterable  # noqa: E402

import numpy as np  # noqa: E402

from benchmarks.utils import WSPolicyClient, client, transport  # noqa: E402


def _as_list(value) -> list[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr.tolist()


def _build_state(obs: dict, keys: Iterable[str]) -> list[float]:
    state: list[float] = []
    missing: list[str] = []
    for key in keys:
        if key not in obs:
            missing.append(key)
            continue
        state.extend(_as_list(obs[key]))
    if missing:
        raise KeyError(f"LIBERO obs missing state key(s): {missing}")
    return state


class OpenWAMLiberoPolicy:
    """Small policy client used by `single_eval.py`."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8848,
        request_timeout: int = 300,
        head_camera_key: str = "agentview_image",
        left_wrist_camera_key: str | None = "robot0_eye_in_hand_image",
        right_wrist_camera_key: str | None = None,
        image_transform: str = "rotate_180",
        send_state: bool = False,
        state_keys: list[str] | None = None,
        state_dim: int | None = None,
        action_dim: int = 7,
        action_indices: list[int] | None = None,
        action_clip: float | None = None,
        debug: bool = False,
        debug_dir: str = "./debug_libero",
    ) -> None:
        self._ws_url = f"ws://{host}:{port}"
        self._client = WSPolicyClient(self._ws_url, timeout=request_timeout)
        self._head_camera_key = head_camera_key
        self._left_wrist_camera_key = left_wrist_camera_key
        self._right_wrist_camera_key = right_wrist_camera_key
        self._image_transform = image_transform
        self._send_state = send_state
        self._state_keys = state_keys or []
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._action_indices = action_indices
        self._action_clip = action_clip
        self._debug = debug
        self._debug_dir = Path(debug_dir)
        self._episode = -1
        self._step = 0
        if debug:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

        if image_transform not in ("none", "rotate_180"):
            raise ValueError("image_transform must be 'none' or 'rotate_180'")

        pong = self._client.ping()
        if pong.get("type") != transport.PONG:
            raise RuntimeError(f"OpenWAM server ping returned unexpected response: {pong}")

        print(
            f"[OpenWAMLiberoPolicy] server={self._ws_url} action_dim={action_dim} "
            f"image_transform={image_transform} send_state={send_state} state_keys={self._state_keys}"
        )

    def close(self) -> None:
        self._client.close()

    def reset(self) -> None:
        self._episode += 1
        self._step = 0
        ack = self._client.reset()
        if ack.get("type") != transport.RESET_ACK:
            raise RuntimeError(f"OpenWAM server reset returned unexpected response: {ack}")

    def act(self, obs: dict, prompt: str) -> np.ndarray:
        payload = client.build_payload(
            head=client.encode_numpy_b64(self._image(obs, self._head_camera_key)),
            left_wrist=self._maybe_encode(obs, self._left_wrist_camera_key),
            right_wrist=self._maybe_encode(obs, self._right_wrist_camera_key),
            prompt=prompt,
            state=self._state(obs),
        )
        response = self._client.predict(payload)
        action = np.asarray(response["action"], dtype=np.float32).reshape(-1)
        if self._action_indices is not None:
            action = action[self._action_indices]
        if action.shape[0] != self._action_dim:
            raise ValueError(f"OpenWAM returned action dim {action.shape[0]}, expected {self._action_dim}")
        if self._action_clip is not None:
            action = np.clip(action, -float(self._action_clip), float(self._action_clip))
        self._maybe_debug(obs, payload, action)
        self._step += 1
        return action

    def _image(self, obs: dict, key: str) -> np.ndarray:
        if key not in obs:
            raise KeyError(f"LIBERO obs missing camera key: {key}")
        image = np.asarray(obs[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera '{key}' must be HxWx3, got {image.shape}")
        image = image.astype(np.uint8, copy=False)
        if self._image_transform == "rotate_180":
            image = image[::-1, ::-1]
        return image

    def _maybe_encode(self, obs: dict, key: str | None) -> str | None:
        if not key:
            return None
        if key not in obs or obs[key] is None:
            return None
        return client.encode_numpy_b64(self._image(obs, key))

    def _state(self, obs: dict) -> list[float] | None:
        if not self._send_state:
            return None
        state = _build_state(obs, self._state_keys)
        if self._state_dim is not None and len(state) != self._state_dim:
            raise ValueError(f"LIBERO state dim {len(state)} != expected {self._state_dim}")
        return state

    def _maybe_debug(self, obs: dict, payload: dict, action: np.ndarray) -> None:
        if not self._debug:
            return
        step_dir = self._debug_dir / f"episode_{self._episode:03d}" / f"step_{self._step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "prompt": payload["prompt"],
            "state_dim": len(payload.get("state", [])) if "state" in payload else None,
            "action": action.tolist(),
            "obs_keys": sorted(obs.keys()),
        }
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
