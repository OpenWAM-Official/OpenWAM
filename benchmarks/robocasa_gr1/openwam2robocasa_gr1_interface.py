"""RoboCasa GR1 tabletop adapter for the OpenWAM Policy Server."""

from __future__ import annotations

import os as _os
import sys as _sys

_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import json  # noqa: E402
from collections.abc import Iterable, Mapping  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from benchmarks.utils import WSPolicyClient, client, transport  # noqa: E402


def _as_vector(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _state_keys(obs: Mapping, configured: Iterable[str] | None) -> list[str]:
    if configured:
        return [str(key) for key in configured]
    return sorted(str(key) for key in obs if str(key).startswith("state."))


def build_state(obs: Mapping, keys: Iterable[str] | None = None) -> list[float]:
    """Flatten RoboCasa GR00T-style state fields in deterministic order."""
    state: list[float] = []
    missing: list[str] = []
    for key in _state_keys(obs, keys):
        if key not in obs:
            missing.append(key)
            continue
        state.extend(_as_vector(obs[key]).tolist())
    if missing:
        raise KeyError(f"RoboCasa obs missing state key(s): {missing}")
    return state


def action_vector_to_dict(
    action: Iterable[float],
    action_space,
    action_keys: Iterable[str] | None = None,
    action_indices: Iterable[int] | None = None,
    action_clip: float | None = None,
) -> dict:
    """Split a flat OpenWAM action vector into a RoboCasa / GR00T action dict."""
    vector = _as_vector(action)
    if action_indices is not None:
        vector = vector[list(action_indices)]
    if action_clip is not None:
        vector = np.clip(vector, -float(action_clip), float(action_clip))

    spaces = getattr(action_space, "spaces", None)
    if spaces is None:
        raise TypeError("RoboCasa action_space must be a gymnasium.spaces.Dict")

    keys = [str(key) for key in action_keys] if action_keys else sorted(str(key) for key in spaces)
    out = {}
    offset = 0
    for key in keys:
        if key not in spaces:
            raise KeyError(f"RoboCasa action_space missing key: {key}")
        space = spaces[key]
        shape = getattr(space, "shape", None)
        if shape is None:
            raise TypeError(f"Discrete action space is not supported for key '{key}'")
        dim = int(np.prod(shape))
        chunk = vector[offset : offset + dim]
        if chunk.shape[0] != dim:
            raise ValueError(f"OpenWAM action too short for key '{key}': need {dim}, have {chunk.shape[0]}")
        out[key] = chunk.reshape(shape).astype(np.float32, copy=False)
        offset += dim

    if offset != vector.shape[0]:
        raise ValueError(
            f"OpenWAM returned action dim {vector.shape[0]}, but RoboCasa action mapping consumed {offset}"
        )
    return out


def zero_action(action_space) -> dict:
    """Build a deterministic all-zero action for a Dict action space."""
    spaces = getattr(action_space, "spaces", None)
    if spaces is None:
        raise TypeError("RoboCasa action_space must be a gymnasium.spaces.Dict")
    return {
        key: (0 if getattr(space, "shape", None) is None else np.zeros(space.shape, dtype=np.float32))
        for key, space in spaces.items()
    }


class OpenWAMRoboCasaGR1Policy:
    """Policy client that maps RoboCasa GR1 observations/actions to OpenWAM."""

    def __init__(
        self,
        action_space,
        host: str = "127.0.0.1",
        port: int = 8848,
        request_timeout: int = 300,
        head_camera_key: str = "video.ego_view_pad_res256_freq20",
        left_wrist_camera_key: str | None = None,
        right_wrist_camera_key: str | None = None,
        prompt_key: str = "annotation.human.coarse_action",
        fallback_prompt_key: str = "annotation.human.action.task_description",
        send_state: bool = True,
        state_keys: list[str] | None = None,
        state_dim: int | None = None,
        action_keys: list[str] | None = None,
        action_indices: list[int] | None = None,
        action_clip: float | None = None,
        debug: bool = False,
        debug_dir: str = "./debug_robocasa_gr1",
    ) -> None:
        self._action_space = action_space
        self._head_camera_key = head_camera_key
        self._left_wrist_camera_key = left_wrist_camera_key
        self._right_wrist_camera_key = right_wrist_camera_key
        self._prompt_key = prompt_key
        self._fallback_prompt_key = fallback_prompt_key
        self._send_state = send_state
        self._state_keys = state_keys
        self._state_dim = state_dim
        self._action_keys = action_keys
        self._action_indices = action_indices
        self._action_clip = action_clip
        self._debug = debug
        self._debug_dir = Path(debug_dir)
        self._episode = -1
        self._step = 0
        if debug:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

        self._ws_url = f"ws://{host}:{port}"
        self._client = WSPolicyClient(self._ws_url, timeout=request_timeout)
        pong = self._client.ping()
        if pong.get("type") != transport.PONG:
            raise RuntimeError(f"OpenWAM server ping returned unexpected response: {pong}")

        action_dims = {
            key: int(np.prod(space.shape))
            for key, space in getattr(action_space, "spaces", {}).items()
            if getattr(space, "shape", None) is not None
        }
        print(
            f"[OpenWAMRoboCasaGR1Policy] server={self._ws_url} send_state={send_state} "
            f"state_dim={state_dim} action_dims={action_dims}"
        )

    def close(self) -> None:
        self._client.close()

    def reset(self) -> None:
        self._episode += 1
        self._step = 0
        ack = self._client.reset()
        if ack.get("type") != transport.RESET_ACK:
            raise RuntimeError(f"OpenWAM server reset returned unexpected response: {ack}")

    def act(self, obs: Mapping) -> dict:
        state = build_state(obs, self._state_keys) if self._send_state else None
        if self._state_dim is not None and state is not None and len(state) != self._state_dim:
            raise ValueError(f"RoboCasa state dim {len(state)} != expected {self._state_dim}")
        prompt = str(obs.get(self._prompt_key) or obs.get(self._fallback_prompt_key) or obs.get("language", ""))
        payload = client.build_payload(
            head=client.encode_numpy_b64(self._image(obs, self._head_camera_key)),
            left_wrist=self._maybe_encode(obs, self._left_wrist_camera_key),
            right_wrist=self._maybe_encode(obs, self._right_wrist_camera_key),
            prompt=prompt,
            state=state,
        )
        response = self._client.predict(payload)
        action = action_vector_to_dict(
            response["action"],
            self._action_space,
            action_keys=self._action_keys,
            action_indices=self._action_indices,
            action_clip=self._action_clip,
        )
        self._maybe_debug(obs, payload, response["action"], action)
        self._step += 1
        return action

    def _image(self, obs: Mapping, key: str) -> np.ndarray:
        if key not in obs:
            raise KeyError(f"RoboCasa obs missing camera key: {key}")
        image = np.asarray(obs[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera '{key}' must be HxWx3, got {image.shape}")
        return image.astype(np.uint8, copy=False)

    def _maybe_encode(self, obs: Mapping, key: str | None) -> str | None:
        if not key or key not in obs or obs[key] is None:
            return None
        return client.encode_numpy_b64(self._image(obs, key))

    def _maybe_debug(self, obs: Mapping, payload: dict, raw_action, action: Mapping[str, np.ndarray]) -> None:
        if not self._debug:
            return
        step_dir = self._debug_dir / f"episode_{self._episode:03d}" / f"step_{self._step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "prompt": payload["prompt"],
            "state_dim": len(payload.get("state", [])) if "state" in payload else None,
            "raw_action_dim": len(raw_action),
            "action_shapes": {key: list(value.shape) for key, value in action.items()},
            "obs_keys": sorted(obs.keys()),
        }
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
