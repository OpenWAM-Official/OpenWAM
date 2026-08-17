"""LIBERO eval adapter for the OpenWAM Policy Server.

The adapter intentionally lives in the benchmark client environment and talks to
an already-running OpenWAM WebSocket server. The model, checkpoint, preprocessing
and action denormalization stay server-side.

EEF10 contract (``action_mode: eef``, the default): the checkpoint was trained
on the raw 10-D single-arm EEF pose ``[xyz3, rot6d6, gripper1]`` (world frame,
FULL pose) mapped to the unified 80-D space. The deploy server gathers the
model's unified output back to raw EEF10 and unnormalizes it, so this client:

* sends the live proprio as raw EEF10 (``libero_obs_to_eef10`` — byte-consistent
  with the canonical dataset converter), and
* converts the returned EEF10 full-pose target to the env's native 7-D OSC
  delta (``eef10_to_libero7d``) using the live controller scales, with the
  CURRENT achieved EEF pose as the delta reference.

The trained gripper channel is an open-scale (``-1 = closed, +1 = open``);
``eef10_to_libero7d`` negates it back to LIBERO's own ``+1 = close`` command, so
nothing here handles the gripper directly. A checkpoint trained before that flip
will drive the gripper inverted — retrain or pin an older client.

``action_mode: native`` keeps the legacy passthrough for checkpoints trained
directly on the 7-D OSC command.
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

from benchmarks.utils import (  # noqa: E402
    WSPolicyClient,
    client,
    eef10_to_libero7d,
    libero_obs_to_eef10,
    quat_xyzw_to_rot6d,
    resize_for_lshape_slot,
    transport,
)
from benchmarks.utils.action_conversion import (  # noqa: E402
    LIBERO_EEF10_DIM,
    LIBERO_OSC_POS_SCALE_DEFAULT,
    LIBERO_OSC_ROT_SCALE_DEFAULT,
)

# Obs keys the EEF10 proprio (and the OSC delta reference) are built from.
PROPRIO_OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def probe_osc_scales(env) -> tuple[float, float]:
    """Best-effort read of the live OSC_POSE ``output_max`` (pos m / rot rad per
    unit action) from a LIBERO env; falls back to the robosuite defaults.

    LIBERO's ``OffScreenRenderEnv`` wraps a robosuite env at ``env.env``; the
    Panda arm controller carries ``output_max = [0.05]*3 + [0.5]*3``.
    """
    candidates = [env, getattr(env, "env", None)]
    for cand in candidates:
        robots = getattr(cand, "robots", None)
        if not robots:
            continue
        controller = getattr(robots[0], "controller", None)
        output_max = getattr(controller, "output_max", None)
        if output_max is None:
            continue
        arr = np.asarray(output_max, np.float64).reshape(-1)
        if arr.shape[0] >= 6 and np.all(arr[:6] > 0):
            return float(arr[0]), float(arr[3])
    print(
        "[OpenWAMLiberoPolicy] WARNING: could not probe OSC output_max from env; "
        f"falling back to defaults pos={LIBERO_OSC_POS_SCALE_DEFAULT} rot={LIBERO_OSC_ROT_SCALE_DEFAULT}"
    )
    return LIBERO_OSC_POS_SCALE_DEFAULT, LIBERO_OSC_ROT_SCALE_DEFAULT


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
        action_mode: str = "eef",
        head_camera_key: str = "agentview_image",
        left_wrist_camera_key: str | None = "robot0_eye_in_hand_image",
        right_wrist_camera_key: str | None = None,
        image_transform: str = "rotate_180",
        send_state: bool = True,
        state_keys: list[str] | None = None,
        state_dim: int | None = None,
        action_dim: int = 7,
        action_indices: list[int] | None = None,
        action_clip: float | None = None,
        osc_pos_scale: float | None = None,
        osc_rot_scale: float | None = None,
        env=None,
        debug: bool = False,
        debug_dir: str = "./debug_libero",
    ) -> None:
        if action_mode not in ("eef", "native"):
            raise ValueError(f"action_mode must be 'eef' or 'native', got {action_mode!r}")
        if image_transform not in ("none", "rotate_180"):
            raise ValueError("image_transform must be 'none' or 'rotate_180'")

        self._ws_url = f"ws://{host}:{port}"
        self._client = WSPolicyClient(self._ws_url, timeout=request_timeout)
        self._action_mode = action_mode
        self._head_camera_key = head_camera_key
        self._left_wrist_camera_key = left_wrist_camera_key
        self._right_wrist_camera_key = right_wrist_camera_key
        self._image_transform = image_transform
        self._send_state = send_state
        self._state_keys = state_keys or []
        self._action_indices = action_indices
        self._action_clip = action_clip
        self._debug = debug
        self._debug_dir = Path(debug_dir)
        self._episode = -1
        self._step = 0
        if debug:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

        if action_mode == "eef":
            # Server returns raw EEF10; the client bridges to the 7-D OSC delta.
            self._state_dim = state_dim if state_dim is not None else LIBERO_EEF10_DIM
            self._action_dim = int(action_dim)
            if osc_pos_scale is None or osc_rot_scale is None:
                probed_pos, probed_rot = probe_osc_scales(env) if env is not None else (
                    LIBERO_OSC_POS_SCALE_DEFAULT,
                    LIBERO_OSC_ROT_SCALE_DEFAULT,
                )
                osc_pos_scale = osc_pos_scale if osc_pos_scale is not None else probed_pos
                osc_rot_scale = osc_rot_scale if osc_rot_scale is not None else probed_rot
            self._osc_pos_scale = float(osc_pos_scale)
            self._osc_rot_scale = float(osc_rot_scale)
        else:
            self._state_dim = state_dim
            self._action_dim = int(action_dim)
            self._osc_pos_scale = None
            self._osc_rot_scale = None

        pong = self._client.ping()
        if pong.get("type") != transport.PONG:
            raise RuntimeError(f"OpenWAM server ping returned unexpected response: {pong}")

        print(
            f"[OpenWAMLiberoPolicy] server={self._ws_url} action_mode={action_mode} "
            f"action_dim={self._action_dim} image_transform={image_transform} "
            f"send_state={send_state} osc_scales=({self._osc_pos_scale}, {self._osc_rot_scale})"
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
            head=client.encode_numpy_b64(
                resize_for_lshape_slot(self._image(obs, self._head_camera_key), "head_camera")
            ),
            left_wrist=self._maybe_encode(obs, self._left_wrist_camera_key, "left_wrist_camera"),
            right_wrist=self._maybe_encode(obs, self._right_wrist_camera_key, "right_wrist_camera"),
            prompt=prompt,
            state=self._state(obs),
        )
        response = self._client.predict(payload)
        action = np.asarray(response["action"], dtype=np.float32).reshape(-1)
        if self._action_indices is not None:
            action = action[self._action_indices]
        raw_eef10 = None
        if self._action_mode == "eef":
            if action.shape[0] != LIBERO_EEF10_DIM:
                raise ValueError(
                    f"OpenWAM returned action dim {action.shape[0]}, expected raw EEF10 "
                    f"({LIBERO_EEF10_DIM}) for action_mode='eef'"
                )
            raw_eef10 = action.copy()
            action = self._bridge_eef10(obs, action)
        if action.shape[0] != self._action_dim:
            raise ValueError(f"OpenWAM returned action dim {action.shape[0]}, expected {self._action_dim}")
        if self._action_clip is not None:
            action = np.clip(action, -float(self._action_clip), float(self._action_clip))
        self._maybe_debug(obs, payload, action, raw_eef10)
        self._step += 1
        return action

    def _bridge_eef10(self, obs: dict, eef10: np.ndarray) -> np.ndarray:
        """EEF10 full pose -> LIBERO 7-D OSC delta, referenced to the CURRENT
        achieved EEF pose from the live obs."""
        missing = [k for k in PROPRIO_OBS_KEYS[:2] if k not in obs]
        if missing:
            raise KeyError(f"LIBERO obs missing EEF reference key(s): {missing}")
        ref_pos = np.asarray(obs["robot0_eef_pos"], np.float32).reshape(-1)
        ref_rot6d = quat_xyzw_to_rot6d(np.asarray(obs["robot0_eef_quat"], np.float32).reshape(-1)[None])[0]
        return eef10_to_libero7d(
            eef10,
            ref_pos,
            ref_rot6d,
            pos_scale=self._osc_pos_scale,
            rot_scale=self._osc_rot_scale,
        )

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

    def _maybe_encode(self, obs: dict, key: str | None, slot: str) -> str | None:
        if not key:
            return None
        if key not in obs or obs[key] is None:
            return None
        return client.encode_numpy_b64(resize_for_lshape_slot(self._image(obs, key), slot))

    def _state(self, obs: dict) -> list[float] | None:
        if not self._send_state:
            return None
        if self._action_mode == "eef":
            missing = [k for k in PROPRIO_OBS_KEYS if k not in obs]
            if missing:
                raise KeyError(f"LIBERO obs missing proprio key(s): {missing}")
            state = libero_obs_to_eef10(
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"],
            ).tolist()
        else:
            state = _build_state(obs, self._state_keys)
        if self._state_dim is not None and len(state) != self._state_dim:
            raise ValueError(f"LIBERO state dim {len(state)} != expected {self._state_dim}")
        return state

    def _maybe_debug(self, obs: dict, payload: dict, action: np.ndarray, raw_eef10) -> None:
        if not self._debug:
            return
        step_dir = self._debug_dir / f"episode_{self._episode:03d}" / f"step_{self._step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "prompt": payload["prompt"],
            "action_mode": self._action_mode,
            "state_dim": len(payload.get("state", [])) if "state" in payload else None,
            "state": payload.get("state"),
            "action": action.tolist(),
            "raw_eef10": None if raw_eef10 is None else np.asarray(raw_eef10, np.float32).reshape(-1).tolist(),
            "osc_scales": [self._osc_pos_scale, self._osc_rot_scale],
            "obs_keys": sorted(obs.keys()),
        }
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
