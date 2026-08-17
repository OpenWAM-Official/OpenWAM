"""RoboCasa GR1 tabletop adapter for the OpenWAM Policy Server."""

from __future__ import annotations

import os as _os
import sys as _sys

_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import json  # noqa: E402
from collections.abc import Mapping  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from benchmarks.utils import WSPolicyClient, client, resize_for_lshape_slot, transport  # noqa: E402
from openwam.dataloader.utils.gr1_kinematics import EEF33_DIM, GR1Kinematics  # noqa: E402


def _as_vector(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


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
        env=None,
        action_mode: str = "eef",
        host: str = "127.0.0.1",
        port: int = 8848,
        request_timeout: int = 300,
        head_camera_key: str = "video.ego_view_pad_res256_freq20",
        left_wrist_camera_key: str | None = None,
        right_wrist_camera_key: str | None = None,
        prompt_key: str = "annotation.human.coarse_action",
        fallback_prompt_key: str = "annotation.human.action.task_description",
        send_state: bool = True,
        state_dim: int | None = None,
        debug: bool = False,
        debug_dir: str = "./debug_robocasa_gr1",
    ) -> None:
        self._action_space = action_space
        self._action_mode = str(action_mode).lower()
        if self._action_mode != "eef":
            raise ValueError(f"RoboCasa GR1 client supports only action_mode='eef', got {action_mode!r}")
        if env is None:
            raise ValueError("RoboCasa GR1 EEF deployment requires env=... for live FK/IK")
        self._env = env
        self._kinematics = GR1Kinematics.from_env(env)
        self._head_camera_key = head_camera_key
        self._left_wrist_camera_key = left_wrist_camera_key
        self._right_wrist_camera_key = right_wrist_camera_key
        self._prompt_key = prompt_key
        self._fallback_prompt_key = fallback_prompt_key
        self._send_state = send_state
        self._state_dim = state_dim
        self._debug = debug
        self._debug_dir = Path(debug_dir)
        self._episode = -1
        self._step = 0
        self._ik_failures = 0
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
        self._ik_failures = 0
        # RoboCasa rebuilds its MuJoCo simulation on reset; discard stale
        # MjModel/MjData handles before computing the next episode's FK/IK.
        self._kinematics = GR1Kinematics.from_env(self._env)
        ack = self._client.reset()
        if ack.get("type") != transport.RESET_ACK:
            raise RuntimeError(f"OpenWAM server reset returned unexpected response: {ack}")

    def act(self, obs: Mapping) -> dict:
        state = self._kinematics.observation_to_eef33(obs).tolist() if self._send_state else None
        if self._state_dim is not None and state is not None and len(state) != self._state_dim:
            raise ValueError(f"RoboCasa state dim {len(state)} != expected {self._state_dim}")
        prompt = str(obs.get(self._prompt_key) or obs.get(self._fallback_prompt_key) or obs.get("language", ""))
        payload = client.build_payload(
            head=client.encode_numpy_b64(
                resize_for_lshape_slot(self._image(obs, self._head_camera_key), "head_camera")
            ),
            left_wrist=self._maybe_encode(obs, self._left_wrist_camera_key, "left_wrist_camera"),
            right_wrist=self._maybe_encode(obs, self._right_wrist_camera_key, "right_wrist_camera"),
            prompt=prompt,
            state=state,
        )
        response = self._client.predict(payload)
        raw_action = _as_vector(response["action"])
        if raw_action.shape != (EEF33_DIM,):
            raise ValueError(f"OpenWAM GR1 server must return EEF33 after deploy gather, got {raw_action.shape}")
        action, ik = self._kinematics.eef33_to_action_dict(raw_action)
        if not ik.converged:
            self._ik_failures += 1
            if self._ik_failures == 1 or self._ik_failures % 50 == 0:
                print(
                    f"[OpenWAMRoboCasaGR1Policy] WARNING: IK not converged {self._ik_failures}x this episode "
                    f"(step={self._step}, pos_err={ik.position_error:.4f}, rot_err={ik.rotation_error:.4f}); "
                    "holding current arm pose"
                )
        self._maybe_debug(obs, payload, raw_action, action, ik=ik)
        self._step += 1
        return action

    def _image(self, obs: Mapping, key: str) -> np.ndarray:
        if key not in obs:
            raise KeyError(f"RoboCasa obs missing camera key: {key}")
        image = np.asarray(obs[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera '{key}' must be HxWx3, got {image.shape}")
        return image.astype(np.uint8, copy=False)

    def _maybe_encode(self, obs: Mapping, key: str | None, slot: str) -> str | None:
        if not key or key not in obs or obs[key] is None:
            return None
        return client.encode_numpy_b64(resize_for_lshape_slot(self._image(obs, key), slot))

    def _maybe_debug(
        self,
        obs: Mapping,
        payload: dict,
        raw_action,
        action: Mapping[str, np.ndarray],
        *,
        ik=None,
    ) -> None:
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
            "ik": (
                {
                    "converged": bool(ik.converged),
                    "position_error": float(ik.position_error),
                    "rotation_error": float(ik.rotation_error),
                }
                if ik is not None
                else None
            ),
        }
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
