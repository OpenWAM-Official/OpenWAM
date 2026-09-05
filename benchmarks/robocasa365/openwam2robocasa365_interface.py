"""OpenWAM evaluation adapter for canonical RoboCasa365 native-action state19/action15.

The client sends the same 19-D physical state stored by the converted training
dataset. The server returns a 15-D compact action whose EEF portion already
represents the native normalized delta command. Rot6d is inverted to the native
3-D rotation vector without applying the controller's 0.5 physical scale and
without using the current achieved EEF observation.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import base64  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Iterable, Optional  # noqa: E402

import numpy as np  # noqa: E402

from benchmarks.utils import (  # noqa: E402
    WSPolicyClient,
    binarize_robocasa_action12,
    build_payload,
    encode_numpy_b64,
    quat_xyzw_to_rot6d,
    resize_for_lshape_slot,
    robocasa_state_to_eef10,
    rot6d_to_axis_angle,
    transport,
)

ACTION_SLICES = {
    "action.end_effector_position": (0, 3),
    "action.end_effector_rotation": (3, 6),
    "action.gripper_close": (6, 7),
    "action.base_motion": (7, 11),
    "action.control_mode": (11, 12),
}
ACTION_DIM = 12
POLICY_ACTION_DIM = 15
STATE_DIM = 19
REPRESENTATION = "robocasa365"
DEFAULT_HEAD_CAMERA_KEY = "video.robot0_agentview_left"
DEFAULT_LEFT_WRIST_CAMERA_KEY = "video.robot0_eye_in_hand"
DEFAULT_RIGHT_CAMERA_KEY = "video.robot0_agentview_right"

DEFAULT_STATE_KEYS = [
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
]
PROPRIO_KEYS = tuple(DEFAULT_STATE_KEYS)
IMAGE_SLOTS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
_SLOT_STEMS = {"head_camera": "head", "left_wrist_camera": "left", "right_wrist_camera": "right"}


def assemble_state(obs: dict, state_keys: Iterable[str]) -> list:
    values: list[float] = []
    missing = []
    for key in state_keys:
        if key not in obs:
            missing.append(key)
        else:
            values.extend(np.asarray(obs[key], np.float32).reshape(-1).tolist())
    if missing:
        raise KeyError(f"RoboCasa365 obs missing state key(s): {missing}")
    return values


def assemble_state19_proprio(obs: dict) -> list:
    missing = [key for key in PROPRIO_KEYS if key not in obs]
    if missing:
        raise KeyError(f"RoboCasa365 obs missing proprio key(s): {missing}")
    eef10 = robocasa_state_to_eef10(
        obs["state.end_effector_position_relative"],
        obs["state.end_effector_rotation_relative"],
        obs["state.gripper_qpos"],
    )
    base_position = np.asarray(obs["state.base_position"], np.float32).reshape(-1)[:3]
    base_rot6d = quat_xyzw_to_rot6d(np.asarray(obs["state.base_rotation"], np.float32).reshape(-1)[:4])
    state19 = np.concatenate([eef10, base_position, base_rot6d]).astype(np.float32)
    if state19.shape != (STATE_DIM,):
        raise ValueError(f"assembled state shape {state19.shape} != ({STATE_DIM},)")
    return state19.tolist()


def slice_action(flat) -> dict:
    flat = np.asarray(flat, np.float32).reshape(-1)
    if flat.shape != (ACTION_DIM,):
        raise ValueError(f"expected a {ACTION_DIM}-D env action, got {flat.shape}")
    return {key: flat[start:end] for key, (start, end) in ACTION_SLICES.items()}


def transform_image(image, mode: str) -> np.ndarray:
    if mode not in ("none", "rotate_180"):
        raise ValueError("image_transform must be 'none' or 'rotate_180'")
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3, got {value.shape}")
    value = value.astype(np.uint8, copy=False)
    return value[::-1, ::-1] if mode == "rotate_180" else value


def build_obs_payload(
    obs: dict,
    *,
    head_camera_key: str,
    left_wrist_camera_key: Optional[str],
    right_wrist_camera_key: Optional[str],
    image_transform: str,
    state_keys: Iterable[str],
    prompt: str,
) -> dict:
    def encode(key: Optional[str], *, required: bool, slot: str) -> Optional[str]:
        if not key:
            if required:
                raise KeyError("head_camera_key is required")
            return None
        if key not in obs or obs[key] is None:
            if required:
                raise KeyError(f"RoboCasa365 obs missing camera key: {key}")
            return None
        image = transform_image(obs[key], image_transform)
        # Match the unchanged training reader's pre-composition LANCZOS resize.
        return encode_numpy_b64(resize_for_lshape_slot(image, slot))

    return build_payload(
        head=encode(head_camera_key, required=True, slot="head_camera"),
        left_wrist=encode(left_wrist_camera_key, required=False, slot="left_wrist_camera"),
        # The transport calls this fixed L-shape slot "right_wrist_camera",
        # but RoboCasa365 intentionally fills it with robot0_agentview_right.
        # An explicit None still permits the legacy black-slot behavior.
        right_wrist=encode(
            right_wrist_camera_key,
            required=bool(right_wrist_camera_key),
            slot="right_wrist_camera",
        ),
        prompt=prompt,
        state=assemble_state19_proprio(obs),
    )


def _montage(images: list, labels: list):
    from PIL import Image, ImageDraw, ImageFont

    tile_h, strip_h, gap = 256, 18, 4
    font = ImageFont.load_default()
    tiles = []
    for image, label in zip(images, labels):
        if image is None:
            image = Image.new("RGB", (tile_h, tile_h), (0, 0, 0))
        else:
            width, height = image.size
            image = image.resize((max(1, round(width * tile_h / height)), tile_h))
        strip = Image.new("RGB", (image.width, strip_h), (0, 0, 0))
        ImageDraw.Draw(strip).text((2, 3), label, fill=(255, 255, 255), font=font)
        tile = Image.new("RGB", (image.width, tile_h + strip_h), (0, 0, 0))
        tile.paste(strip, (0, 0))
        tile.paste(image, (0, strip_h))
        tiles.append(tile)
    canvas = Image.new(
        "RGB",
        (sum(tile.width for tile in tiles) + gap * (len(tiles) - 1), tile_h + strip_h),
        (0, 0, 0),
    )
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width + gap
    return canvas


def dump_obs_debug(
    obs: dict,
    payload: dict,
    out_dir,
    *,
    state_keys: Iterable[str] = DEFAULT_STATE_KEYS,
    action=None,
    episode=None,
    step=None,
    server_step=None,
    latency_ms=None,
    save_montage: bool = True,
    expected_state_dim: int = STATE_DIM,
) -> Path:
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for slot in IMAGE_SLOTS:
        encoded = payload.get("images", {}).get(slot)
        stem = _SLOT_STEMS[slot]
        if encoded is None:
            images.append(None)
            (out_dir / f"{stem}_missing.txt").write_text(f"{slot} not sent", encoding="utf-8")
            continue
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        image.save(out_dir / f"{stem}.png", format="PNG")
        images.append(image)
    if save_montage:
        _montage(images, list(IMAGE_SLOTS)).save(out_dir / "cameras.png")
    state = payload.get("state")
    metadata = {
        "episode": episode,
        "step": step,
        "prompt": payload.get("prompt", ""),
        "state": state,
        "state_breakdown": {
            key: np.asarray(obs[key], dtype=float).reshape(-1).tolist() for key in state_keys if key in obs
        },
        "image_slots": {slot: None if image is None else list(image.size) for slot, image in zip(IMAGE_SLOTS, images)},
        "server_step": server_step,
        "latency_ms": latency_ms,
        "checks": {
            "state_dim_ok": state is not None and len(state) == expected_state_dim,
            "head_and_wrist_present": (
                payload.get("images", {}).get("head_camera") is not None
                and payload.get("images", {}).get("left_wrist_camera") is not None
            ),
        },
    }
    if action is not None:
        flat = np.asarray(action, np.float32).reshape(-1).tolist()
        metadata["action"] = flat
        metadata["action_sliced"] = {key: flat[start:end] for key, (start, end) in ACTION_SLICES.items()}
        metadata["checks"]["action_dim_is_12"] = len(flat) == ACTION_DIM
    (out_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


class OpenWAMRoboCasa365Policy:
    """Bridge native-delta compact action15 to RoboCasa's native action12."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8848,
        request_timeout: int = 300,
        head_camera_key: str = DEFAULT_HEAD_CAMERA_KEY,
        left_wrist_camera_key: Optional[str] = DEFAULT_LEFT_WRIST_CAMERA_KEY,
        right_wrist_camera_key: Optional[str] = DEFAULT_RIGHT_CAMERA_KEY,
        image_transform: str = "none",
        state_keys: Optional[list] = None,
        state_dim: Optional[int] = None,
        action_dim: int = ACTION_DIM,
        debug: bool = False,
        debug_dir: str = "./debug_robocasa365",
        _client=None,
    ) -> None:
        if image_transform not in ("none", "rotate_180"):
            raise ValueError("image_transform must be 'none' or 'rotate_180'")
        self._client = _client or WSPolicyClient(f"ws://{host}:{port}", timeout=request_timeout)
        self._head_camera_key = head_camera_key
        self._left_wrist_camera_key = left_wrist_camera_key
        self._right_wrist_camera_key = right_wrist_camera_key
        self._image_transform = image_transform
        self._state_keys = list(state_keys) if state_keys else list(DEFAULT_STATE_KEYS)
        self._state_dim = STATE_DIM if state_dim is None else int(state_dim)
        self._action_dim = int(action_dim)
        self._debug = bool(debug)
        self._debug_dir = debug_dir
        self._episode = -1
        self._step = 0

        pong = self._client.ping()
        if pong.get("type") != transport.PONG:
            raise RuntimeError(f"OpenWAM server ping returned unexpected response: {pong}")
        if pong.get("representation") != REPRESENTATION:
            raise RuntimeError(
                f"representation mismatch: eval requires {REPRESENTATION!r}, server advertises {pong.get('representation')!r}"
            )
        print(
            f"[OpenWAMRoboCasa365Policy] state_dim={self._state_dim} "
            f"policy_action_dim={POLICY_ACTION_DIM} "
            f"env_action_dim={self._action_dim} image_transform={image_transform}"
        )

    def close(self) -> None:
        self._client.close()

    def reset(self) -> None:
        self._episode += 1
        self._step = 0
        ack = self._client.reset()
        if ack.get("type") != transport.RESET_ACK:
            raise RuntimeError(f"OpenWAM server reset returned unexpected response: {ack}")

    def _bridge_action15(self, action15: np.ndarray) -> np.ndarray:
        action15 = np.asarray(action15, np.float32).reshape(-1)
        if action15.shape != (POLICY_ACTION_DIM,):
            raise ValueError(f"expected compact action15, got {action15.shape}")
        delta_xyz = np.clip(action15[0:3], -1.0, 1.0)
        delta_rotvec = np.clip(rot6d_to_axis_angle(action15[3:9]), -1.0, 1.0)
        native_gripper = -action15[9:10]
        base5 = action15[10:15]
        # Environment flat order is EEF xyz3 + rotvec3 + gripper1 + base4 +
        # control_mode1.  Dataset source order is rearranged into this contract.
        return np.concatenate([delta_xyz, delta_rotvec, native_gripper, base5[:4], base5[4:5]]).astype(np.float32)

    def act(self, obs: dict, prompt: str) -> dict:
        payload = build_obs_payload(
            obs,
            head_camera_key=self._head_camera_key,
            left_wrist_camera_key=self._left_wrist_camera_key,
            right_wrist_camera_key=self._right_wrist_camera_key,
            image_transform=self._image_transform,
            state_keys=self._state_keys,
            prompt=prompt,
        )
        if len(payload["state"]) != self._state_dim:
            raise ValueError(f"RoboCasa365 state dim {len(payload['state'])} != expected {self._state_dim}")
        response = self._client.predict(payload)
        flat = np.asarray(response["action"], np.float32).reshape(-1)
        if flat.shape[0] == POLICY_ACTION_DIM:
            flat = self._bridge_action15(flat)
        if flat.shape[0] != self._action_dim:
            raise ValueError(f"OpenWAM returned action dim {flat.shape[0]}, expected {self._action_dim}")
        flat = binarize_robocasa_action12(flat)
        if self._debug:
            dump_obs_debug(
                obs,
                payload,
                Path(self._debug_dir) / f"ep{self._episode:04d}" / f"step_{self._step:04d}",
                state_keys=self._state_keys,
                action=flat,
                episode=self._episode,
                step=self._step,
                server_step=response.get("step"),
                latency_ms=response.get("latency_ms"),
                save_montage=self._step == 0,
                expected_state_dim=self._state_dim,
            )
        self._step += 1
        return slice_action(flat)


__all__ = [
    "ACTION_DIM",
    "ACTION_SLICES",
    "DEFAULT_HEAD_CAMERA_KEY",
    "DEFAULT_LEFT_WRIST_CAMERA_KEY",
    "DEFAULT_RIGHT_CAMERA_KEY",
    "DEFAULT_STATE_KEYS",
    "OpenWAMRoboCasa365Policy",
    "POLICY_ACTION_DIM",
    "STATE_DIM",
    "assemble_state",
    "assemble_state19_proprio",
    "build_obs_payload",
    "dump_obs_debug",
    "slice_action",
    "transform_image",
]
