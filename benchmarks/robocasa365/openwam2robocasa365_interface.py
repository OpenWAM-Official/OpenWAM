"""RoboCasa365 eval adapter for the OpenWAM policy server.

Lives in the benchmark client environment and talks to an already-running
OpenWAM WebSocket server. The model, checkpoint, image preprocessing and action
denormalization all stay server-side. Mirrors ``benchmarks/libero`` but for the
single-arm PandaOmron 16-D state layout that
``robocasa.wrappers.gym_wrapper.RoboCasaGymEnv`` produces.

Action spaces (two layers — don't conflate):
  * ``RoboCasaGymEnv`` consumes a fixed **12-D robosuite OSC + base** action
    (``eef_pos3 + eef_rot3 + grip1 + base4 + mode1``, the env's native delta-OSC).
  * The OpenWAM model predicts a **20-D full base-relative EEF pose** (repo-standard EEF schema, dual
    of robotwin) — "full pose" (not the env's per-step OSC *delta*), expressed in the robot **base
    frame** (``robot0_base_to_eef_*``), NOT world-frame; the bridge converts it to the OSC delta.
    With the default **mobile base** it ALSO emits the 5-D base command, so the server
    returns **25-D** ``[arm20, base5]``. ``act()`` bridges the arm 20-D → 12-D via
    ``benchmarks.utils.eef20d_to_robocasa12d`` and passes ``base5`` (x/y/yaw vel, torso,
    control_mode) through. A 20-D (arm-only) action bridges with a zero base; a 12-D server action
    is passed through unchanged. The arm bridge needs the env's OSC scaling (``osc_pos_scale`` /
    ``osc_rot_scale`` from the OSC_POSE controller config; unset → raises). The OSC delta's reference
    is **control_mode-aware** (achieved → current eef; desired / base-mode → previous target), so the
    arm stays placed while the base drives.

Proprio: the client sends the model's **20-D single-arm EEF** proprio (converted from the env's raw
16-D state). For a ``base_proprio_velocity`` checkpoint (set ``base_proprio_velocity: true`` in the
policy config), it appends the body-frame base velocity → **23-D** ``[arm20, base_vel3]``.

Debug dumping follows the robotwin convention (``ep{N}/step_{N}/`` with per-camera
JPGs + ``meta.json``) and adds a labeled ``cameras.png`` montage plus
state/action breakdowns and pass/fail ``checks`` for quick verification.
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
    base_velocity_body,
    build_payload,
    eef20d_to_robocasa12d,
    encode_numpy_b64,
    quat_xyzw_to_rot6d,
    robocasa_state_to_eef20d,
    transport,
)

# Flat 12-D server action -> the action dict RoboCasaGymEnv.step() consumes.
# Order matches robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py.
ACTION_SLICES = {
    "action.end_effector_position": (0, 3),
    "action.end_effector_rotation": (3, 6),
    "action.gripper_close": (6, 7),
    "action.base_motion": (7, 11),
    "action.control_mode": (11, 12),
}
ACTION_DIM = 12

# Raw 16-D obs state keys (kept for the debug breakdown). The model is NOT trained on
# this raw 16-D vector — it trains on the 20-D single-arm EEF proprio (see below), so the
# client converts before sending. DEFAULT_STATE_KEYS is only used for the debug display now.
DEFAULT_STATE_KEYS = [
    "state.base_position",                    # 3
    "state.base_rotation",                    # 4
    "state.end_effector_position_relative",   # 3
    "state.end_effector_rotation_relative",   # 4
    "state.gripper_qpos",                     # 2
]
# The 3 obs keys the 20-D EEF proprio is built from (base POSE dropped, like training; base
# VELOCITY is added separately from BASE_POSE_KEYS when base_proprio_velocity).
PROPRIO_EEF_KEYS = (
    "state.end_effector_position_relative",   # 3
    "state.end_effector_rotation_relative",   # 4 (quat xyzw)
    "state.gripper_qpos",                     # 2
)
# Proprio is sent as the SAME 20-D EEF representation the model trains on
# (RoboCasa365Dataset proprio = eef20d[0:1]); the server validates this dim and normalizes it.
STATE_DIM = 20
# base_proprio_velocity ckpts append the current body-frame base velocity [vx, vy, vyaw] → 23-D
# proprio [arm20, base_vel3]; the server normalizes base_vel with its 'base_vel' stats block and
# scatters it into the unified reserved slots [68:71) (dual of the action base command in [68:73)).
BASE_VEL_DIM = 3
# The two obs keys the body-frame base velocity is finite-differenced from (world base pose).
BASE_POSE_KEYS = ("state.base_position", "state.base_rotation")  # 3 + 4 = 7-D pose

# Fixed client-side camera slots the server expects (head required). The stems
# match robotwin's per-camera debug JPG names (head.jpg / left.jpg / right.jpg).
IMAGE_SLOTS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
_SLOT_STEMS = {"head_camera": "head", "left_wrist_camera": "left", "right_wrist_camera": "right"}


def assemble_state(obs: dict, state_keys: Iterable[str]) -> list:
    """Concatenate the raw proprio state keys (in order) into a flat float list.

    Kept for the debug breakdown; NOT what gets sent (see ``assemble_eef20d_proprio``).
    """
    state: list = []
    missing: list = []
    for key in state_keys:
        if key not in obs:
            missing.append(key)
            continue
        state.extend(np.asarray(obs[key], dtype=np.float32).reshape(-1).tolist())
    if missing:
        raise KeyError(f"RoboCasa365 obs missing state key(s): {missing}")
    return state


def assemble_eef20d_proprio(obs: dict, base_vel: Optional[np.ndarray] = None) -> list:
    """Build the single-arm EEF proprio (RAW) the model trains on, from a RoboCasa obs.

    Mirrors the dataloader's proprio exactly (``state_to_arm10`` + ``assemble_single_arm_left``):
    ``[eef_pos_rel(3), rot6d(eef_rot_rel quat,6), gripper_separation(1), <right 10 zeros>]``.
    Sent raw (physical) — the server normalizes. This replaces the stale 16-D raw send so the
    deploy proprio matches the trained representation (the dual of robotwin's _extract_eef_proprio).

    When ``base_vel`` (3-D body-frame ``[vx, vy, vyaw]``, raw) is given — a base_proprio_velocity
    ckpt — it is appended → 23-D ``[arm20, base_vel3]``; the server normalizes+scatters it to [68:71).
    """
    missing = [k for k in PROPRIO_EEF_KEYS if k not in obs]
    if missing:
        raise KeyError(f"RoboCasa365 obs missing proprio key(s): {missing}")
    eef20d = robocasa_state_to_eef20d(
        obs["state.end_effector_position_relative"],
        obs["state.end_effector_rotation_relative"],
        obs["state.gripper_qpos"],
    )
    proprio = eef20d.astype(np.float32).reshape(-1).tolist()
    if base_vel is not None:
        proprio.extend(np.asarray(base_vel, np.float32).reshape(-1)[:BASE_VEL_DIM].tolist())
    return proprio


def slice_action(flat) -> dict:
    """Split the flat 12-D server action into the 5-key RoboCasaGymEnv action dict."""
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    if flat.shape[0] != ACTION_DIM:
        raise ValueError(f"expected a {ACTION_DIM}-D action, got {flat.shape[0]}")
    return {key: flat[start:end] for key, (start, end) in ACTION_SLICES.items()}


def transform_image(image, mode: str) -> np.ndarray:
    """Return an H×W×3 uint8 image.

    ``RoboCasaGymEnv`` already flips offscreen frames upright (``img[::-1]``), so
    the default ``mode='none'`` is a passthrough; ``'rotate_180'`` is kept as an
    escape hatch for checkpoints trained on un-flipped frames.
    """
    if mode not in ("none", "rotate_180"):
        raise ValueError("image_transform must be 'none' or 'rotate_180'")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3, got {arr.shape}")
    arr = arr.astype(np.uint8, copy=False)
    if mode == "rotate_180":
        return arr[::-1, ::-1]
    return arr


def build_obs_payload(
    obs: dict,
    *,
    head_camera_key: str,
    left_wrist_camera_key: Optional[str],
    right_wrist_camera_key: Optional[str],
    image_transform: str,
    state_keys: Iterable[str],
    prompt: str,
    base_vel: Optional[np.ndarray] = None,
) -> dict:
    """Turn a RoboCasaGymEnv obs dict into an OpenWAM obs payload (no server).

    Encodes the 3 camera slots (head required) and assembles the proprio state. ``base_vel`` (when
    given, a base_proprio_velocity ckpt) is appended to the proprio → 23-D.
    """

    def _encode(key: Optional[str], *, required: bool) -> Optional[str]:
        if not key:
            if required:
                raise KeyError("head_camera_key is required")
            return None
        if key not in obs or obs[key] is None:
            if required:
                raise KeyError(f"RoboCasa365 obs missing camera key: {key}")
            return None
        return encode_numpy_b64(transform_image(obs[key], image_transform))

    return build_payload(
        head=_encode(head_camera_key, required=True),
        left_wrist=_encode(left_wrist_camera_key, required=False),
        right_wrist=_encode(right_wrist_camera_key, required=False),
        prompt=prompt,
        # Send the 20-D EEF proprio the model trains on (NOT the raw 16-D), + base_vel3 when a
        # base_proprio_velocity ckpt. state_keys is retained for the debug breakdown only.
        state=assemble_eef20d_proprio(obs, base_vel=base_vel),
    )


def _montage(images: list, labels: list):
    """Lay decoded camera frames side by side with a label strip (black = empty, matching the server's black-fill)."""
    from PIL import Image, ImageDraw, ImageFont

    tile_h, strip_h, gap = 256, 18, 4
    font = ImageFont.load_default()
    tiles = []
    for im, label in zip(images, labels):
        if im is None:
            im = Image.new("RGB", (tile_h, tile_h), (0, 0, 0))
        else:
            w, h = im.size
            im = im.resize((max(1, round(w * tile_h / h)), tile_h))
        strip = Image.new("RGB", (im.width, strip_h), (0, 0, 0))
        ImageDraw.Draw(strip).text((2, 3), label, fill=(255, 255, 255), font=font)
        tile = Image.new("RGB", (im.width, tile_h + strip_h), (0, 0, 0))
        tile.paste(strip, (0, 0))
        tile.paste(im, (0, strip_h))
        tiles.append(tile)
    total_w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    canvas = Image.new("RGB", (total_w, tile_h + strip_h), (0, 0, 0))
    x = 0
    for t in tiles:
        canvas.paste(t, (x, 0))
        x += t.width + gap
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
    """Write one step's debug bundle (robotwin layout + a montage + checks).

    Files (under ``out_dir``, conventionally ``{debug_dir}/ep{N}/step_{N}/``):
      ``head.jpg`` / ``left.jpg`` / ``right.jpg`` - the frames the client sent
          (a missing slot writes a ``{stem}_missing.txt`` stub, like robotwin).
      ``cameras.png`` - the same 3 frames decoded + labeled side by side.
      ``meta.json`` - robotwin fields (episode/step/prompt/state/action/
          server_step/latency_ms) plus state_breakdown / action_sliced / checks.
    """
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for slot in IMAGE_SLOTS:
        b64 = payload.get("images", {}).get(slot)
        stem = _SLOT_STEMS[slot]
        if b64 is None:
            images.append(None)
            (out_dir / f"{stem}_missing.txt").write_text(f"{slot} not sent", encoding="utf-8")
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        im.save(out_dir / f"{stem}.jpg", format="JPEG")
        images.append(im)
    if save_montage:
        _montage(images, list(IMAGE_SLOTS)).save(out_dir / "cameras.png")

    state = payload.get("state")
    meta = {
        "episode": episode,
        "step": step,
        "prompt": payload.get("prompt", ""),
        "state": state,
        "state_breakdown": {
            key: np.asarray(obs[key], dtype=float).reshape(-1).tolist() for key in state_keys if key in obs
        },
        "image_slots": {slot: (None if im is None else list(im.size)) for slot, im in zip(IMAGE_SLOTS, images)},
        "server_step": server_step,
        "latency_ms": latency_ms,
    }
    checks = {
        "state_dim_ok": state is not None and len(state) == expected_state_dim,
        "head_and_wrist_present": (
            payload.get("images", {}).get("head_camera") is not None
            and payload.get("images", {}).get("left_wrist_camera") is not None
        ),
    }
    if action is not None:
        flat = np.asarray(action, dtype=np.float32).reshape(-1).tolist()
        meta["action"] = flat
        meta["action_sliced"] = {key: flat[start:end] for key, (start, end) in ACTION_SLICES.items()}
        checks["action_dim_is_12"] = len(flat) == ACTION_DIM
    meta["checks"] = checks
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


class OpenWAMRoboCasa365Policy:
    """WebSocket policy client for RoboCasa365 single-arm eval.

    ``act(obs, prompt)`` returns the action **dict** that
    ``RoboCasaGymEnv.step()`` expects. Pass ``_client`` to inject a fake
    transport in unit tests (skips the real WebSocket connection). With
    ``debug=True`` it writes a per-step bundle under
    ``{debug_dir}/ep{N}/step_{N}/`` (montage only on the first step of each episode).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8848,
        request_timeout: int = 300,
        head_camera_key: str = "video.robot0_agentview_left",   # 3rd-person workspace view
        left_wrist_camera_key: Optional[str] = "video.robot0_eye_in_hand",  # the single arm's real wrist cam
        right_wrist_camera_key: Optional[str] = None,  # single-arm has no 2nd wrist -> server black-fills this slot
        image_transform: str = "none",
        state_keys: Optional[list] = None,
        state_dim: Optional[int] = None,
        action_dim: int = ACTION_DIM,
        osc_pos_scale: Optional[float] = None,
        osc_rot_scale: Optional[float] = None,
        base_proprio_velocity: bool = False,
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
        # base_proprio_velocity ckpts append the 3-D body-frame base velocity to the proprio → 23-D.
        # Must match the ckpt's dataloader.base_proprio_velocity (set from the eval policy config).
        self._base_proprio_velocity = bool(base_proprio_velocity)
        # Expected proprio width for the fail-fast guard: 20-D EEF (+ 3-D base velocity when on).
        self._state_dim = (
            state_dim if state_dim is not None else STATE_DIM + (BASE_VEL_DIM if self._base_proprio_velocity else 0)
        )
        self._action_dim = action_dim
        self._osc_pos_scale = osc_pos_scale
        self._osc_rot_scale = osc_rot_scale
        self._debug = debug
        self._debug_dir = debug_dir
        self._episode = -1
        self._step = 0
        # Previous step's absolute arm target (pos3 + rot6d6) — the reference for the OSC delta when
        # the env is in "desired" goal-update mode (control_mode>=0.5). Cleared per episode in reset().
        self._prev_target_pose: Optional[np.ndarray] = None
        # Previous step's world base pose (pos3 + quat4) — the reference for the body-frame base
        # velocity finite-diff (base_proprio_velocity). Cleared per episode in reset().
        self._prev_base_pose: Optional[np.ndarray] = None

        pong = self._client.ping()
        if pong.get("type") != transport.PONG:
            raise RuntimeError(f"OpenWAM server ping returned unexpected response: {pong}")
        print(
            f"[OpenWAMRoboCasa365Policy] action_dim={action_dim} state_dim={self._state_dim} "
            f"base_proprio_velocity={self._base_proprio_velocity} image_transform={image_transform} "
            f"cameras=({head_camera_key}, {left_wrist_camera_key}, {right_wrist_camera_key})"
        )

    def close(self) -> None:
        self._client.close()

    def reset(self) -> None:
        self._episode += 1
        self._step = 0
        self._prev_target_pose = None  # new episode: OSC goal re-inits to the current eef
        self._prev_base_pose = None  # new episode: base velocity re-inits to zero at the first step
        ack = self._client.reset()
        if ack.get("type") != transport.RESET_ACK:
            raise RuntimeError(f"OpenWAM server reset returned unexpected response: {ack}")

    def _bridge_eef20d(self, obs: dict, arm20: np.ndarray, base5: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert a 20-D absolute EEF action (+ optional 5-D RoboCasa-native base command) to the
        env's 12-D OSC action.

        The arm needs the current proprio (to form the OSC delta) and the env's OSC scaling. The
        base command (mobile) is passed through RAW: ``base5`` = [x/y/yaw vel, torso, control_mode];
        the first 4 fill ``base_motion``, the 5th is ``control_mode`` (gym thresholds at 0.5).
        Raises if the scales weren't configured — emitting an unscaled action would drive
        wrong-magnitude motions.
        """
        if self._osc_pos_scale is None or self._osc_rot_scale is None:
            raise ValueError(
                "server returned a 20-D EEF action but osc_pos_scale/osc_rot_scale are unset; "
                "set them from the eval env's OSC_POSE controller config to enable the 20-D->12-D bridge."
            )
        arm20 = np.asarray(arm20, np.float32).reshape(-1)
        base5 = None if base5 is None else np.asarray(base5, np.float32).reshape(-1)
        control_mode = float(base5[4]) if base5 is not None else -1.0
        # The OSC delta's reference frame depends on the arm goal-update mode robosuite picks from
        # control_mode (composite_controller: control_mode>0 -> "desired", else "achieved"):
        #   achieved (control_mode<0.5, OR the first step with no prior target): goal = current_eef +
        #     delta -> reference = the CURRENT observed eef -> delta = target - current.
        #   desired  (control_mode>=0.5, "base mode": the base is driving): goal = last_desired_goal +
        #     delta, and that last goal is the PREVIOUS step's target -> reference = previous target ->
        #     delta = target - previous_target. Without this, a desired-mode step applies an
        #     achieved-relative delta on top of the desired goal and mis-places the arm exactly while
        #     the base moves (control_mode=+1 is ~7% of steps globally, up to 91% on NavigateKitchen,
        #     and co-occurs with base motion 92-99% of the time).
        if control_mode >= 0.5 and self._prev_target_pose is not None:
            ref_pos = self._prev_target_pose[0:3]
            ref_rot6d = self._prev_target_pose[3:9]
        else:
            ref_pos = np.asarray(obs["state.end_effector_position_relative"], np.float32).reshape(-1)
            ref_rot6d = quat_xyzw_to_rot6d(
                np.asarray(obs["state.end_effector_rotation_relative"], np.float32).reshape(-1)
            )
        base_kw = {} if base5 is None else dict(base_motion=base5[0:4], control_mode=control_mode)
        twelve = eef20d_to_robocasa12d(
            arm20,
            proprio_eef_pos=ref_pos,
            proprio_eef_rot6d=ref_rot6d,
            pos_scale=self._osc_pos_scale,
            rot_scale=self._osc_rot_scale,
            **base_kw,
        )
        # Remember this step's absolute arm target (pos3 + rot6d6) as next step's desired reference.
        self._prev_target_pose = arm20[0:9].copy()
        return twelve

    def _base_velocity(self, obs: dict) -> np.ndarray:
        """Current body-frame base velocity [vx, vy, vyaw] via stateful finite-diff of the world base
        pose (base_position + base_rotation), the SAME derivation the dataloader uses at train time.
        The first step of an episode (no previous pose) is zero. Updates the tracked previous pose."""
        missing = [k for k in BASE_POSE_KEYS if k not in obs]
        if missing:
            raise KeyError(f"base_proprio_velocity needs obs base-pose key(s): {missing}")
        cur = np.concatenate([
            np.asarray(obs["state.base_position"], np.float32).reshape(-1)[:3],
            np.asarray(obs["state.base_rotation"], np.float32).reshape(-1)[:4],
        ])
        vel = np.zeros(BASE_VEL_DIM, np.float32) if self._prev_base_pose is None else base_velocity_body(
            self._prev_base_pose, cur
        )
        self._prev_base_pose = cur
        return vel

    def act(self, obs: dict, prompt: str) -> dict:
        base_vel = self._base_velocity(obs) if self._base_proprio_velocity else None
        payload = build_obs_payload(
            obs,
            head_camera_key=self._head_camera_key,
            left_wrist_camera_key=self._left_wrist_camera_key,
            right_wrist_camera_key=self._right_wrist_camera_key,
            image_transform=self._image_transform,
            state_keys=self._state_keys,
            prompt=prompt,
            base_vel=base_vel,
        )
        if self._state_dim is not None and len(payload["state"]) != self._state_dim:
            raise ValueError(f"RoboCasa365 state dim {len(payload['state'])} != expected {self._state_dim}")
        response = self._client.predict(payload)
        flat = np.asarray(response["action"], dtype=np.float32).reshape(-1)
        # Server returns the un-unified RoboCasa action; the client bridges the arm and passes the
        # base command through:
        #   25-D = [arm20 absolute EEF, base5 (x/y/yaw vel, torso, control_mode)] — mobile ckpt.
        #   20-D = arm-only (fixed-base ckpt): base is zero-filled by the bridge.
        #   12-D = a raw env action, passed through unchanged.
        if flat.shape[0] == 25:
            flat = self._bridge_eef20d(obs, flat[:20], flat[20:25])
        elif flat.shape[0] == 20:
            flat = self._bridge_eef20d(obs, flat)
        if flat.shape[0] != self._action_dim:
            raise ValueError(f"OpenWAM returned action dim {flat.shape[0]}, expected {self._action_dim}")
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
                save_montage=(self._step == 0),
                expected_state_dim=self._state_dim,
            )
        self._step += 1
        return slice_action(flat)
