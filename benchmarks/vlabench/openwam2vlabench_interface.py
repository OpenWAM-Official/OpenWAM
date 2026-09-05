"""VLABench eval adapter for the OpenWAM Policy Server.

The adapter lives in the benchmark client environment and talks to an
already-running OpenWAM WebSocket server. The model, checkpoint, preprocessing
and action denormalization all stay server-side. It deliberately imports
NOTHING from VLABench — the evaluator only duck-types ``name``,
``control_mode``, ``reset()`` and ``predict()`` — so it works against any
VLABench checkout without patching that repo.

EEF10 contract
--------------
The checkpoint is trained on raw 10-D single-arm EEF ``[xyz3, rot6d6, grip1]``
scattered into the unified 80-D left-arm slots (``unify_action_map: ["0-9"]``,
see ``configs/dataloader/vlabench.yaml``). The deploy server gathers the model's
unified output back to raw EEF10 and unnormalizes it, so this client speaks
plain EEF10 in both directions.

Frames
------
Training data is in the ROBOT BASE frame (VLABench's LeRobot converter
subtracts the base position). VLABench's evaluator, however, runs IK on an
absolute WORLD-frame target. So the client subtracts the base offset on the way
in and adds it back on the way out, reading the live base position from
``obs["robot_frame"]`` (injected by the evaluator from
``env.get_robot_frame_position()``) rather than assuming the converter's
``[0, -0.4, 0.78]`` fallback.

Gripper
-------
State and action carry OPPOSITE gripper polarity in the VLABench dataset
(measured correlation -0.93), because the Franka's ``get_ee_open_state`` returns
True when CLOSED — an acknowledged upstream bug. The proprio is forwarded
verbatim (the eval env reads the same buggy accessor, so it cancels), while the
returned action is thresholded with 1 = OPEN. See
``openwam/dataloader/vlabench.py`` for the full derivation.

Chunking
--------
None here: the server buffers action chunks internally and the wire protocol is
one obs in, one action out. Unlike the openpi / lingbot_va adapters, this
client keeps no ``action_plan`` deque.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from benchmarks.utils import (  # noqa: E402
    WSPolicyClient,
    client,
    eef10_to_vlabench_ee,
    resize_for_lshape_slot,
    transport,
    vlabench_obs_to_eef10,
)
from benchmarks.utils.action_conversion import (  # noqa: E402
    VLABENCH_EEF10_DIM,
    VLABENCH_GRIPPER_OPEN_THRESHOLD,
    VLABENCH_GRIPPER_OPEN_WIDTH,
    VLABENCH_ROBOT_BASE_DEFAULT,
)

# obs["rgb"] camera order, verified against the live MuJoCo model (names read
# via mj_id2name) and VLABench's LeRobot converter:
#   rgb[0] "right"                    -> dataset second_image -> right wrist slot
#   rgb[1] "left"                     -> unused
#   rgb[2] "forward"                  -> dataset image        -> head slot
#   rgb[3] "franka/Franka_wrist_cam"  -> dataset wrist_image   -> left wrist slot
HEAD_CAMERA_INDEX = 2
LEFT_WRIST_CAMERA_INDEX = 3
RIGHT_WRIST_CAMERA_INDEX = 0


class OpenWAMVLABenchPolicy:
    """Adapter from VLABench's Policy duck-type to an OpenWAM policy server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8848,
        request_timeout: int = 300,
        head_camera_index: int = HEAD_CAMERA_INDEX,
        left_wrist_camera_index: int | None = LEFT_WRIST_CAMERA_INDEX,
        right_wrist_camera_index: int | None = RIGHT_WRIST_CAMERA_INDEX,
        send_state: bool = True,
        state_dim: int | None = VLABENCH_EEF10_DIM,
        action_dim: int = VLABENCH_EEF10_DIM,
        gripper_open_threshold: float = VLABENCH_GRIPPER_OPEN_THRESHOLD,
        gripper_open_width: float = VLABENCH_GRIPPER_OPEN_WIDTH,
        robot_base_fallback: tuple = VLABENCH_ROBOT_BASE_DEFAULT,
        debug: bool = False,
        debug_dir: str = "./debug_vlabench",
    ) -> None:
        self._ws_url = f"ws://{host}:{port}"
        self._client = WSPolicyClient(self._ws_url, timeout=request_timeout)
        self._head_idx = int(head_camera_index)
        self._left_idx = None if left_wrist_camera_index is None else int(left_wrist_camera_index)
        self._right_idx = None if right_wrist_camera_index is None else int(right_wrist_camera_index)
        self._send_state = bool(send_state)
        self._state_dim = state_dim
        self._action_dim = int(action_dim)
        self._gripper_open_threshold = float(gripper_open_threshold)
        self._gripper_open_width = float(gripper_open_width)
        self._robot_base_fallback = np.asarray(robot_base_fallback, np.float64).reshape(-1)
        if self._robot_base_fallback.shape[0] != 3:
            raise ValueError(f"robot_base_fallback must be 3-D, got {self._robot_base_fallback.shape[0]}")
        self._debug = bool(debug)
        self._debug_dir = Path(debug_dir)
        self._episode = -1
        self._step = 0
        self._warned_missing_robot_frame = False
        if self._debug:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

        pong = self._client.ping()
        if pong.get("type") != transport.PONG:
            raise RuntimeError(f"OpenWAM server ping returned unexpected response: {pong}")

        print(
            f"[OpenWAMVLABenchPolicy] server={self._ws_url} "
            f"cams=(head={self._head_idx}, lwrist={self._left_idx}, rwrist={self._right_idx}) "
            f"send_state={self._send_state} action_dim={self._action_dim} "
            f"gripper(open>={self._gripper_open_threshold}, width={self._gripper_open_width})"
        )

    # ---- VLABench Policy duck-type ----
    @property
    def name(self) -> str:
        return "openwam"

    @property
    def control_mode(self) -> str:
        return "ee"

    def reset(self) -> None:
        self._episode += 1
        self._step = 0
        ack = self._client.reset()
        if ack.get("type") != transport.RESET_ACK:
            raise RuntimeError(f"OpenWAM server reset returned unexpected response: {ack}")

    def close(self) -> None:
        self._client.close()

    def predict(self, obs: dict, **kwargs):
        """One control step: VLABench obs -> ``(pos, euler, gripper_state)``."""
        robot_base = self._robot_base(obs)
        payload = client.build_payload(
            head=client.encode_numpy_b64(
                resize_for_lshape_slot(self._image(obs, self._head_idx, "head"), "head_camera")
            ),
            left_wrist=self._maybe_encode(obs, self._left_idx, "left_wrist", "left_wrist_camera"),
            right_wrist=self._maybe_encode(obs, self._right_idx, "right_wrist", "right_wrist_camera"),
            prompt=self._prompt(obs),
            state=self._state(obs, robot_base),
        )
        response = self._client.predict(payload)
        action = np.asarray(response["action"], dtype=np.float32).reshape(-1)
        if action.shape[0] != self._action_dim:
            raise ValueError(
                f"OpenWAM returned action dim {action.shape[0]}, expected {self._action_dim} "
                "(raw EEF10 for a unify_action_map: ['0-9'] checkpoint)"
            )
        target_pos, target_euler, gripper_state = eef10_to_vlabench_ee(
            action,
            robot_base,
            gripper_open_threshold=self._gripper_open_threshold,
            gripper_open_width=self._gripper_open_width,
        )
        self._maybe_debug(payload, action, target_pos, target_euler, gripper_state, robot_base)
        self._step += 1
        return target_pos, target_euler, gripper_state

    # ---- obs translation ----
    def _robot_base(self, obs: dict) -> np.ndarray:
        """Live robot base position, falling back to the converter's constant."""
        base = obs.get("robot_frame")
        if base is None:
            if not self._warned_missing_robot_frame:
                print(
                    "[OpenWAMVLABenchPolicy] WARNING: obs has no 'robot_frame'; falling back to "
                    f"{tuple(self._robot_base_fallback)}. Episodes whose config sets an explicit "
                    "robot.position will be mis-framed."
                )
                self._warned_missing_robot_frame = True
            return self._robot_base_fallback
        arr = np.asarray(base, np.float64).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"obs['robot_frame'] must be 3-D, got {arr.shape[0]}")
        return arr

    @staticmethod
    def _prompt(obs: dict) -> str:
        """The instruction, forwarded verbatim.

        Training prompts came from the same ``instruction`` strings that
        ``env.task.get_instruction()`` returns (VLABench's converter stored them
        straight into ``meta/tasks.parquet``), so no template wrapping applies.
        """
        prompt = obs.get("instruction")
        if not prompt:
            # The evaluator injects this from task.get_instruction(), which
            # returns None when a task's instruction list is empty. Fail loudly
            # rather than send the model an empty prompt it never saw in training.
            raise KeyError(
                f"VLABench obs has no usable 'instruction' (got {prompt!r}); "
                "task.get_instruction() returns None for a task with no instructions"
            )
        return str(prompt)

    def _image(self, obs: dict, index: int, label: str) -> np.ndarray:
        rgb = obs.get("rgb")
        if rgb is None:
            raise KeyError("VLABench obs missing 'rgb'")
        frames = np.asarray(rgb)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"obs['rgb'] must be (ncam, H, W, 3), got {frames.shape}")
        if not (0 <= index < frames.shape[0]):
            raise IndexError(f"{label} camera index {index} out of range for obs['rgb'] with {frames.shape[0]} cameras")
        return np.ascontiguousarray(frames[index].astype(np.uint8, copy=False))

    def _maybe_encode(self, obs: dict, index: int | None, label: str, slot: str) -> str | None:
        if index is None:
            return None
        image = resize_for_lshape_slot(self._image(obs, index, label), slot)
        return client.encode_numpy_b64(image)

    def _state(self, obs: dict, robot_base: np.ndarray) -> list | None:
        if not self._send_state:
            return None
        ee_state = obs.get("ee_state")
        if ee_state is None:
            raise KeyError("VLABench obs missing 'ee_state'")
        state = vlabench_obs_to_eef10(ee_state, robot_base).tolist()
        if self._state_dim is not None and len(state) != self._state_dim:
            raise ValueError(f"VLABench state dim {len(state)} != expected {self._state_dim}")
        return state

    def _maybe_debug(self, payload, action, pos, euler, gripper, robot_base) -> None:
        if not self._debug:
            return
        step_dir = self._debug_dir / f"episode_{self._episode:03d}" / f"step_{self._step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "prompt": payload["prompt"],
            "state": payload.get("state"),
            "robot_base": robot_base.tolist(),
            "raw_eef10": np.asarray(action, np.float32).reshape(-1).tolist(),
            "target_pos": np.asarray(pos, np.float64).tolist(),
            "target_euler": np.asarray(euler, np.float64).tolist(),
            "gripper_state": np.asarray(gripper, np.float64).tolist(),
        }
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


__all__ = ["OpenWAMVLABenchPolicy"]
