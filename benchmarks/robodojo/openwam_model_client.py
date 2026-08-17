"""Synchronous RoboDojo model-client surface backed by OpenWAM JSON WebSockets.

The adapter is intentionally Isaac-free at import time. Pose conversion uses
the built-in dual-X5 base constants; a live runner still passes EvalEnv for
observations and control.
"""

from __future__ import annotations

import base64
import copy
import io
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from PIL import Image

from benchmarks.utils import WSPolicyClient, build_payload, encode_numpy_b64
from openwam.dataloader.transforms.multiview import format_prompt_for_inference
from benchmarks.robodojo.contract import (
    EEF20_DIM,
    ROBODOJO_EMBODIMENT,
    arx_x5_calibration,
    validate_calibration,
)
from benchmarks.robodojo.frames import (
    arms_to_eef20,
    eef20_to_arms,
    env_relative_world_to_robot_base,
    robot_base_to_env_relative_world,
)

_CAMERA_TO_PAYLOAD = {
    "cam_head": "head_camera",
    "cam_left_wrist": "left_wrist_camera",
    "cam_right_wrist": "right_wrist_camera",
}
_STATE_FIELDS = (
    "left_ee_pose",
    "left_ee_joint_state",
    "right_ee_pose",
    "right_ee_joint_state",
)
_EXPECTED_ARM_DIMS = [6, 6]
_EXPECTED_EE_DIMS = [1, 1]
_QUATERNION_ATOL = 1e-6
# Official HDF5 / live obs occasionally store a closed gripper as ~-3e-17.
_GRIPPER_ATOL = 1e-8


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    if array.shape != shape:
        raise ValueError(f"{name} must have exact shape {shape}, got {array.shape}")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return array.copy()


def _validate_pose(value: Any, name: str) -> np.ndarray:
    pose = _finite_vector(value, (7,), name)
    norm = float(np.linalg.norm(pose[3:7]))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=_QUATERNION_ATOL):
        raise ValueError(
            f"{name} must contain a unit quaternion in wxyz order; norm is {norm:.8g}"
        )
    pose[3:7] /= norm
    return pose


def _validate_gripper(value: Any, name: str) -> np.ndarray:
    gripper = _finite_vector(value, (1,), name)
    if np.any((gripper < -_GRIPPER_ATOL) | (gripper > 1.0 + _GRIPPER_ATOL)):
        raise ValueError(f"{name} gripper value must be within [0, 1]")
    return np.clip(gripper, 0.0, 1.0)


def _decode_instruction(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError("instruction must be a scalar string or bytes")
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("instruction bytes must be valid UTF-8") from error
    if not isinstance(value, str):
        raise ValueError("instruction must be a scalar string or bytes")
    instruction = value.strip()
    if not instruction:
        raise ValueError("instruction must be a non-empty scalar string")
    return instruction


def _jpeg_bytes(value: Any, camera_name: str) -> bytes:
    if isinstance(value, np.bytes_):
        encoded = bytes(value)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
    elif isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 1:
        encoded = value.tobytes()
    else:
        raise ValueError(
            f"vision/{camera_name}/color encoded input must be bytes or "
            "one-dimensional uint8"
        )
    if not encoded:
        raise ValueError(f"vision/{camera_name}/color encoded JPEG must not be empty")
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            if image.format != "JPEG":
                raise ValueError(
                    f"vision/{camera_name}/color encoded input must be JPEG"
                )
            image.verify()
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            f"vision/{camera_name}/color could not be decoded as JPEG"
        ) from error
    return encoded


def _encode_camera(value: Any, camera_name: str) -> tuple[str, list[int]]:
    if isinstance(value, np.ndarray) and value.ndim != 1:
        if value.dtype != np.uint8:
            raise ValueError(f"vision/{camera_name}/color RGB array must use uint8")
        if value.ndim != 3 or value.shape[2] != 3 or 0 in value.shape:
            raise ValueError(
                f"vision/{camera_name}/color RGB array must have shape (H, W, 3), "
                f"got {value.shape}"
            )
        return encode_numpy_b64(value), list(value.shape)
    encoded = _jpeg_bytes(value, camera_name)
    return base64.b64encode(encoded).decode("ascii"), [len(encoded)]


def _validate_robot_dimensions(robot_action_dim_info: Any) -> None:
    if not isinstance(robot_action_dim_info, Mapping):
        raise ValueError("robot_action_dim_info must be a mapping")
    arm_dims = robot_action_dim_info.get("arm_dim")
    ee_dims = robot_action_dim_info.get("ee_dim")
    if list(arm_dims or []) != _EXPECTED_ARM_DIMS:
        raise ValueError(
            "RoboDojo OpenWAM evaluation requires arm_dim [6, 6], "
            f"got {arm_dims!r}"
        )
    if list(ee_dims or []) != _EXPECTED_EE_DIMS:
        raise ValueError(
            "RoboDojo OpenWAM evaluation requires ee_dim [1, 1], "
            f"got {ee_dims!r}"
        )


def _validate_live_target_robots(task_env: Any) -> None:
    try:
        robots = task_env.robot_manager.robot_list
    except AttributeError as error:
        raise ValueError(
            "live RoboDojo EvalEnv must expose robot_manager.robot_list"
        ) from error
    targets = [robot for robot in robots if getattr(robot, "type", None) == "target"]
    valid = (
        len(targets) == 2
        and all(getattr(robot, "robot_type", None) == "arm" for robot in targets)
        and all(getattr(robot, "robot_name", None) == "x5" for robot in targets)
        and {getattr(robot, "arm_name", None) for robot in targets}
        == {"left_arm", "right_arm"}
    )
    if not valid:
        raise ValueError(
            "RoboDojo OpenWAM evaluation requires exactly two target X5 arms "
            "named left_arm and right_arm"
        )


class OpenWAMRoboDojoModelClient:
    """Match XPolicyLab's synchronous model-client API for one ``arx_x5`` env."""

    def __init__(
        self,
        *,
        task_env: Any | None,
        host: str,
        port: int,
        timeout: float = 300.0,
        num_envs: int = 1,
        env_config: str = ROBODOJO_EMBODIMENT,
        robot_action_dim_info: Mapping[str, Any] | None = None,
        transport_factory: Callable[..., Any] = WSPolicyClient,
        frame_provider: Callable[[], Mapping[str, Any]] | None = None,
        debug: bool = False,
    ):
        if (
            isinstance(num_envs, bool)
            or not isinstance(num_envs, (int, np.integer))
            or int(num_envs) != 1
        ):
            raise ValueError(
                "OpenWAM's RoboDojo adapter supports exactly one environment "
                "and requires integer num_envs == 1, "
                f"got {num_envs!r}"
            )
        if env_config != ROBODOJO_EMBODIMENT:
            raise ValueError(
                f"OpenWAM's RoboDojo adapter supports only arx_x5, got {env_config!r}"
            )
        if robot_action_dim_info is None and task_env is not None:
            robot_action_dim_info = getattr(task_env, "robot_action_dim_info", None)
        _validate_robot_dimensions(robot_action_dim_info)

        if task_env is not None and hasattr(task_env, "num_envs"):
            if (
                isinstance(task_env.num_envs, bool)
                or not isinstance(task_env.num_envs, (int, np.integer))
                or int(task_env.num_envs) != 1
            ):
                raise ValueError(
                    "live RoboDojo EvalEnv must contain integer num_envs == 1, "
                    f"got {task_env.num_envs!r}"
                )
        if frame_provider is None:
            if task_env is None:
                raise ValueError(
                    "task_env is required for live calibration; frame_provider is "
                    "reserved for fake/debug runs"
                )
            _validate_live_target_robots(task_env)

        if (
            not isinstance(host, str)
            or not host.strip()
            or host != host.strip()
            or "://" in host
            or "/" in host
            or any(character.isspace() for character in host)
        ):
            raise ValueError("host must be a non-empty hostname or IP address")
        try:
            parsed_port = int(port)
        except (TypeError, ValueError) as error:
            raise ValueError(f"port must be in [1, 65535], got {port!r}") from error
        if isinstance(port, bool) or not 1 <= parsed_port <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {port!r}")
        try:
            parsed_timeout = float(timeout)
        except (TypeError, ValueError) as error:
            raise ValueError(f"timeout must be positive, got {timeout!r}") from error
        if not math.isfinite(parsed_timeout) or parsed_timeout <= 0.0:
            raise ValueError(f"timeout must be positive, got {timeout!r}")

        self.task_env = task_env
        self.frame_provider = frame_provider
        self.debug = bool(debug)
        self.debug_records: list[dict[str, Any]] = []
        self._observation: dict[str, Any] | None = None
        self._closed = False
        # Ping proves liveness, not episode synchronization. The native/debug
        # env must obtain reset_ack before sending the first observation.
        self._poisoned = True

        ws_url = f"ws://{host.strip()}:{parsed_port}"
        self._transport = transport_factory(
            ws_url,
            timeout=parsed_timeout,
            open_timeout=min(10.0, parsed_timeout),
        )
        try:
            pong = self._transport.ping()
            if not isinstance(pong, Mapping) or pong.get("type") != "pong":
                raise ConnectionError(
                    "OpenWAM liveness probe expected {'type': 'pong'}, "
                    f"got {pong!r}"
                )
        except BaseException:
            self._transport.close()
            self._closed = True
            raise

    def _live_calibration(self) -> dict[str, Any]:
        if self.frame_provider is not None:
            return validate_calibration(self.frame_provider())
        return arx_x5_calibration()

    @staticmethod
    def _validated_observation(obs: Any) -> dict[str, Any]:
        if isinstance(obs, list):
            if len(obs) != 1:
                raise ValueError(
                    "single-env RoboDojo observation list must contain exactly one item"
                )
            obs = obs[0]
        if not isinstance(obs, Mapping):
            raise TypeError("RoboDojo observation must be a mapping or one-item list")

        vision = obs.get("vision")
        if not isinstance(vision, Mapping):
            raise ValueError("RoboDojo observation vision must be a mapping")
        images: dict[str, str] = {}
        camera_shapes: dict[str, list[int]] = {}
        for camera_name, payload_name in _CAMERA_TO_PAYLOAD.items():
            camera = vision.get(camera_name)
            if not isinstance(camera, Mapping):
                raise KeyError(f"missing required camera vision/{camera_name}")
            if "color" not in camera:
                raise KeyError(
                    f"missing required camera field vision/{camera_name}/color"
                )
            images[payload_name], camera_shapes[camera_name] = _encode_camera(
                camera["color"], camera_name
            )

        instruction = _decode_instruction(obs.get("instruction"))
        state = obs.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("RoboDojo observation state must be a mapping")
        missing = [field for field in _STATE_FIELDS if field not in state]
        if missing:
            raise KeyError(
                "RoboDojo observation state is missing required field(s): "
                + ", ".join(missing)
            )
        env_idx = obs.get("env_idx")
        if isinstance(env_idx, bool) or not isinstance(env_idx, (int, np.integer)):
            raise ValueError(f"env_idx must be integer 0, got {env_idx!r}")
        if int(env_idx) != 0:
            raise ValueError(
                f"single-env RoboDojo observation requires env_idx == 0, got {env_idx!r}"
            )

        return {
            "images": images,
            "camera_shapes": camera_shapes,
            "instruction": instruction,
            "left_ee_pose": _validate_pose(
                state["left_ee_pose"], "left_ee_pose"
            ),
            "left_ee_joint_state": _validate_gripper(
                state["left_ee_joint_state"], "left_ee_joint_state"
            ),
            "right_ee_pose": _validate_pose(
                state["right_ee_pose"], "right_ee_pose"
            ),
            "right_ee_joint_state": _validate_gripper(
                state["right_ee_joint_state"], "right_ee_joint_state"
            ),
        }

    @staticmethod
    def _state_in_base(
        observation: Mapping[str, Any],
        calibration: Mapping[str, Any],
    ) -> np.ndarray:
        left = calibration["arms"]["left"]
        right = calibration["arms"]["right"]
        left_pose = env_relative_world_to_robot_base(
            observation["left_ee_pose"],
            left["base_pos_relative_to_env_origin"],
            left["base_quat_wxyz"],
        )
        right_pose = env_relative_world_to_robot_base(
            observation["right_ee_pose"],
            right["base_pos_relative_to_env_origin"],
            right["base_quat_wxyz"],
        )
        state = arms_to_eef20(
            left_pose,
            observation["left_ee_joint_state"],
            right_pose,
            observation["right_ee_joint_state"],
        )
        if state.shape != (EEF20_DIM,):
            raise RuntimeError(
                f"internal RoboDojo state conversion produced {state.shape}, expected (20,)"
            )
        return state

    @staticmethod
    def _validated_server_action(response: Any) -> np.ndarray:
        if not isinstance(response, Mapping) or response.get("type") != "action":
            response_type = (
                response.get("type") if isinstance(response, Mapping) else None
            )
            raise ValueError(
                "OpenWAM response type must be 'action', "
                f"got {response_type!r}"
            )
        action = np.asarray(response.get("action"))
        if action.dtype.kind not in "fiu":
            raise ValueError("OpenWAM action must contain real numeric values")
        if action.shape != (EEF20_DIM,):
            raise ValueError(
                "OpenWAM action must be a flat array with exact shape (20,), "
                f"got {action.shape}"
            )
        action = action.astype(np.float64, copy=False)
        if not np.all(np.isfinite(action)):
            raise ValueError("OpenWAM action must contain only finite values")
        # This performs the binding non-degenerate rot6d validation.
        eef20_to_arms(action)
        return action.copy()

    @staticmethod
    def _native_action(
        action: np.ndarray,
        calibration: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        left_pose, left_gripper, right_pose, right_gripper = eef20_to_arms(action)
        left = calibration["arms"]["left"]
        right = calibration["arms"]["right"]
        return {
            "left_ee_pose": robot_base_to_env_relative_world(
                left_pose,
                left["base_pos_relative_to_env_origin"],
                left["base_quat_wxyz"],
            ),
            "left_ee_joint_state": np.clip(left_gripper, 0.0, 1.0),
            "right_ee_pose": robot_base_to_env_relative_world(
                right_pose,
                right["base_pos_relative_to_env_origin"],
                right["base_quat_wxyz"],
            ),
            "right_ee_joint_state": np.clip(right_gripper, 0.0, 1.0),
        }

    def _reset(self) -> None:
        # Reset is the only recovery path. Fail closed until its acknowledgement
        # proves that both sides have discarded any uncertain action state.
        self._observation = None
        self._poisoned = True
        response = self._transport.reset()
        if not isinstance(response, Mapping) or response.get("type") != "reset_ack":
            raise RuntimeError(
                "OpenWAM reset expected {'type': 'reset_ack'}, "
                f"got {response!r}"
            )
        self._poisoned = False

    def _update_obs(self, obs: Any) -> None:
        if self._observation is not None:
            raise RuntimeError(
                "update_obs received a duplicate observation before the previous "
                "one was consumed by get_action"
            )
        self._observation = self._validated_observation(obs)

    def _get_action(self) -> list[dict[str, np.ndarray]]:
        if self._observation is None:
            raise RuntimeError("get_action requires a preceding update_obs call")
        observation = self._observation
        self._observation = None
        calibration = self._live_calibration()
        raw_state = self._state_in_base(observation, calibration)
        payload = build_payload(
            head=observation["images"]["head_camera"],
            left_wrist=observation["images"]["left_wrist_camera"],
            right_wrist=observation["images"]["right_wrist_camera"],
            prompt=format_prompt_for_inference(observation["instruction"]),
            state=raw_state.tolist(),
        )
        try:
            response = self._transport.predict_once(payload)
            action = self._validated_server_action(response)
            native = self._native_action(action, calibration)
        except Exception:
            self._observation = None
            self._poisoned = True
            raise

        if self.debug:
            self.debug_records.append(
                {
                    "observation_eef20": raw_state.copy(),
                    "server_action_eef20": action.copy(),
                    "base_transforms": {
                        side: {
                            "base_pos_relative_to_env_origin": np.asarray(
                                calibration["arms"][side][
                                    "base_pos_relative_to_env_origin"
                                ],
                                dtype=np.float64,
                            ).copy(),
                            "base_quat_wxyz": np.asarray(
                                calibration["arms"][side]["base_quat_wxyz"],
                                dtype=np.float64,
                            ).copy(),
                        }
                        for side in ("left", "right")
                    },
                    "converted_action": {
                        key: value.copy() for key, value in native.items()
                    },
                    "response_step": response.get("step"),
                    "latency_ms": response.get("latency_ms"),
                    "camera_shapes": copy.deepcopy(observation["camera_shapes"]),
                }
            )
        return [native]

    def call(
        self,
        func_name: str | None = None,
        obs: Any = None,
        **kwargs: Any,
    ) -> Any:
        if self._closed:
            raise RuntimeError("OpenWAMRoboDojoModelClient is closed")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                "OpenWAMRoboDojoModelClient.call() accepts only func_name and "
                f"obs; unexpected keyword(s): {unexpected}"
            )
        if func_name != "reset" and self._poisoned:
            raise RuntimeError(
                "OpenWAMRoboDojoModelClient is poisoned after an uncertain "
                "action/reset; a successful reset is required"
            )
        if func_name in {"update_obs_batch", "get_action_batch"}:
            raise NotImplementedError(
                f"{func_name} is a batch call and is unavailable in this single-env integration"
            )
        if func_name == "reset":
            if obs is not None:
                raise TypeError("reset takes no obs payload")
            return self._reset()
        if func_name == "update_obs":
            if obs is None:
                raise TypeError("update_obs requires an obs payload")
            return self._update_obs(obs)
        if func_name == "get_action":
            if obs is not None:
                raise TypeError("get_action takes no obs payload")
            return self._get_action()
        raise NotImplementedError(f"unknown RoboDojo model call: {func_name!r}")

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._transport.close()
        finally:
            self._closed = True
            self._observation = None

    def __enter__(self) -> OpenWAMRoboDojoModelClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["OpenWAMRoboDojoModelClient"]
