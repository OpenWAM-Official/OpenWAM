"""RoboDojo dataset layout, frame contracts, and sim calibration schema.

Generic pose / EEF20 math lives in ``openwam.dataloader.utils.poses``.
This module only holds the numbers and file-layout rules the reader needs.

The Isaac eval runtime (XPolicyLab) keeps a pinned copy of this contract.
The two must stay aligned on ``ROBODOJO_CONTRACT_ID``. Production modules
on either side must not import the other.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROBODOJO_CONTRACT_ID = "robodojo-eef20-v1"
ROBODOJO_REAL_CONTRACT_ID = "robodojo-real-native-eef20-v1"
ROBODOJO_EMBODIMENT = "arx_x5"
ROBODOJO_SIM_VARIANT = "sim"
ROBODOJO_REAL_VARIANT = "real"
ROBODOJO_DATASET_VARIANTS = (ROBODOJO_SIM_VARIANT, ROBODOJO_REAL_VARIANT)
ROBODOJO_REAL_EMBODIMENTS = ("arx_x5", "piper", "piper_x")
CALIBRATION_SCHEMA_VERSION = 1
REAL_FRAME_SCHEMA_VERSION = 1
ENDPOINT_LINK_NAME = "link6"
ENDPOINT_POSE_FRAME_CONTRACT = "rigid_terminal_arm_frame_independent_of_gripper_motion"
REAL_ENDPOINT_NAME = "recorded_ee_pose"
ROBODOJO_SIM_SOURCE_FRAME = "env_origin_relative_position_world_orientation_wxyz"
ROBODOJO_REAL_SOURCE_FRAME = "per_arm_robot_base_position_and_orientation_wxyz"
ROBODOJO_TARGET_FRAME = "per_arm_robot_base"
ARM_NAMES = ("left", "right")
# Raw EEF20 gripper: 0 = closed, 1 = open. Matches the pretrain mixture.
GRIPPER_CONVENTION = "zero_closed_one_open"
# Small negative sensor excursions within the configured tolerance are clipped
# to zero; larger excursions remain schema errors.
ROBODOJO_REAL_GRIPPER_SENSOR_ATOL = 0.05

# Source of truth: RoboDojo ``env_cfg/robot/dual_x5.yml``.
# The first listed robot is ``left_arm``; the second is ``right_arm``.
# Positions are relative to the Isaac environment origin. Rotations are wxyz.
DUAL_X5_LEFT_BASE_POS = (-0.3, -0.45, 0.765)
DUAL_X5_LEFT_BASE_QUAT_WXYZ = (0.707, 0.0, 0.0, 0.707)
DUAL_X5_RIGHT_BASE_POS = (0.3, -0.45, 0.765)
DUAL_X5_RIGHT_BASE_QUAT_WXYZ = (0.707, 0.0, 0.0, 0.707)
EEF20_DIM = 20
EEF20_LAYOUT = (
    "left.xyz",
    "left.rot6d",
    "left.gripper",
    "right.xyz",
    "right.rot6d",
    "right.gripper",
)
FORMAL_EPISODE_GLOB = "episode_*.hdf5"

_ROOT_KEYS = {"schema_version", "embodiment", "endpoint", "arms"}
_ENDPOINT_KEYS = {"link_name", "pose_frame_contract"}
_ARM_CALIBRATION_KEYS = {
    "base_pos_relative_to_env_origin",
    "base_quat_wxyz",
}
_QUATERNION_ATOL = 1e-6


def validate_dataset_variant(variant: str) -> None:
    """Validate the explicit simulation/real dataset release selector."""
    if variant not in ROBODOJO_DATASET_VARIANTS:
        raise ValueError(f"RoboDojo variant must be one of {list(ROBODOJO_DATASET_VARIANTS)}, got {variant!r}")


def validate_embodiment(
    embodiment: str,
    *,
    variant: str = ROBODOJO_SIM_VARIANT,
) -> None:
    """Validate an embodiment against the selected RoboDojo release."""
    validate_dataset_variant(variant)
    supported = (ROBODOJO_EMBODIMENT,) if variant == ROBODOJO_SIM_VARIANT else ROBODOJO_REAL_EMBODIMENTS
    if embodiment not in supported:
        if variant == ROBODOJO_SIM_VARIANT:
            raise ValueError(
                f"RoboDojo sim's only supported OpenWAM embodiment is '{ROBODOJO_EMBODIMENT}', got {embodiment!r}"
            )
        raise ValueError(
            f"RoboDojo {variant}'s supported OpenWAM embodiments are {list(supported)}, got {embodiment!r}"
        )


def robodojo_real_frame_contract(embodiment: str) -> dict[str, Any]:
    """Return the native, identity-pose contract for one real embodiment."""
    validate_embodiment(embodiment, variant=ROBODOJO_REAL_VARIANT)
    return {
        "schema_version": REAL_FRAME_SCHEMA_VERSION,
        "variant": ROBODOJO_REAL_VARIANT,
        "embodiment": embodiment,
        "source_frame": ROBODOJO_REAL_SOURCE_FRAME,
        "target_frame": ROBODOJO_TARGET_FRAME,
        "pose_transform": "identity_before_quaternion_to_rot6d",
        "endpoint": REAL_ENDPOINT_NAME,
        "gripper_convention": GRIPPER_CONVENTION,
        "gripper_preprocessing": "clip_sensor_noise_to_[0,1]",
        "gripper_sensor_atol": ROBODOJO_REAL_GRIPPER_SENSOR_ATOL,
        "contract_id": ROBODOJO_REAL_CONTRACT_ID,
    }


def _validate_task_name(task: str) -> None:
    if not isinstance(task, str) or not task or task in {".", ".."} or "/" in task or "\\" in task:
        raise ValueError(f"RoboDojo task must be a non-empty single path component, got {task!r}")


def discover_episodes(
    dataset_root: str | Path,
    task: str,
    *,
    embodiment: str = ROBODOJO_EMBODIMENT,
    variant: str = ROBODOJO_SIM_VARIANT,
) -> list[Path]:
    """Discover only the formal ``<root>/<task>/<embodiment>/data`` layout.

    The historical flat demo layout at ``<root>/arx_x5/data`` is deliberately
    rejected rather than used as a fallback.
    """
    validate_embodiment(embodiment, variant=variant)
    _validate_task_name(task)
    root = Path(dataset_root)
    data_dir = root / task / embodiment / "data"
    if not data_dir.is_dir():
        flat_data_dir = root / embodiment / "data"
        flat_episodes = (
            [path for path in flat_data_dir.glob(FORMAL_EPISODE_GLOB) if path.is_file()]
            if flat_data_dir.is_dir()
            else []
        )
        if flat_episodes:
            raise ValueError(
                "flat RoboDojo demo layout '<dataset_root>/arx_x5/data' is not supported; "
                "use '<dataset_root>/<task>/arx_x5/data'"
            )
        raise FileNotFoundError(f"formal RoboDojo data directory does not exist: {data_dir}")
    episodes = sorted(path for path in data_dir.glob(FORMAL_EPISODE_GLOB) if path.is_file())
    if not episodes:
        raise FileNotFoundError(f"no RoboDojo episodes matching {FORMAL_EPISODE_GLOB!r} in {data_dir}")
    return episodes


def _require_mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping,
    expected: set[str],
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing fields: {', '.join(missing)}")
    if extra:
        details.append(f"unexpected fields: {', '.join(extra)}")
    if details:
        raise ValueError(f"{name} has {'; '.join(details)}")


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    if array.shape != shape:
        raise ValueError(f"{name} must have exact shape {shape}, got {array.shape}")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return array


def _validate_arm_calibration(value: Any, arm_name: str) -> dict[str, list[float]]:
    calibration = _require_mapping(value, f"calibration arms.{arm_name}")
    _require_exact_keys(
        calibration,
        _ARM_CALIBRATION_KEYS,
        f"calibration arms.{arm_name}",
    )
    position = _finite_vector(
        calibration["base_pos_relative_to_env_origin"],
        (3,),
        f"calibration arms.{arm_name}.base_pos_relative_to_env_origin",
    )
    quaternion = _finite_vector(
        calibration["base_quat_wxyz"],
        (4,),
        f"calibration arms.{arm_name}.base_quat_wxyz",
    )
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=_QUATERNION_ATOL):
        raise ValueError(
            f"calibration arms.{arm_name}.base_quat_wxyz must be a unit wxyz quaternion; norm is {norm:.8g}"
        )
    quaternion = quaternion / norm
    return {
        "base_pos_relative_to_env_origin": position.tolist(),
        "base_quat_wxyz": quaternion.tolist(),
    }


def _normalized_wxyz(value: Any, name: str) -> list[float]:
    quaternion = _finite_vector(value, (4,), name)
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        raise ValueError(f"{name} must be a non-zero wxyz quaternion")
    return (quaternion / norm).tolist()


def arx_x5_calibration() -> dict[str, Any]:
    """Return the built-in dual-X5 base transforms used for data conversion."""

    return validate_calibration(
        {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "embodiment": ROBODOJO_EMBODIMENT,
            "endpoint": {
                "link_name": ENDPOINT_LINK_NAME,
                "pose_frame_contract": ENDPOINT_POSE_FRAME_CONTRACT,
            },
            "arms": {
                "left": {
                    "base_pos_relative_to_env_origin": list(DUAL_X5_LEFT_BASE_POS),
                    "base_quat_wxyz": _normalized_wxyz(
                        DUAL_X5_LEFT_BASE_QUAT_WXYZ,
                        "DUAL_X5_LEFT_BASE_QUAT_WXYZ",
                    ),
                },
                "right": {
                    "base_pos_relative_to_env_origin": list(DUAL_X5_RIGHT_BASE_POS),
                    "base_quat_wxyz": _normalized_wxyz(
                        DUAL_X5_RIGHT_BASE_QUAT_WXYZ,
                        "DUAL_X5_RIGHT_BASE_QUAT_WXYZ",
                    ),
                },
            },
        }
    )


def resolve_robodojo_calibration(calibration: Mapping | None = None) -> dict[str, Any]:
    """Use an explicit mapping when tests inject one; otherwise dual_x5 constants."""

    if calibration is None:
        return arx_x5_calibration()
    return validate_calibration(calibration)


def validate_calibration(value: Any) -> dict[str, Any]:
    """Validate and return a JSON-serializable canonical calibration copy."""
    calibration = _require_mapping(value, "calibration")
    _require_exact_keys(calibration, _ROOT_KEYS, "calibration")

    schema_version = calibration["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CALIBRATION_SCHEMA_VERSION
    ):
        raise ValueError(f"calibration schema_version must be {CALIBRATION_SCHEMA_VERSION}, got {schema_version!r}")

    embodiment = calibration["embodiment"]
    if embodiment != ROBODOJO_EMBODIMENT:
        raise ValueError(f"calibration embodiment must be {ROBODOJO_EMBODIMENT!r}, got {embodiment!r}")

    endpoint = _require_mapping(calibration["endpoint"], "calibration endpoint")
    _require_exact_keys(endpoint, _ENDPOINT_KEYS, "calibration endpoint")
    if endpoint["link_name"] != ENDPOINT_LINK_NAME:
        raise ValueError(
            f"calibration endpoint link_name must be {ENDPOINT_LINK_NAME!r}, got {endpoint['link_name']!r}"
        )
    if endpoint["pose_frame_contract"] != ENDPOINT_POSE_FRAME_CONTRACT:
        raise ValueError(
            "calibration endpoint pose_frame_contract must be "
            f"{ENDPOINT_POSE_FRAME_CONTRACT!r}, "
            f"got {endpoint['pose_frame_contract']!r}"
        )

    arms = _require_mapping(calibration["arms"], "calibration arms")
    if set(arms) != set(ARM_NAMES):
        missing = sorted(set(ARM_NAMES) - set(arms))
        extra = sorted(set(arms) - set(ARM_NAMES))
        raise ValueError(f"calibration arms must contain exactly left and right; missing={missing}, extra={extra}")

    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "embodiment": ROBODOJO_EMBODIMENT,
        "endpoint": {
            "link_name": ENDPOINT_LINK_NAME,
            "pose_frame_contract": ENDPOINT_POSE_FRAME_CONTRACT,
        },
        "arms": {arm_name: _validate_arm_calibration(arms[arm_name], arm_name) for arm_name in ARM_NAMES},
    }


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Load a calibration JSON file and reject any contract drift."""
    calibration_path = Path(path)
    try:
        value = json.loads(calibration_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid calibration JSON in {calibration_path}: {error.msg}") from error
    return validate_calibration(value)


def save_calibration(value: Any, path: str | Path) -> dict[str, Any]:
    """Validate and save calibration JSON, returning the canonical payload."""
    calibration = validate_calibration(value)
    calibration_path = Path(path)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        calibration,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    calibration_path.write_text(f"{serialized}\n", encoding="utf-8")
    return calibration


__all__ = [
    "ARM_NAMES",
    "CALIBRATION_SCHEMA_VERSION",
    "DUAL_X5_LEFT_BASE_POS",
    "DUAL_X5_LEFT_BASE_QUAT_WXYZ",
    "DUAL_X5_RIGHT_BASE_POS",
    "DUAL_X5_RIGHT_BASE_QUAT_WXYZ",
    "EEF20_DIM",
    "EEF20_LAYOUT",
    "ENDPOINT_LINK_NAME",
    "ENDPOINT_POSE_FRAME_CONTRACT",
    "FORMAL_EPISODE_GLOB",
    "GRIPPER_CONVENTION",
    "ROBODOJO_CONTRACT_ID",
    "ROBODOJO_DATASET_VARIANTS",
    "ROBODOJO_EMBODIMENT",
    "ROBODOJO_REAL_CONTRACT_ID",
    "ROBODOJO_REAL_EMBODIMENTS",
    "ROBODOJO_REAL_GRIPPER_SENSOR_ATOL",
    "ROBODOJO_REAL_SOURCE_FRAME",
    "ROBODOJO_REAL_VARIANT",
    "ROBODOJO_SIM_SOURCE_FRAME",
    "ROBODOJO_SIM_VARIANT",
    "ROBODOJO_TARGET_FRAME",
    "REAL_ENDPOINT_NAME",
    "REAL_FRAME_SCHEMA_VERSION",
    "arx_x5_calibration",
    "discover_episodes",
    "load_calibration",
    "robodojo_real_frame_contract",
    "resolve_robodojo_calibration",
    "save_calibration",
    "validate_calibration",
    "validate_dataset_variant",
    "validate_embodiment",
]
