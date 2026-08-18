"""Pin the train/eval RoboDojo copies and forbid cross-runtime imports.

Production modules may not import the other runtime. Tests in this file are
the allowed exception.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import benchmarks.robodojo.contract as eval_contract
import benchmarks.robodojo.frames as eval_frames
import openwam.dataloader.robodojo_contract as train_contract
import openwam.dataloader.utils.poses as train_poses
from benchmarks.robodojo.prompt_template import (
    format_prompt_for_inference as eval_prompt,
)
from openwam.dataloader.robodojo import calibration_fingerprint
from openwam.dataloader.transforms.multiview import (
    format_prompt_for_inference as train_prompt,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OPENWAM_IMPORT = ("openwam", "openwam.")
_BENCHMARKS_IMPORT = ("benchmarks", "benchmarks.")


def _production_py_files(relative: str) -> list[Path]:
    root = _REPO_ROOT / relative
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _starts_with(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes)


def test_openwam_production_does_not_import_benchmarks():
    offenders = []
    for path in _production_py_files("openwam"):
        for name in _imported_roots(path):
            if _starts_with(name, _BENCHMARKS_IMPORT):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert offenders == []


def test_robodojo_eval_production_does_not_import_openwam():
    offenders = []
    for path in _production_py_files("benchmarks/robodojo"):
        for name in _imported_roots(path):
            if _starts_with(name, _OPENWAM_IMPORT):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert offenders == []


def test_contract_identifiers_and_calibration_fingerprint_match():
    assert eval_contract.ROBODOJO_CONTRACT_ID == train_contract.ROBODOJO_CONTRACT_ID
    assert eval_contract.ROBODOJO_CONTRACT_ID == "robodojo-eef20-v1"
    assert eval_contract.ROBODOJO_EMBODIMENT == train_contract.ROBODOJO_EMBODIMENT
    assert eval_contract.EEF20_LAYOUT == train_contract.EEF20_LAYOUT
    assert eval_contract.GRIPPER_CONVENTION == train_contract.GRIPPER_CONVENTION
    assert eval_contract.GRIPPER_CONVENTION == "zero_closed_one_open"
    assert eval_contract.DUAL_X5_LEFT_BASE_POS == train_contract.DUAL_X5_LEFT_BASE_POS
    assert eval_contract.DUAL_X5_RIGHT_BASE_POS == train_contract.DUAL_X5_RIGHT_BASE_POS
    assert eval_contract.DUAL_X5_LEFT_BASE_QUAT_WXYZ == train_contract.DUAL_X5_LEFT_BASE_QUAT_WXYZ
    assert eval_contract.DUAL_X5_RIGHT_BASE_QUAT_WXYZ == train_contract.DUAL_X5_RIGHT_BASE_QUAT_WXYZ
    train_cal = train_contract.arx_x5_calibration()
    eval_cal = eval_contract.arx_x5_calibration()
    assert train_cal == eval_cal
    assert calibration_fingerprint(train_cal) == calibration_fingerprint(eval_cal)


def _random_unit_quaternions(count: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(count, 4))
    return raw / np.linalg.norm(raw, axis=-1, keepdims=True)


def test_rot6d_and_180_degree_round_trips_match():
    rng = np.random.default_rng(0)
    quaternions = _random_unit_quaternions(64, rng)
    # Canonical 180° rotations about the three axes, plus one diagonal.
    one_eighties = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 2**-0.5, 2**-0.5, 0.0],
        ]
    )
    for source in (quaternions, one_eighties):
        train_rot = train_poses.quat_wxyz_to_rot6d(source)
        eval_rot = eval_frames.quat_wxyz_to_rot6d(source)
        np.testing.assert_array_equal(train_rot, eval_rot)
        train_back = train_poses.rot6d_to_quat_wxyz(train_rot)
        eval_back = eval_frames.rot6d_to_quat_wxyz(eval_rot)
        np.testing.assert_allclose(np.abs(np.sum(train_back * source, axis=-1)), 1.0, atol=1e-6)
        np.testing.assert_allclose(train_back, eval_back, atol=1e-6)


def test_base_transform_round_trip_and_pose_at_base_match():
    calibration = train_contract.arx_x5_calibration()
    rng = np.random.default_rng(1)
    pose = np.concatenate(
        (rng.normal(size=(8, 3)) * 0.1, _random_unit_quaternions(8, rng)),
        axis=-1,
    )
    for arm_name in ("left", "right"):
        arm = calibration["arms"][arm_name]
        base_pos = arm["base_pos_relative_to_env_origin"]
        base_quat = arm["base_quat_wxyz"]
        train_base = train_poses.env_relative_world_to_robot_base(pose, base_pos, base_quat)
        eval_base = eval_frames.env_relative_world_to_robot_base(pose, base_pos, base_quat)
        np.testing.assert_allclose(train_base, eval_base, atol=1e-12)
        train_world = train_poses.robot_base_to_env_relative_world(
            train_base, base_pos, base_quat
        )
        eval_world = eval_frames.robot_base_to_env_relative_world(
            eval_base, base_pos, base_quat
        )
        np.testing.assert_allclose(train_world[..., :3], pose[..., :3], atol=1e-12)
        np.testing.assert_allclose(eval_world[..., :3], pose[..., :3], atol=1e-12)
        at_base = np.concatenate((np.asarray(base_pos), np.asarray(base_quat)))
        expected = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            train_poses.env_relative_world_to_robot_base(at_base, base_pos, base_quat),
            expected,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            eval_frames.env_relative_world_to_robot_base(at_base, base_pos, base_quat),
            expected,
            atol=1e-12,
        )


def test_eef20_pack_unpack_and_gripper_direction_match():
    identity = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    closed = np.array([0.0])
    opened = np.array([1.0])
    noise = np.array([-3.21e-17])
    train_eef = train_poses.arms_to_eef20(identity, closed, identity, opened)
    eval_eef = eval_frames.arms_to_eef20(identity, closed, identity, opened)
    np.testing.assert_array_equal(train_eef, eval_eef)
    assert train_eef[9] == pytest.approx(0.0)
    assert train_eef[19] == pytest.approx(1.0)
    train_left, train_lg, train_right, train_rg = train_poses.eef20_to_arms(train_eef)
    eval_left, eval_lg, eval_right, eval_rg = eval_frames.eef20_to_arms(eval_eef)
    np.testing.assert_allclose(train_left, eval_left, atol=1e-6)
    np.testing.assert_allclose(train_right, eval_right, atol=1e-6)
    np.testing.assert_array_equal(train_lg, eval_lg)
    np.testing.assert_array_equal(train_rg, eval_rg)
    # Official closed-gripper float noise is a reader/client clip concern;
    # packing itself must accept a value that is numerically ~0.
    packed_noise = train_poses.arms_to_eef20(identity, noise, identity, opened)
    assert packed_noise[9] == pytest.approx(0.0, abs=1e-16)


def test_prompt_templates_are_byte_identical():
    for base in ("stack the blocks", "fold the towel.", "", "brace {x} and (y)"):
        assert eval_prompt(base) == train_prompt(base)
