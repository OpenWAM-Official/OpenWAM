"""Focused tests for the shared RoboDojo EEF20 and frame contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from openwam.dataloader.robodojo_contract import (
    CALIBRATION_SCHEMA_VERSION,
    EEF20_LAYOUT,
    ENDPOINT_LINK_NAME,
    ENDPOINT_POSE_FRAME_CONTRACT,
    ROBODOJO_EMBODIMENT,
    discover_episodes,
    load_calibration,
    save_calibration,
    validate_calibration,
    validate_embodiment,
)
from openwam.dataloader.utils.eef import quat_wxyz_to_rot6d as shared_quat_wxyz_to_rot6d
from openwam.dataloader.utils.poses import (
    arms_to_eef20,
    eef20_to_arms,
    env_relative_world_to_robot_base,
    quat_wxyz_to_rot6d,
    robot_base_to_env_relative_world,
    rot6d_to_quat_wxyz,
)


def _valid_calibration() -> dict:
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "embodiment": ROBODOJO_EMBODIMENT,
        "endpoint": {
            "link_name": ENDPOINT_LINK_NAME,
            "pose_frame_contract": ENDPOINT_POSE_FRAME_CONTRACT,
        },
        "arms": {
            "left": {
                "base_pos_relative_to_env_origin": [-0.3, -0.45, 0.765],
                "base_quat_wxyz": [2**-0.5, 0.0, 0.0, 2**-0.5],
            },
            "right": {
                "base_pos_relative_to_env_origin": [0.3, -0.45, 0.765],
                "base_quat_wxyz": [2**-0.5, 0.0, 0.0, 2**-0.5],
            },
        },
    }


def test_contract_constants_pin_arx_x5_eef20_and_link6():
    assert ROBODOJO_EMBODIMENT == "arx_x5"
    assert ENDPOINT_LINK_NAME == "link6"
    assert EEF20_LAYOUT == (
        "left.xyz",
        "left.rot6d",
        "left.gripper",
        "right.xyz",
        "right.rot6d",
        "right.gripper",
    )
    validate_embodiment("arx_x5")
    with pytest.raises(ValueError, match="only supported.*arx_x5"):
        validate_embodiment("franka")
    for embodiment in ("arx_x5", "piper", "piper_x"):
        validate_embodiment(embodiment, variant="real")
    with pytest.raises(ValueError, match="only supported.*arx_x5"):
        validate_embodiment("piper", variant="sim")
    with pytest.raises(ValueError, match="variant"):
        validate_embodiment("arx_x5", variant="hardware")


def test_builtin_dual_x5_calibration_matches_robot_yaml_constants():
    from openwam.dataloader.robodojo_contract import (
        DUAL_X5_LEFT_BASE_POS,
        DUAL_X5_LEFT_BASE_QUAT_WXYZ,
        DUAL_X5_RIGHT_BASE_POS,
        DUAL_X5_RIGHT_BASE_QUAT_WXYZ,
        arx_x5_calibration,
    )

    assert DUAL_X5_LEFT_BASE_POS == (-0.3, -0.45, 0.765)
    assert DUAL_X5_RIGHT_BASE_POS == (0.3, -0.45, 0.765)
    assert DUAL_X5_LEFT_BASE_QUAT_WXYZ == (0.707, 0.0, 0.0, 0.707)
    assert DUAL_X5_RIGHT_BASE_QUAT_WXYZ == (0.707, 0.0, 0.0, 0.707)

    calibration = arx_x5_calibration()
    np.testing.assert_allclose(
        calibration["arms"]["left"]["base_pos_relative_to_env_origin"],
        DUAL_X5_LEFT_BASE_POS,
    )
    np.testing.assert_allclose(
        calibration["arms"]["right"]["base_pos_relative_to_env_origin"],
        DUAL_X5_RIGHT_BASE_POS,
    )
    left_quat = np.asarray(calibration["arms"]["left"]["base_quat_wxyz"])
    right_quat = np.asarray(calibration["arms"]["right"]["base_quat_wxyz"])
    np.testing.assert_allclose(np.linalg.norm(left_quat), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(right_quat), 1.0, atol=1e-6)
    np.testing.assert_allclose(left_quat, right_quat)


def test_formal_episode_discovery_and_flat_layout_rejection(tmp_path: Path):
    formal = tmp_path / "pick_mug" / "arx_x5" / "data"
    formal.mkdir(parents=True)
    expected = [formal / "episode_0001.hdf5", formal / "episode_0002.hdf5"]
    for path in reversed(expected):
        path.touch()
    (formal / "notes.hdf5").touch()

    assert discover_episodes(tmp_path, "pick_mug") == expected

    flat_root = tmp_path / "flat"
    flat = flat_root / "arx_x5" / "data"
    flat.mkdir(parents=True)
    (flat / "episode_0000.hdf5").touch()
    with pytest.raises(ValueError, match="flat.*not supported"):
        discover_episodes(flat_root, "pick_mug")


def test_formal_episode_discovery_rejects_missing_or_empty_layout(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="formal RoboDojo data directory"):
        discover_episodes(tmp_path, "pick_mug")

    formal = tmp_path / "pick_mug" / "arx_x5" / "data"
    formal.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=r"episode_\*\.hdf5"):
        discover_episodes(tmp_path, "pick_mug")

    with pytest.raises(ValueError, match="single path component"):
        discover_episodes(tmp_path, "../pick_mug")


def test_calibration_json_round_trip_is_validated_and_versioned(tmp_path: Path):
    payload = _valid_calibration()
    path = tmp_path / "frames.json"

    save_calibration(payload, path)

    assert load_calibration(path) == payload
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert on_disk["embodiment"] == "arx_x5"
    assert on_disk["endpoint"]["link_name"] == "link6"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda x: x.update(schema_version=999), "schema_version"),
        (lambda x: x.update(embodiment="franka"), "embodiment"),
        (lambda x: x["endpoint"].update(link_name="gripper"), "link6"),
        (lambda x: x["endpoint"].update(pose_frame_contract="tcp"), "pose_frame_contract"),
        (lambda x: x["arms"].pop("left"), "arms.*left.*right"),
        (lambda x: x["arms"].update(center=x["arms"]["left"]), "arms.*left.*right"),
        (lambda x: x["arms"]["left"].pop("base_quat_wxyz"), "base_quat_wxyz"),
        (
            lambda x: x["arms"]["left"].update(base_pos_relative_to_env_origin=[0.0, np.nan, 0.0]),
            "finite",
        ),
        (lambda x: x["arms"]["right"].update(base_quat_wxyz=[1.0, 1.0, 0.0, 0.0]), "unit"),
    ],
)
def test_calibration_schema_rejects_malformed_payloads(mutation, message):
    payload = copy.deepcopy(_valid_calibration())
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        validate_calibration(payload)


def test_calibration_schema_rejects_unknown_fields_and_non_mapping():
    payload = _valid_calibration()
    payload["surprise"] = True
    with pytest.raises(ValueError, match="unexpected.*surprise"):
        validate_calibration(payload)
    with pytest.raises(ValueError, match="JSON object"):
        validate_calibration([])


def test_wxyz_quaternion_convention_and_rot6d_round_trip():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat_wxyz_to_rot6d(identity), [1, 0, 0, 0, 1, 0], atol=2e-6)

    half_sqrt = 2**-0.5
    z_90_wxyz = np.array([half_sqrt, 0.0, 0.0, half_sqrt])
    rot6d = quat_wxyz_to_rot6d(z_90_wxyz)
    np.testing.assert_allclose(rot6d, [0, 1, 0, -1, 0, 0], atol=2e-6)
    round_trip = rot6d_to_quat_wxyz(rot6d)
    np.testing.assert_allclose(round_trip, z_90_wxyz, atol=2e-6)


def test_robodojo_rot6d_matches_shared_eef_helper():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(64, 4))
    quaternions = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        quat_wxyz_to_rot6d(quaternions),
        shared_quat_wxyz_to_rot6d(quaternions),
        atol=2e-6,
    )


def test_eef20_left_right_slots_and_round_trip():
    half_sqrt = 2**-0.5
    left_pose = np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
    right_pose = np.array([[4.0, 5.0, 6.0, half_sqrt, 0.0, 0.0, half_sqrt]])
    left_gripper = np.array([[0.25]])
    right_gripper = np.array([[0.75]])

    eef20 = arms_to_eef20(left_pose, left_gripper, right_pose, right_gripper)

    assert eef20.shape == (1, 20)
    np.testing.assert_allclose(eef20[0, 0:3], left_pose[0, :3])
    np.testing.assert_allclose(eef20[0, 3:9], [1, 0, 0, 0, 1, 0], atol=2e-6)
    assert eef20[0, 9] == 0.25
    np.testing.assert_allclose(eef20[0, 10:13], right_pose[0, :3])
    np.testing.assert_allclose(eef20[0, 13:19], [0, 1, 0, -1, 0, 0], atol=2e-6)
    assert eef20[0, 19] == 0.75

    actual = eef20_to_arms(eef20)
    for result, expected in zip(actual, (left_pose, left_gripper, right_pose, right_gripper), strict=True):
        np.testing.assert_allclose(result, expected, atol=1e-12)


def test_env_relative_world_and_robot_base_known_transform_and_round_trip():
    half_sqrt = 2**-0.5
    base_pos = np.array([1.0, 2.0, 3.0])
    base_quat = np.array([half_sqrt, 0.0, 0.0, half_sqrt])  # base +x points along world +y
    world_pose = np.array([1.0, 3.0, 3.0, *base_quat])

    base_pose = env_relative_world_to_robot_base(world_pose, base_pos, base_quat)

    np.testing.assert_allclose(base_pose, [1, 0, 0, 1, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(
        robot_base_to_env_relative_world(base_pose, base_pos, base_quat),
        world_pose,
        atol=1e-12,
    )

    poses = np.array(
        [
            [0.2, -0.4, 1.1, 1.0, 0.0, 0.0, 0.0],
            [2.0, 0.5, -1.0, 0.5, 0.5, 0.5, 0.5],
        ]
    )
    converted = env_relative_world_to_robot_base(poses, base_pos, base_quat)
    np.testing.assert_allclose(
        robot_base_to_env_relative_world(converted, base_pos, base_quat),
        poses,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: quat_wxyz_to_rot6d(np.ones(3)), r"shape.*4"),
        (lambda: quat_wxyz_to_rot6d(np.array([1.0, 0.0, np.inf, 0.0])), "finite"),
        (lambda: quat_wxyz_to_rot6d(np.array([2.0, 0.0, 0.0, 0.0])), "unit"),
        (lambda: rot6d_to_quat_wxyz(np.zeros(6)), "degenerate"),
        (lambda: rot6d_to_quat_wxyz(np.array([1, 0, 0, 2, 0, 0])), "degenerate"),
        (
            lambda: env_relative_world_to_robot_base(
                np.zeros(6),
                np.zeros(3),
                np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            r"pose.*shape.*7",
        ),
        (
            lambda: arms_to_eef20(
                np.array([0, 0, 0, 1, 0, 0, 0]),
                np.array([np.nan]),
                np.array([0, 0, 0, 1, 0, 0, 0]),
                np.array([0.0]),
            ),
            "left_gripper.*finite",
        ),
        (
            lambda: arms_to_eef20(
                np.zeros((2, 7)),
                np.zeros((2, 1)),
                np.zeros((3, 7)),
                np.zeros((3, 1)),
            ),
            "leading shapes",
        ),
        (lambda: eef20_to_arms(np.zeros(19)), r"EEF20.*shape.*20"),
    ],
)
def test_frame_helpers_reject_malformed_inputs(call, message):
    with pytest.raises(ValueError, match=message):
        call()
