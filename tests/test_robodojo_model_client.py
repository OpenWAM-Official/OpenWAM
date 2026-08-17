"""Unit contract for the OpenWAM-only RoboDojo evaluation bridge.

These tests intentionally use fake transports and tiny EvalEnv-like objects.
Importing this module must not import or launch Isaac.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import benchmarks.robodojo.single_eval as single_eval
import benchmarks.robodojo.smoke_robodojo as smoke_module
from benchmarks.robodojo.openwam_model_client import OpenWAMRoboDojoModelClient
from benchmarks.robodojo.single_eval import (
    _run_native_episodes,
    configure_runtime_import_paths,
    construct_with_no_network_client,
    prepare_launcher_runtime,
    runtime_working_directory,
    validate_runner_config,
    verify_runtime_import_provenance,
)
from benchmarks.robodojo.smoke_robodojo import (
    make_synthetic_debug_observation,
    run_debug_protocol_rollout,
)
from openwam.dataloader.transforms.multiview import format_prompt_for_inference
from benchmarks.robodojo.frames import (
    arms_to_eef20,
    env_relative_world_to_robot_base,
    robot_base_to_env_relative_world,
)

IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
ROBOT_DIMS = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def valid_calibration() -> dict:
    half_sqrt = 2**-0.5
    return {
        "schema_version": 1,
        "embodiment": "arx_x5",
        "endpoint": {
            "link_name": "link6",
            "pose_frame_contract": "rigid_terminal_arm_frame_independent_of_gripper_motion",
        },
        "arms": {
            "left": {
                "base_pos_relative_to_env_origin": [-0.3, -0.45, 0.765],
                "base_quat_wxyz": [half_sqrt, 0.0, 0.0, half_sqrt],
            },
            "right": {
                "base_pos_relative_to_env_origin": [0.3, -0.45, 0.765],
                "base_quat_wxyz": [half_sqrt, 0.0, 0.0, half_sqrt],
            },
        },
    }


def encoded_jpeg(color=(12, 34, 56), size=(13, 11)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def valid_observation(*, instruction="Move the block.", env_idx=0) -> dict:
    calibration = valid_calibration()
    left_cal = calibration["arms"]["left"]
    right_cal = calibration["arms"]["right"]
    left_base = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    right_base = np.array([-0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0])
    left_world = robot_base_to_env_relative_world(
        left_base,
        left_cal["base_pos_relative_to_env_origin"],
        left_cal["base_quat_wxyz"],
    )
    right_world = robot_base_to_env_relative_world(
        right_base,
        right_cal["base_pos_relative_to_env_origin"],
        right_cal["base_quat_wxyz"],
    )
    return {
        "vision": {
            "cam_head": {
                "color": np.full((11, 13, 3), (10, 20, 30), dtype=np.uint8)
            },
            "cam_left_wrist": {"color": encoded_jpeg((40, 50, 60))},
            "cam_right_wrist": {
                "color": np.frombuffer(encoded_jpeg((70, 80, 90)), dtype=np.uint8)
            },
        },
        "instruction": instruction,
        "state": {
            "left_ee_pose": left_world,
            "left_ee_joint_state": np.array([0.25]),
            "right_ee_pose": right_world,
            "right_ee_joint_state": np.array([0.75]),
        },
        "env_idx": env_idx,
    }


def valid_action(
    *,
    left_xyz=(0.4, 0.5, 0.6),
    left_gripper=0.2,
    right_xyz=(-0.4, 0.1, 0.8),
    right_gripper=0.9,
) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(left_xyz, dtype=np.float64),
            IDENTITY_ROT6D,
            [left_gripper],
            np.asarray(right_xyz, dtype=np.float64),
            IDENTITY_ROT6D,
            [right_gripper],
        )
    )


class FakeTransport:
    def __init__(
        self,
        *,
        pong=None,
        resets=None,
        actions=None,
        ping_error: BaseException | None = None,
        reset_errors=None,
        action_errors=None,
    ):
        self.pong = {"type": "pong"} if pong is None else pong
        self.resets = list(resets or [{"type": "reset_ack"}])
        self.actions = list(
            actions
            or [
                {
                    "type": "action",
                    "action": valid_action().tolist(),
                    "step": 3,
                    "latency_ms": 12.5,
                }
            ]
        )
        self.ping_error = ping_error
        self.reset_errors = list(reset_errors or [])
        self.action_errors = list(action_errors or [])
        self.payloads = []
        self.calls = []
        self.closed = False

    def ping(self):
        self.calls.append("ping")
        if self.ping_error is not None:
            raise self.ping_error
        return self.pong

    def reset(self):
        self.calls.append("reset")
        if self.reset_errors:
            raise self.reset_errors.pop(0)
        return self.resets.pop(0)

    def predict_once(self, payload):
        self.calls.append("predict_once")
        self.payloads.append(copy.deepcopy(payload))
        if self.action_errors:
            raise self.action_errors.pop(0)
        return self.actions.pop(0)

    def predict(self, payload):
        raise AssertionError("RoboDojo adapter must use predict_once")

    def close(self):
        self.calls.append("close")
        self.closed = True


class TransportFactory:
    def __init__(self, transport):
        self.transport = transport
        self.calls = []

    def __call__(self, ws_url, **kwargs):
        self.calls.append((ws_url, kwargs))
        return self.transport


def make_client(
    transport: FakeTransport | None = None,
    *,
    calibration: dict | None = None,
    **kwargs,
) -> tuple[OpenWAMRoboDojoModelClient, FakeTransport, TransportFactory]:
    transport = transport or FakeTransport()
    factory = TransportFactory(transport)
    auto_reset = kwargs.pop("auto_reset", True)
    client = OpenWAMRoboDojoModelClient(
        task_env=kwargs.pop("task_env", None),
        host=kwargs.pop("host", "127.0.0.1"),
        port=kwargs.pop("port", 8848),
        timeout=kwargs.pop("timeout", 42.0),
        num_envs=kwargs.pop("num_envs", 1),
        env_config=kwargs.pop("env_config", "arx_x5"),
        robot_action_dim_info=kwargs.pop(
            "robot_action_dim_info", copy.deepcopy(ROBOT_DIMS)
        ),
        transport_factory=factory,
        frame_provider=kwargs.pop(
            "frame_provider", lambda: calibration or valid_calibration()
        ),
        **kwargs,
    )
    if auto_reset:
        client.call(func_name="reset")
    return client, transport, factory


def test_fake_transport_ping_reset_action_close_and_camera_passthrough():
    client, transport, factory = make_client(auto_reset=False)
    assert transport.calls == ["ping"]
    assert factory.calls == [
        (
            "ws://127.0.0.1:8848",
            {"timeout": 42.0, "open_timeout": 10.0},
        )
    ]

    assert client.call(func_name="reset") is None
    observation = valid_observation()
    client.call(func_name="update_obs", obs=observation)
    actions = client.call(func_name="get_action")
    client.close()

    assert transport.calls == ["ping", "reset", "predict_once", "close"]
    assert transport.closed
    assert isinstance(actions, list) and len(actions) == 1
    payload = transport.payloads[0]
    assert set(payload["images"]) == {
        "head_camera",
        "left_wrist_camera",
        "right_wrist_camera",
    }
    with Image.open(io.BytesIO(base64.b64decode(payload["images"]["head_camera"]))) as image:
        assert image.size == (13, 11)
    assert base64.b64decode(payload["images"]["left_wrist_camera"]) == observation[
        "vision"
    ]["cam_left_wrist"]["color"]
    assert base64.b64decode(payload["images"]["right_wrist_camera"]) == observation[
        "vision"
    ]["cam_right_wrist"]["color"].tobytes()


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        (FakeTransport(pong={"type": "action"}), "pong"),
        (FakeTransport(ping_error=OSError("offline")), "offline"),
    ],
)
def test_constructor_rejects_wrong_ping_ack_and_transport_failure(transport, message):
    factory = TransportFactory(transport)
    with pytest.raises((ConnectionError, OSError, RuntimeError), match=message):
        OpenWAMRoboDojoModelClient(
            task_env=None,
            host="localhost",
            port=8848,
            num_envs=1,
            env_config="arx_x5",
            robot_action_dim_info=ROBOT_DIMS,
            transport_factory=factory,
            frame_provider=valid_calibration,
        )
    assert transport.closed


def test_reset_rejects_wrong_ack_and_clears_cached_observation():
    transport = FakeTransport(resets=[{"type": "nope"}, {"type": "reset_ack"}])
    client, _, _ = make_client(transport, auto_reset=False)
    with pytest.raises(RuntimeError, match="poisoned.*reset"):
        client.call(func_name="update_obs", obs=valid_observation())
    with pytest.raises(RuntimeError, match="reset_ack"):
        client.call(func_name="reset")

    with pytest.raises(RuntimeError, match="poisoned.*reset"):
        client.call(func_name="get_action")
    client.call(func_name="reset")
    with pytest.raises(RuntimeError, match="update_obs"):
        client.call(func_name="get_action")


def test_constructor_starts_poisoned_until_reset_ack():
    client, transport, _ = make_client(auto_reset=False)

    with pytest.raises(RuntimeError, match="poisoned.*reset"):
        client.call(func_name="update_obs", obs=valid_observation())
    with pytest.raises(RuntimeError, match="poisoned.*reset"):
        client.call(func_name="get_action")

    client.call(func_name="reset")
    client.call(func_name="update_obs", obs=valid_observation())
    assert len(client.call(func_name="get_action")) == 1
    assert transport.calls == ["ping", "reset", "predict_once"]


def test_ambiguous_action_transport_poison_requires_successful_reset():
    transport = FakeTransport(
        action_errors=[OSError("connection dropped after send")],
        actions=[{"type": "action", "action": valid_action().tolist()}],
        resets=[{"type": "reset_ack"}, {"type": "reset_ack"}],
    )
    client, _, _ = make_client(transport)
    client.call(func_name="update_obs", obs=valid_observation())

    with pytest.raises(OSError, match="dropped after send"):
        client.call(func_name="get_action")
    with pytest.raises(RuntimeError, match="poisoned.*reset"):
        client.call(func_name="update_obs", obs=valid_observation())

    client.call(func_name="reset")
    client.call(func_name="update_obs", obs=valid_observation())
    assert len(client.call(func_name="get_action")) == 1
    assert transport.calls == [
        "ping",
        "reset",
        "predict_once",
        "reset",
        "predict_once",
    ]


def test_invalid_action_response_poison_survives_failed_reset():
    transport = FakeTransport(
        actions=[
            {"type": "action", "action": np.zeros(14).tolist()},
            {"type": "action", "action": valid_action().tolist()},
        ],
        resets=[
            {"type": "reset_ack"},
            {"type": "wrong"},
            {"type": "reset_ack"},
        ],
    )
    client, _, _ = make_client(transport)
    client.call(func_name="update_obs", obs=valid_observation())

    with pytest.raises(ValueError, match=r"shape \(20,\)"):
        client.call(func_name="get_action")
    with pytest.raises(RuntimeError, match="reset_ack"):
        client.call(func_name="reset")
    with pytest.raises(RuntimeError, match="poisoned.*reset"):
        client.call(func_name="update_obs", obs=valid_observation())

    client.call(func_name="reset")
    client.call(func_name="update_obs", obs=valid_observation())
    assert len(client.call(func_name="get_action")) == 1


def test_reset_transport_failure_clears_obs_and_poison_stays_closed():
    transport = FakeTransport(
        resets=[{"type": "reset_ack"}, {"type": "reset_ack"}],
    )
    client, _, _ = make_client(transport)
    transport.reset_errors.append(OSError("reset transport failed"))
    client.call(func_name="update_obs", obs=valid_observation())

    with pytest.raises(OSError, match="reset transport failed"):
        client.call(func_name="reset")
    with pytest.raises(RuntimeError, match="poisoned.*reset"):
        client.call(func_name="get_action")

    client.call(func_name="reset")
    with pytest.raises(RuntimeError, match="update_obs"):
        client.call(func_name="get_action")


def test_all_model_calls_fail_after_close():
    client, _, _ = make_client()
    client.close()

    calls = [
        lambda: client.call(func_name="reset"),
        lambda: client.call(func_name="update_obs", obs=valid_observation()),
        lambda: client.call(func_name="get_action"),
        lambda: client.call(func_name="get_action_batch", obs=[0]),
        lambda: client.call(func_name="unknown"),
    ]
    for call in calls:
        with pytest.raises(RuntimeError, match="closed"):
            call()


@pytest.mark.parametrize("wrap", [lambda obs: obs, lambda obs: [obs]])
def test_direct_and_one_item_list_observations_are_accepted(wrap):
    client, transport, _ = make_client()
    client.call(func_name="update_obs", obs=wrap(valid_observation()))
    client.call(func_name="get_action")
    assert len(transport.payloads) == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda obs: obs["vision"].pop("cam_head"), "cam_head"),
        (
            lambda obs: obs["vision"]["cam_left_wrist"].pop("color"),
            "cam_left_wrist.*color",
        ),
        (
            lambda obs: obs["vision"]["cam_head"].update(
                color=np.zeros((10, 10), dtype=np.uint8)
            ),
            "cam_head.*color",
        ),
        (
            lambda obs: obs["vision"]["cam_head"].update(
                color=np.zeros((10, 10, 3), dtype=np.float32)
            ),
            "uint8",
        ),
        (lambda obs: obs.update(instruction="  "), "instruction"),
        (lambda obs: obs.update(instruction=np.array(["not", "scalar"])), "scalar"),
        (lambda obs: obs["state"].pop("left_ee_pose"), "left_ee_pose"),
        (
            lambda obs: obs["state"].update(left_ee_pose=np.zeros(6)),
            r"left_ee_pose.*\(7,\)",
        ),
        (
            lambda obs: obs["state"]["right_ee_pose"].__setitem__(0, np.nan),
            "finite",
        ),
        (
            lambda obs: obs["state"]["left_ee_pose"].__setitem__(
                slice(3, 7), [2.0, 0.0, 0.0, 0.0]
            ),
            "unit.*wxyz",
        ),
        (
            lambda obs: obs["state"].update(right_ee_joint_state=np.zeros(2)),
            r"right_ee_joint_state.*\(1,\)",
        ),
        (
            lambda obs: obs["state"].update(left_ee_joint_state=[-0.1]),
            r"\[0, 1\]",
        ),
        (lambda obs: obs.update(env_idx=1), "env_idx.*0"),
    ],
)
def test_nested_observation_schema_rejects_malformed_inputs(mutation, message):
    client, _, _ = make_client()
    obs = valid_observation()
    mutation(obs)
    with pytest.raises((KeyError, TypeError, ValueError), match=message):
        client.call(func_name="update_obs", obs=obs)


def test_observation_closed_gripper_float_noise_is_accepted_and_clipped():
    client, transport, _ = make_client()
    obs = valid_observation()
    obs["state"]["left_ee_joint_state"] = np.array([-3.21e-17])
    client.call(func_name="update_obs", obs=obs)
    client.call(func_name="get_action")
    assert transport.payloads[0]["state"][9] == 0.0


@pytest.mark.parametrize("bad_obs", [[], [{}, {}], "not-an-observation"])
def test_observation_wrapper_must_contain_exactly_one_mapping(bad_obs):
    client, _, _ = make_client()
    with pytest.raises((TypeError, ValueError), match="observation|one"):
        client.call(func_name="update_obs", obs=bad_obs)


@pytest.mark.parametrize("instruction", ["Lift the cup.", b"Lift the cup.", np.bytes_(b"Lift the cup.")])
def test_prompt_has_byte_string_parity_with_training_formatter(instruction):
    client, transport, _ = make_client()
    client.call(func_name="update_obs", obs=valid_observation(instruction=instruction))
    client.call(func_name="get_action")
    assert transport.payloads[0]["prompt"] == format_prompt_for_inference(
        "Lift the cup."
    )


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeRobotManager:
    def __init__(self, robots, base_world_poses):
        self.robot_list = robots
        self.base_world_poses = base_world_poses
        self.calls = []

    def get_link_pose(self, *, robot, link_name, env_idx_list, is_relative):
        self.calls.append((robot.arm_name, link_name, env_idx_list, is_relative))
        return {env_idx_list[0]: self.base_world_poses[robot.arm_name]}


def fake_live_env():
    left = SimpleNamespace(
        type="target",
        robot_type="arm",
        robot_name="x5",
        arm_name="left_arm",
    )
    right = SimpleNamespace(
        type="target",
        robot_type="arm",
        robot_name="x5",
        arm_name="right_arm",
    )
    origin = np.array([10.0, 20.0, 30.0])
    calibration = valid_calibration()
    poses = {}
    for side, robot in (("left", left), ("right", right)):
        arm = calibration["arms"][side]
        poses[robot.arm_name] = np.r_[
            origin + np.asarray(arm["base_pos_relative_to_env_origin"]),
            arm["base_quat_wxyz"],
        ]
    manager = FakeRobotManager([left, right], poses)
    env = SimpleNamespace(
        num_envs=1,
        robot_action_dim_info=copy.deepcopy(ROBOT_DIMS),
        robot_manager=manager,
        env_origins=FakeTensor([origin]),
    )
    return env, manager


def test_live_base_transform_on_input_and_exact_inverse_on_output():
    calibration = valid_calibration()
    action = valid_action(
        left_xyz=(0.7, -0.1, 0.2),
        left_gripper=-2.0,
        right_xyz=(-0.6, 0.3, 0.9),
        right_gripper=3.0,
    )
    transport = FakeTransport(
        actions=[
            {
                "type": "action",
                "action": action.tolist(),
                "step": 8,
                "latency_ms": 4.25,
            }
        ]
    )
    env, manager = fake_live_env()
    client, _, _ = make_client(
        transport,
        task_env=env,
        frame_provider=None,
        num_envs=1,
        robot_action_dim_info=ROBOT_DIMS,
    )
    obs = valid_observation()
    client.call(func_name="update_obs", obs=obs)
    result = client.call(func_name="get_action")[0]

    left_cal = calibration["arms"]["left"]
    right_cal = calibration["arms"]["right"]
    expected_state = arms_to_eef20(
        env_relative_world_to_robot_base(
            obs["state"]["left_ee_pose"],
            left_cal["base_pos_relative_to_env_origin"],
            left_cal["base_quat_wxyz"],
        ),
        obs["state"]["left_ee_joint_state"],
        env_relative_world_to_robot_base(
            obs["state"]["right_ee_pose"],
            right_cal["base_pos_relative_to_env_origin"],
            right_cal["base_quat_wxyz"],
        ),
        obs["state"]["right_ee_joint_state"],
    )
    np.testing.assert_allclose(transport.payloads[0]["state"], expected_state)
    np.testing.assert_allclose(
        result["left_ee_pose"],
        robot_base_to_env_relative_world(
            np.r_[action[:3], [1.0, 0.0, 0.0, 0.0]],
            left_cal["base_pos_relative_to_env_origin"],
            left_cal["base_quat_wxyz"],
        ),
    )
    np.testing.assert_allclose(
        result["right_ee_pose"],
        robot_base_to_env_relative_world(
            np.r_[action[10:13], [1.0, 0.0, 0.0, 0.0]],
            right_cal["base_pos_relative_to_env_origin"],
            right_cal["base_quat_wxyz"],
        ),
        atol=1e-12,
    )
    assert manager.calls == []


def test_native_action_dict_keys_shapes_and_only_grippers_are_clipped():
    unclipped_pose = (7.0, -8.0, 9.0)
    action = valid_action(
        left_xyz=unclipped_pose,
        left_gripper=-0.5,
        right_xyz=(-7.0, 8.0, -9.0),
        right_gripper=1.5,
    )
    transport = FakeTransport(
        actions=[{"type": "action", "action": action.tolist()}]
    )
    client, _, _ = make_client(transport)
    client.call(func_name="update_obs", obs=valid_observation())
    output = client.call(func_name="get_action")
    assert len(output) == 1
    native = output[0]
    assert set(native) == {
        "left_ee_pose",
        "left_ee_joint_state",
        "right_ee_pose",
        "right_ee_joint_state",
    }
    assert native["left_ee_pose"].shape == (7,)
    assert native["right_ee_pose"].shape == (7,)
    assert native["left_ee_joint_state"].shape == (1,)
    assert native["right_ee_joint_state"].shape == (1,)
    assert native["left_ee_joint_state"][0] == 0.0
    assert native["right_ee_joint_state"][0] == 1.0
    left_cal = valid_calibration()["arms"]["left"]
    left_base_again = env_relative_world_to_robot_base(
        native["left_ee_pose"],
        left_cal["base_pos_relative_to_env_origin"],
        left_cal["base_quat_wxyz"],
    )
    np.testing.assert_allclose(left_base_again[:3], unclipped_pose)


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (np.zeros(14), r"\(20,\).*14|width.*20"),
        (np.zeros(80), r"\(20,\).*80|width.*20"),
        (np.r_[valid_action()[:-1], np.nan], "finite"),
        (
            np.r_[
                valid_action()[:3],
                np.zeros(6),
                valid_action()[9:],
            ],
            "degenerate",
        ),
    ],
)
def test_server_action_rejects_wrong_width_nan_and_degenerate_rot6d(action, message):
    transport = FakeTransport(
        actions=[{"type": "action", "action": np.asarray(action).tolist()}]
    )
    client, _, _ = make_client(transport)
    client.call(func_name="update_obs", obs=valid_observation())
    with pytest.raises(ValueError, match=message):
        client.call(func_name="get_action")


def test_server_action_requires_action_response_type_and_flat_array():
    for response, message in (
        ({"type": "pong", "action": valid_action().tolist()}, "type.*action"),
        ({"type": "action", "action": [valid_action().tolist()]}, r"flat.*\(20,\)"),
    ):
        client, _, _ = make_client(FakeTransport(actions=[response]))
        client.call(func_name="update_obs", obs=valid_observation())
        with pytest.raises(ValueError, match=message):
            client.call(func_name="get_action")


def test_out_of_order_unknown_payload_and_batch_calls_fail_clearly():
    client, _, _ = make_client()
    with pytest.raises(RuntimeError, match="update_obs"):
        client.call(func_name="get_action")
    with pytest.raises(TypeError, match="obs"):
        client.call(func_name="reset", obs={})
    with pytest.raises(TypeError, match="obs"):
        client.call(func_name="get_action", obs=[])
    with pytest.raises(NotImplementedError, match="batch.*single-env"):
        client.call(func_name="update_obs_batch", obs=[valid_observation()])
    with pytest.raises(NotImplementedError, match="batch.*single-env"):
        client.call(func_name="get_action_batch", obs=[0])
    with pytest.raises(NotImplementedError, match="unknown"):
        client.call(func_name="unknown")

    client.call(func_name="update_obs", obs=valid_observation())
    client.call(func_name="get_action")
    with pytest.raises(RuntimeError, match="update_obs"):
        client.call(func_name="get_action")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_envs": 2}, "exactly one"),
        ({"num_envs": 1.5}, "integer"),
        ({"env_config": "franka"}, "arx_x5"),
        (
            {"robot_action_dim_info": {"arm_dim": [7, 7], "ee_dim": [1, 1]}},
            r"\[6, 6\]",
        ),
        (
            {"robot_action_dim_info": {"arm_dim": [6, 6], "ee_dim": [2, 2]}},
            r"\[1, 1\]",
        ),
    ],
)
def test_one_env_embodiment_and_dimension_guards_run_before_transport(
    overrides, message
):
    transport = FakeTransport()
    factory = TransportFactory(transport)
    kwargs = {
        "task_env": None,
        "host": "localhost",
        "port": 8848,
        "num_envs": 1,
        "env_config": "arx_x5",
        "robot_action_dim_info": copy.deepcopy(ROBOT_DIMS),
        "transport_factory": factory,
        "frame_provider": valid_calibration,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        OpenWAMRoboDojoModelClient(**kwargs)
    assert factory.calls == []


def test_live_env_guard_rejects_non_dual_target_x5():
    env, _ = fake_live_env()
    env.robot_manager.robot_list[1].robot_name = "franka"
    factory = TransportFactory(FakeTransport())
    with pytest.raises(ValueError, match="two target X5|dual-arm.*x5"):
        OpenWAMRoboDojoModelClient(
            task_env=env,
            host="localhost",
            port=8848,
            num_envs=1,
            env_config="arx_x5",
            robot_action_dim_info=ROBOT_DIMS,
            transport_factory=factory,
        )
    assert factory.calls == []


def test_debug_metadata_records_canonical_server_eef20_and_frame_details():
    action = valid_action()
    transport = FakeTransport(
        actions=[
            {
                "type": "action",
                "action": action.tolist(),
                "step": 17,
                "latency_ms": 99.5,
            }
        ]
    )
    client, _, _ = make_client(transport, debug=True)
    client.call(func_name="update_obs", obs=valid_observation())
    native = client.call(func_name="get_action")[0]

    assert len(client.debug_records) == 1
    record = client.debug_records[0]
    np.testing.assert_array_equal(record["server_action_eef20"], action)
    assert np.asarray(record["observation_eef20"]).shape == (20,)
    assert record["response_step"] == 17
    assert record["latency_ms"] == 99.5
    assert set(record["base_transforms"]) == {"left", "right"}
    assert record["camera_shapes"]["cam_head"] == [11, 13, 3]
    np.testing.assert_array_equal(
        record["converted_action"]["left_ee_pose"], native["left_ee_pose"]
    )


class OriginalNetworkClient:
    pass


def test_no_network_placeholder_factory_is_restored_on_success():
    owner = SimpleNamespace(WsModelClient=OriginalNetworkClient)
    captured = {}

    def constructor():
        captured["during"] = owner.WsModelClient
        captured["client"] = owner.WsModelClient(url="ws://must-not-connect")
        return SimpleNamespace(model_client=captured["client"])

    env = construct_with_no_network_client(owner, constructor)
    assert owner.WsModelClient is OriginalNetworkClient
    assert captured["during"] is not OriginalNetworkClient
    assert captured["client"].closed
    assert env.model_client is None


def test_no_network_placeholder_factory_is_restored_on_failure():
    owner = SimpleNamespace(WsModelClient=OriginalNetworkClient)
    captured = {}

    def constructor():
        captured["client"] = owner.WsModelClient(url="ws://must-not-connect")
        raise RuntimeError("construction failed")

    with pytest.raises(RuntimeError, match="construction failed"):
        construct_with_no_network_client(owner, constructor)
    assert owner.WsModelClient is OriginalNetworkClient
    assert captured["client"].closed


def valid_runner_config(root: Path) -> dict:
    return {
        "robodojo_root": str(root),
        "host": "127.0.0.1",
        "port": 8848,
        "task": "stack_blocks",
        "env_config": "arx_x5",
        "device_id": 0,
        "num_envs": 1,
        "eval_count": 1,
        "eval_batch": False,
        "timeout": 300.0,
        "seed": 0,
        "headless": True,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg.update(robodojo_root="/definitely/missing"), "root"),
        (lambda cfg: cfg.update(env_config="franka"), "arx_x5"),
        (lambda cfg: cfg.update(num_envs=2), "exactly one"),
        (lambda cfg: cfg.update(eval_batch=True), "eval_batch"),
        (lambda cfg: cfg.update(device_id=-1), "device"),
        (lambda cfg: cfg.update(host=""), "host"),
        (lambda cfg: cfg.update(host="ws://localhost"), "host"),
        (lambda cfg: cfg.update(port=0), "port"),
        (lambda cfg: cfg.update(port=70000), "port"),
        (lambda cfg: cfg.update(timeout=0), "timeout"),
        (lambda cfg: cfg.update(eval_count=0), "eval_count"),
        (lambda cfg: cfg.update(task=""), "task"),
    ],
)
def test_runner_validation_rejects_invalid_prelaunch_config(
    tmp_path: Path, mutation, message
):
    cfg = valid_runner_config(tmp_path)
    mutation(cfg)
    with pytest.raises((FileNotFoundError, TypeError, ValueError), match=message):
        validate_runner_config(cfg)


def test_runner_validation_returns_normalized_single_env_config(tmp_path: Path):
    cfg = valid_runner_config(tmp_path)
    cfg.update(port="8848", timeout="10", eval_count="2")
    result = validate_runner_config(cfg)
    assert result["robodojo_root"] == str(tmp_path.resolve())
    assert result["port"] == 8848
    assert result["timeout"] == 10.0
    assert result["eval_count"] == 2
    assert result["num_envs"] == 1
    assert result["eval_batch"] is False


@pytest.mark.parametrize("max_steps", [True, False, 0, -1, 1.5, "1.5"])
def test_runner_validation_rejects_nonpositive_or_noninteger_max_steps(
    tmp_path: Path,
    max_steps,
):
    cfg = valid_runner_config(tmp_path)
    cfg["max_steps"] = max_steps

    with pytest.raises(ValueError, match="max_steps.*positive integer"):
        validate_runner_config(cfg)


def test_runner_validation_normalizes_optional_max_steps(tmp_path: Path):
    cfg = valid_runner_config(tmp_path)
    cfg["max_steps"] = None
    assert validate_runner_config(cfg)["max_steps"] is None

    cfg["max_steps"] = "7"
    assert validate_runner_config(cfg)["max_steps"] == 7


def test_processed_native_max_steps_is_preserved_or_explicitly_overridden():
    from omegaconf import OmegaConf

    processed = OmegaConf.create({"eval_cfg": {"max_steps": 321}})
    single_eval.apply_max_steps_override(processed, None)
    assert processed.eval_cfg.max_steps == 321

    single_eval.apply_max_steps_override(processed, 7)
    assert processed.eval_cfg.max_steps == 7


def test_single_eval_cli_max_steps_overrides_nullable_yaml(monkeypatch):
    captured = []
    monkeypatch.setattr(
        single_eval,
        "load_runner_config",
        lambda _path: {"max_steps": None},
    )
    monkeypatch.setattr(
        single_eval,
        "run_eval",
        lambda config: captured.append(config) or 0,
    )

    assert single_eval.main(["--max-steps", "9"]) == 0
    assert captured == [{"max_steps": 9}]


def test_policy_config_leaves_native_max_steps_unset():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "robodojo"
        / "policy_config.yml"
    )
    assert single_eval.load_runner_config(config_path)["max_steps"] is None


ISAACLAB_SOURCE_PACKAGES = (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_contrib",
    "isaaclab_mimic",
    "isaaclab_rl",
    "isaaclab_tasks",
)
PROVENANCE_MODULES = (
    "benchmarks",
    "openwam",
    "env",
    "task",
    "utils",
    "XPolicyLab",
    "client_server",
    "src",
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_tasks",
    "curobo",
)


def _write_runtime_package(root: Path, package: str) -> Path:
    package_dir = root / package
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        f"ORIGIN = {str(package_dir)!r}\n",
        encoding="utf-8",
    )
    return package_dir


def _make_runtime_checkouts(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    openwam_root = tmp_path / "OpenWAM"
    robodojo_root = tmp_path / "RoboDojo"
    openwam_root.mkdir()
    robodojo_root.mkdir()
    for package in ("benchmarks", "openwam"):
        _write_runtime_package(openwam_root, package)
    for package in ("env", "task", "utils", "XPolicyLab", "src"):
        _write_runtime_package(robodojo_root, package)
    _write_runtime_package(robodojo_root / "XPolicyLab", "client_server")

    source_roots = []
    isaac_source = robodojo_root / "third_party" / "IsaacLab" / "source"
    for package in ISAACLAB_SOURCE_PACKAGES:
        root = isaac_source / package
        _write_runtime_package(root, package)
        source_roots.append(root)
    curobo_root = robodojo_root / "third_party" / "curobo"
    _write_runtime_package(curobo_root, "curobo")
    source_roots.append(curobo_root)
    return openwam_root, robodojo_root, source_roots


def test_runtime_import_paths_prioritize_all_configured_source_roots(
    tmp_path: Path, monkeypatch
):
    openwam_root, robodojo_root, source_roots = _make_runtime_checkouts(tmp_path)
    xpolicylab_root = robodojo_root / "XPolicyLab"
    monkeypatch.setattr(
        sys,
        "path",
        [
            "/other",
            "__editable__.nvidia_curobo-0.dev.finder.__path_hook__",
            "__editable__.xpolicylab-0.dev.finder.__path_hook__",
            str(robodojo_root),
            str(openwam_root),
            str(robodojo_root),
        ],
    )
    configure_runtime_import_paths(openwam_root, robodojo_root)
    expected = [
        str(openwam_root.resolve()),
        str(robodojo_root.resolve()),
        str(xpolicylab_root.resolve()),
        *(str(root.resolve()) for root in source_roots),
    ]
    assert sys.path[: len(expected)] == expected
    assert sys.path.count(str(openwam_root.resolve())) == 1
    assert sys.path.count(str(robodojo_root.resolve())) == 1
    assert sys.path.count(str(xpolicylab_root.resolve())) == 1
    assert not any("__editable__." in entry for entry in sys.path)


def test_runtime_import_provenance_accepts_only_configured_checkout(
    tmp_path: Path, monkeypatch
):
    openwam_root, robodojo_root, _ = _make_runtime_checkouts(tmp_path)
    for name in PROVENANCE_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", ["/old-editable"])

    configure_runtime_import_paths(openwam_root, robodojo_root)
    origins = verify_runtime_import_provenance(openwam_root, robodojo_root)

    assert set(origins) == set(PROVENANCE_MODULES)
    for name in ("benchmarks", "openwam"):
        assert Path(origins[name]).is_relative_to(openwam_root)
    for name in ("env", "task", "utils", "XPolicyLab", "src"):
        assert Path(origins[name]).is_relative_to(robodojo_root)
    assert Path(origins["client_server"]).is_relative_to(
        robodojo_root / "XPolicyLab"
    )
    for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "curobo"):
        assert Path(origins[name]).is_relative_to(robodojo_root / "third_party")


def test_runtime_import_provenance_rejects_loaded_old_editable(
    tmp_path: Path, monkeypatch
):
    openwam_root, robodojo_root, _ = _make_runtime_checkouts(tmp_path)
    stale_root = tmp_path / "old-editable"
    stale_root.mkdir()
    for name in PROVENANCE_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    configure_runtime_import_paths(openwam_root, robodojo_root)
    monkeypatch.setitem(
        sys.modules,
        "env",
        SimpleNamespace(__file__=str(stale_root / "env" / "__init__.py")),
    )

    with pytest.raises(RuntimeError, match="env.*configured RoboDojo checkout"):
        verify_runtime_import_provenance(openwam_root, robodojo_root)


def test_runtime_import_provenance_rejects_stale_loaded_openwam_package(
    tmp_path: Path, monkeypatch
):
    openwam_root, robodojo_root, _ = _make_runtime_checkouts(tmp_path)
    stale_root = tmp_path / "stale-OpenWAM"
    configure_runtime_import_paths(openwam_root, robodojo_root)
    monkeypatch.setitem(
        sys.modules,
        "benchmarks",
        SimpleNamespace(__file__=str(stale_root / "benchmarks" / "__init__.py")),
    )

    with pytest.raises(RuntimeError, match="benchmarks.*configured OpenWAM checkout"):
        verify_runtime_import_provenance(openwam_root, robodojo_root)


def test_runtime_import_provenance_rejects_mixed_filesystem_namespace(
    tmp_path: Path, monkeypatch
):
    openwam_root, robodojo_root, _ = _make_runtime_checkouts(tmp_path)
    stale_root = tmp_path / "stale-editable"
    (robodojo_root / "env" / "__init__.py").unlink()
    (stale_root / "env").mkdir(parents=True)
    for name in PROVENANCE_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", [str(stale_root)])
    configure_runtime_import_paths(openwam_root, robodojo_root)

    with pytest.raises(RuntimeError, match="env.*mixed.*stale-editable"):
        verify_runtime_import_provenance(openwam_root, robodojo_root)


def test_real_configured_checkout_provenance_excludes_stale_luminis_release():
    openwam_root = Path(__file__).resolve().parents[1]
    configured = os.environ.get("ROBODOJO_ROOT")
    if not configured:
        pytest.skip("ROBODOJO_ROOT is unset")
    robodojo_root = Path(configured)
    if not robodojo_root.is_dir():
        pytest.skip("configured RoboDojo checkout is unavailable")
    code = (
        "import json;"
        "from benchmarks.robodojo.single_eval import "
        "configure_runtime_import_paths,verify_runtime_import_provenance;"
        f"configure_runtime_import_paths({str(openwam_root)!r},{str(robodojo_root)!r});"
        f"print(json.dumps(verify_runtime_import_provenance("
        f"{str(openwam_root)!r},{str(robodojo_root)!r}),sort_keys=True))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=openwam_root,
        check=True,
        capture_output=True,
        text=True,
    )
    origins = json.loads(result.stdout)
    assert set(PROVENANCE_MODULES) <= set(origins)
    assert all(
        str(Path(origin).resolve()).startswith(
            (str(openwam_root.resolve()), str(robodojo_root.resolve()))
        )
        for origin in origins.values()
    )
    assert "luminis_release" not in result.stdout


def test_launcher_runtime_masks_physical_gpu_and_enables_native_extensions():
    environment = {"CUDA_VISIBLE_DEVICES": "old"}
    kwargs = prepare_launcher_runtime(
        {"device_id": 7, "headless": True},
        environ=environment,
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert kwargs["device"] == "cuda:0"
    assert kwargs["headless"] is True
    assert kwargs["enable_cameras"] is True
    assert kwargs["kit_args"].split() == [
        "--enable",
        "isaacsim.replicator.behavior",
        "--enable",
        "isaacsim.sensors.camera",
    ]


def test_isolate_launcher_argv_drops_openwam_cli_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmarks.robodojo.single_eval",
            "--config",
            "benchmarks/robodojo/policy_config.yml",
            "--calibration-output",
            "/tmp/calib.json",
        ],
    )
    preserved = single_eval.isolate_launcher_argv()
    assert preserved[0].endswith("single_eval")
    assert "--calibration-output" in preserved
    assert sys.argv == ["benchmarks.robodojo.single_eval"]


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("Articulation: {}\n", False),
        ("Articulation:\n  cabinet: {}\n", True),
        ("Other: true\n", False),
        ("not: [valid\n", True),
    ],
)
def test_physx_monitor_detection_matches_articulation_yaml(
    tmp_path: Path,
    contents: str,
    expected: bool,
):
    config_path = tmp_path / "task.yml"
    config_path.write_text(contents, encoding="utf-8")
    assert single_eval.physx_monitor_needed(config_path) is expected


def test_physx_monitor_detection_fails_safe_for_missing_yaml(tmp_path: Path):
    assert single_eval.physx_monitor_needed(tmp_path / "missing.yml") is True


def test_start_physx_monitor_loads_and_starts_only_when_needed(tmp_path: Path):
    enabled_path = tmp_path / "enabled.yml"
    enabled_path.write_text("Articulation:\n  object: {}\n", encoding="utf-8")
    disabled_path = tmp_path / "disabled.yml"
    disabled_path.write_text("Articulation: {}\n", encoding="utf-8")
    events = []

    class Monitor:
        def start(self, *, enabled):
            events.append(("start", enabled))

    module = SimpleNamespace(
        PhysXBrokenError=FakePhysXBrokenError,
        PhysXFatalError=FakePhysXFatalError,
        get_monitor=lambda: Monitor(),
    )

    def loader(name):
        events.append(("load", name))
        return module

    runtime = single_eval.start_physx_monitor(enabled_path, module_loader=loader)
    assert runtime.enabled is True
    assert runtime.monitor is not None
    assert events == [
        ("load", "src.eval_client.physx_warning_monitor"),
        ("start", True),
    ]

    events.clear()
    runtime = single_eval.start_physx_monitor(disabled_path, module_loader=loader)
    assert runtime.enabled is False
    assert runtime.monitor is None
    assert events == []


def test_stdio_fd_preservation_restores_and_closes_in_order():
    events = []

    def duplicate(target):
        events.append(("dup", target))
        return target + 10

    def duplicate_to(saved, target):
        events.append(("dup2", saved, target))

    def close(saved):
        events.append(("close", saved))

    with single_eval.preserve_stdio_fds(
        dup_fn=duplicate,
        dup2_fn=duplicate_to,
        close_fn=close,
    ):
        events.append("body")

    assert events == [
        ("dup", 1),
        ("dup", 2),
        "body",
        ("dup2", 11, 1),
        ("close", 11),
        ("dup2", 12, 2),
        ("close", 12),
    ]


def test_stdio_fd_preservation_closes_partial_duplicates():
    events = []

    def duplicate(target):
        events.append(("dup", target))
        if target == 2:
            raise OSError("dup stderr failed")
        return 11

    with pytest.raises(RuntimeError, match="preserve.*fd 2"):
        with single_eval.preserve_stdio_fds(
            dup_fn=duplicate,
            dup2_fn=lambda *_: None,
            close_fn=lambda saved: events.append(("close", saved)),
        ):
            pytest.fail("body must not run")

    assert events == [("dup", 1), ("dup", 2), ("close", 11)]


def test_stdio_restore_failure_does_not_mask_primary_and_closes_all():
    closed = []

    def restore(saved, target):
        if target == 1:
            raise OSError("stdout restore failed")

    primary = ValueError("primary failure")
    with pytest.raises(ValueError) as raised:
        with single_eval.preserve_stdio_fds(
            dup_fn=lambda target: target + 10,
            dup2_fn=restore,
            close_fn=closed.append,
        ):
            raise primary

    assert raised.value is primary
    assert closed == [11, 12]


def test_stdio_restore_failure_without_primary_is_explicit():
    closed = []

    with pytest.raises(RuntimeError, match="restore.*fd 1"):
        with single_eval.preserve_stdio_fds(
            dup_fn=lambda target: target + 10,
            dup2_fn=lambda _saved, target: (
                (_ for _ in ()).throw(OSError("restore failed"))
                if target == 1
                else None
            ),
            close_fn=closed.append,
        ):
            pass

    assert closed == [11, 12]


def test_real_monitor_shutdown_then_reexec_has_healthy_stdio():
    openwam_root = Path(__file__).resolve().parents[1]
    configured = os.environ.get("ROBODOJO_ROOT")
    if not configured:
        pytest.skip("ROBODOJO_ROOT is unset")
    robodojo_root = Path(configured)
    if not robodojo_root.is_dir():
        pytest.skip("configured RoboDojo checkout is unavailable")
    task_config = (
        robodojo_root / "task" / "RoboDojo" / "config" / "press_by_number.yml"
    )
    code = f"""
import os
from benchmarks.robodojo import single_eval
single_eval.configure_runtime_import_paths(
    {str(openwam_root)!r},
    {str(robodojo_root)!r},
)
with single_eval._physx_runtime_session({str(task_config)!r}) as runtime:
    assert runtime.enabled
os.execv(
    {sys.executable!r},
    [{sys.executable!r}, "-c", "print('monitor-reexec-stdio-ok', flush=True)"],
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=openwam_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "monitor-reexec-stdio-ok" in result.stdout


def test_eval_config_overrides_propagate_physx_monitor_flag(tmp_path: Path):
    config = valid_runner_config(tmp_path)
    overrides = single_eval.build_eval_config_overrides(
        config,
        physx_monitor_enabled=True,
    )
    assert overrides["physx_monitor_enabled"] is True
    assert overrides["num_envs"] == 1
    assert overrides["policy_name"] == "openwam"


class FakeUnStableError(Exception):
    pass


class FakePhysXBrokenError(Exception):
    def __init__(self, broken_envs):
        self.broken_envs = set(broken_envs)
        super().__init__(f"broken {sorted(self.broken_envs)}")


class FakePhysXFatalError(Exception):
    pass


class FakePhysXMonitor:
    def __init__(
        self,
        *,
        shell_restart=False,
        fatal=False,
        broken_envs=(),
        fatal_message="monitor fatal",
    ):
        self.reset_count = 0
        self.shell_restart = shell_restart
        self.fatal = fatal
        self.broken_envs = set(broken_envs)
        self.fatal_message = fatal_message

    def reset(self):
        self.reset_count += 1

    def requires_shell_restart(self):
        return self.shell_restart

    def is_fatal(self):
        return self.fatal

    def get_fatal_message(self):
        return self.fatal_message

    def get_broken_envs(self):
        return set(self.broken_envs)


class FakeSeedManager:
    def __init__(self, batches, events):
        self.batches = iter(batches)
        self.events = events

    def get_seeds(self, max_count=None):
        self.events.append(("get_seeds", max_count))
        return next(self.batches)

    def eval_step(self):
        self.events.append("eval_step")


class FakeNativeEnv:
    def __init__(self, batches, *, unstable_resets=0, protocol_error=None):
        self.events = []
        self.seed_manager = FakeSeedManager(batches, self.events)
        self.success_nums = 0
        self.fail_nums = 0
        self.unstable_resets = unstable_resets
        self.protocol_error = protocol_error

    def reset(self, seed):
        self.events.append(("reset", list(seed)))
        if self.unstable_resets:
            self.unstable_resets -= 1
            raise FakeUnStableError("unstable layout")

    def run_eval(self):
        self.events.append("run_eval")
        if self.protocol_error is not None:
            raise self.protocol_error
        self.success_nums += 1

    def close(self):
        self.events.append("close")


class FakePhysXEnv(FakeNativeEnv):
    def __init__(self, batches, outcomes):
        super().__init__(batches)
        self.outcomes = iter(outcomes)
        self.abandoned_seeds = set()
        self.current_env_seed_map = {}
        self.persisted_restart_counts = []

    def reset(self, seed):
        self.events.append(("reset", list(seed)))
        self.current_env_seed_map = {0: seed[0]}

    def run_eval(self):
        self.events.append("run_eval")
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        self.success_nums += 1

    def get_seeds_for_envs(self, env_idxs):
        return {
            self.current_env_seed_map[index]
            for index in env_idxs
            if index in self.current_env_seed_map
        }

    def persist_resume_manifest(self, *, restart_count):
        self.events.append(("persist", restart_count))
        self.persisted_restart_counts.append(restart_count)


class FakeGenericPhysXEnv(FakePhysXEnv):
    def __init__(self, batches, *, reset_outcomes, run_outcomes):
        super().__init__(batches, run_outcomes)
        self.reset_outcomes = iter(reset_outcomes)
        self.num_envs = 1

    def reset(self, seed):
        self.events.append(("reset", list(seed)))
        self.current_env_seed_map = {0: seed[0]}
        outcome = next(self.reset_outcomes, None)
        if isinstance(outcome, BaseException):
            raise outcome


def test_native_runner_skips_unstable_seed_then_calibrates_after_successful_reset():
    env = FakeNativeEnv([[10], [11]], unstable_resets=1)

    _run_native_episodes(
        env,
        1,
        unstable_error=FakeUnStableError,
        calibration_callback=lambda current_env: current_env.events.append(
            "calibration"
        ),
    )

    assert env.success_nums == 1
    assert env.events == [
        ("get_seeds", 1),
        ("reset", [10]),
        "eval_step",
        "close",
        ("get_seeds", 1),
        ("reset", [11]),
        "calibration",
        "run_eval",
        "eval_step",
    ]


def test_native_runner_fails_when_seeds_exhaust_before_completion():
    env = FakeNativeEnv([None])

    with pytest.raises(RuntimeError, match="exhausted.*1"):
        _run_native_episodes(
            env,
            1,
            unstable_error=FakeUnStableError,
        )


def test_native_runner_calibration_callback_runs_only_once():
    env = FakeNativeEnv([[1], [2]])

    _run_native_episodes(
        env,
        2,
        unstable_error=FakeUnStableError,
        calibration_callback=lambda current_env: current_env.events.append(
            "calibration"
        ),
    )

    assert env.events.count("calibration") == 1
    first_reset = env.events.index(("reset", [1]))
    calibration = env.events.index("calibration")
    first_run = env.events.index("run_eval")
    assert first_reset < calibration < first_run


def test_native_runner_treats_eval_count_as_total_completed_target():
    env = FakeNativeEnv([[2], [3]])
    env.success_nums = 1

    _run_native_episodes(
        env,
        2,
        unstable_error=FakeUnStableError,
    )

    assert env.success_nums == 2
    assert env.events.count("run_eval") == 1
    assert ("get_seeds", 1) in env.events


def test_native_runner_does_not_swallow_protocol_errors():
    protocol_error = ConnectionError("policy protocol failed")
    env = FakeNativeEnv([[5]], protocol_error=protocol_error)

    with pytest.raises(ConnectionError, match="policy protocol failed"):
        _run_native_episodes(
            env,
            1,
            unstable_error=FakeUnStableError,
        )
    assert "eval_step" not in env.events


def test_native_runner_abandons_physx_broken_seed_and_continues():
    env = FakePhysXEnv(
        [[31], [32]],
        [FakePhysXBrokenError({0}), "success"],
    )
    monitor = FakePhysXMonitor()

    _run_native_episodes(
        env,
        1,
        unstable_error=FakeUnStableError,
        physx_broken_error=FakePhysXBrokenError,
        physx_fatal_error=FakePhysXFatalError,
        monitor=monitor,
    )

    assert env.success_nums == 1
    assert env.abandoned_seeds == {31}
    assert monitor.reset_count == 2
    assert env.events == [
        ("get_seeds", 1),
        ("reset", [31]),
        "run_eval",
        "close",
        ("get_seeds", 1),
        ("reset", [32]),
        "run_eval",
        "eval_step",
    ]


@pytest.mark.parametrize("shell_restart", [False, True])
def test_native_runner_persists_then_surfaces_distinct_physx_fatal_restart(
    shell_restart,
):
    env = FakePhysXEnv([[41]], [FakePhysXFatalError("kernel failed")])
    monitor = FakePhysXMonitor(shell_restart=shell_restart)

    with pytest.raises(single_eval.PhysXRestartRequired) as raised:
        _run_native_episodes(
            env,
            1,
            unstable_error=FakeUnStableError,
            physx_broken_error=FakePhysXBrokenError,
            physx_fatal_error=FakePhysXFatalError,
            monitor=monitor,
            fatal_restart_count=2,
        )

    assert env.persisted_restart_counts == [2]
    assert raised.value.restart_count == 2
    assert raised.value.shell_restart is shell_restart
    assert env.events[-1] == ("persist", 2)


def test_physx_monitor_does_not_convert_protocol_errors_to_recovery():
    protocol_error = ConnectionError("OpenWAM response failed")
    env = FakeNativeEnv([[9]], protocol_error=protocol_error)
    monitor = FakePhysXMonitor()

    with pytest.raises(ConnectionError, match="OpenWAM response failed"):
        _run_native_episodes(
            env,
            1,
            unstable_error=FakeUnStableError,
            physx_broken_error=FakePhysXBrokenError,
            physx_fatal_error=FakePhysXFatalError,
            monitor=monitor,
        )
    assert monitor.reset_count == 1


@pytest.mark.parametrize("phase", ["reset", "run_eval"])
def test_generic_exception_uses_monitor_fatal_backstop(phase):
    generic = RuntimeError(f"generic during {phase}")
    env = FakeGenericPhysXEnv(
        [[51]],
        reset_outcomes=[generic] if phase == "reset" else [None],
        run_outcomes=[generic] if phase == "run_eval" else [],
    )
    monitor = FakePhysXMonitor(fatal=True, fatal_message="GPU solver died")

    with pytest.raises(single_eval.PhysXRestartRequired, match="GPU solver died"):
        _run_native_episodes(
            env,
            1,
            unstable_error=FakeUnStableError,
            physx_broken_error=FakePhysXBrokenError,
            physx_fatal_error=FakePhysXFatalError,
            monitor=monitor,
            fatal_restart_count=3,
        )

    assert env.persisted_restart_counts == [3]


@pytest.mark.parametrize("phase", ["reset", "run_eval"])
def test_generic_exception_uses_monitor_broken_env_backstop(phase):
    generic = RuntimeError(f"generic during {phase}")
    env = FakeGenericPhysXEnv(
        [[61], [62]],
        reset_outcomes=[generic, None] if phase == "reset" else [None, None],
        run_outcomes=[generic, "success"]
        if phase == "run_eval"
        else ["success"],
    )
    monitor = FakePhysXMonitor(broken_envs={0, 99})

    _run_native_episodes(
        env,
        1,
        unstable_error=FakeUnStableError,
        physx_broken_error=FakePhysXBrokenError,
        physx_fatal_error=FakePhysXFatalError,
        monitor=monitor,
    )

    assert env.abandoned_seeds == {61}
    assert env.success_nums == 1
    assert env.events.count("close") == 1


@pytest.mark.parametrize("phase", ["reset", "run_eval"])
def test_generic_exception_with_clean_monitor_stays_fail_fast(phase):
    generic = ConnectionError(f"protocol during {phase}")
    env = FakeGenericPhysXEnv(
        [[71]],
        reset_outcomes=[generic] if phase == "reset" else [None],
        run_outcomes=[generic] if phase == "run_eval" else [],
    )

    with pytest.raises(ConnectionError) as raised:
        _run_native_episodes(
            env,
            1,
            unstable_error=FakeUnStableError,
            physx_broken_error=FakePhysXBrokenError,
            physx_fatal_error=FakePhysXFatalError,
            monitor=FakePhysXMonitor(),
        )
    assert raised.value is generic


@pytest.mark.parametrize("phase", ["reset", "run_eval"])
def test_generic_exception_with_monitor_disabled_stays_fail_fast(phase):
    generic = RuntimeError(f"general during {phase}")
    env = FakeGenericPhysXEnv(
        [[81]],
        reset_outcomes=[generic] if phase == "reset" else [None],
        run_outcomes=[generic] if phase == "run_eval" else [],
    )

    with pytest.raises(RuntimeError) as raised:
        _run_native_episodes(
            env,
            1,
            unstable_error=FakeUnStableError,
            monitor=None,
        )
    assert raised.value is generic


def test_inprocess_restart_preserves_run_id_and_module_invocation():
    restart = single_eval.PhysXRestartRequired(
        "kernel failed",
        restart_count=2,
        shell_restart=False,
    )
    environment = {"ROBODOJO_RUN_ID": "fixed-run"}
    calls = []

    result = single_eval.handle_physx_restart(
        restart,
        ["--config", "/tmp/policy.yml"],
        environ=environment,
        executable="/python",
        execv=lambda executable, argv: calls.append((executable, argv)),
    )

    assert result == 99
    assert environment == {
        "ROBODOJO_RUN_ID": "fixed-run",
        "ROBODOJO_FATAL_RESTART_COUNT": "2",
    }
    assert calls == [
        (
            "/python",
            [
                "/python",
                "-m",
                "benchmarks.robodojo.single_eval",
                "--config",
                "/tmp/policy.yml",
            ],
        )
    ]


@pytest.mark.parametrize(
    "restart",
    [
        pytest.param(
            lambda: single_eval.PhysXRestartRequired(
                "shell restart",
                restart_count=1,
                shell_restart=True,
            ),
            id="shell-restart",
        ),
        pytest.param(
            lambda: single_eval.PhysXRestartRequired(
                "restart cap",
                restart_count=4,
                shell_restart=False,
            ),
            id="restart-cap",
        ),
    ],
)
def test_shell_restart_or_cap_returns_99_without_exec(restart):
    calls = []
    result = single_eval.handle_physx_restart(
        restart(),
        [],
        environ={"ROBODOJO_RUN_ID": "fixed"},
        executable="/python",
        execv=lambda *args: calls.append(args),
        max_inprocess_restarts=3,
    )
    assert result == 99
    assert calls == []


def _resume_eval_cfg() -> dict:
    return {
        "task_name": "stack_blocks",
        "policy_name": "openwam",
        "config_name": "arx_x5",
        "seed": 7,
        "additional_info": "review",
    }


def test_resume_manifest_path_matches_current_upstream_layout():
    path = single_eval.resume_manifest_path(
        _resume_eval_cfg(),
        "run-123",
        benchmark="RoboDojo",
    )
    assert path == Path(
        "eval_result/RoboDojo/stack_blocks/openwam/arx_x5/"
        "7_review/_resume_run-123.json"
    )


def test_resume_manifest_load_and_best_effort_delete(tmp_path: Path):
    manifest = tmp_path / "resume.json"
    payload = {
        "run_id": "run-123",
        "success_nums": 2,
        "fail_nums": 1,
        "abandoned_layout_ids": [9],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert single_eval.load_resume_manifest(manifest) == payload
    assert single_eval.load_resume_manifest(tmp_path / "missing.json") is None

    env = SimpleNamespace(resume_manifest_path=lambda: str(manifest))
    assert single_eval.delete_resume_manifest(env) is True
    assert not manifest.exists()
    assert single_eval.delete_resume_manifest(env) is False


def test_construct_eval_env_loads_resume_before_create_and_restores_patch(
    tmp_path: Path,
):
    env_cfg = SimpleNamespace(eval_cfg=_resume_eval_cfg())
    run_id = "resume-wiring"
    path = tmp_path / single_eval.resume_manifest_path(
        env_cfg.eval_cfg,
        run_id,
        benchmark="RoboDojo",
    )
    path.parent.mkdir(parents=True)
    resume_state = {"success_nums": 3, "abandoned_layout_ids": [12]}
    path.write_text(json.dumps(resume_state), encoding="utf-8")
    events = []
    owner = SimpleNamespace(WsModelClient=OriginalNetworkClient)

    def create_eval_env(config, app, *, resume_state):
        events.append(("create", config, app, copy.deepcopy(resume_state)))
        return SimpleNamespace(model_client=owner.WsModelClient("no-network"))

    owner.create_eval_env = create_eval_env
    with runtime_working_directory(tmp_path):
        env = single_eval.construct_eval_env_with_resume(
            owner,
            env_cfg,
            object(),
            run_id=run_id,
            benchmark="RoboDojo",
        )

    assert events[0][0] == "create"
    assert events[0][3] == resume_state
    assert owner.WsModelClient is OriginalNetworkClient
    assert env.model_client is None


def test_runtime_working_directory_restores_cwd_on_failure(tmp_path: Path):
    original = Path.cwd()

    with pytest.raises(RuntimeError, match="inside"):
        with runtime_working_directory(tmp_path):
            assert Path.cwd() == tmp_path.resolve()
            raise RuntimeError("inside")

    assert Path.cwd() == original


def test_smoke_shell_uses_yaml_endpoint_unless_environment_overrides(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks" / "robodojo" / "run_smoke.sh"
    stub = tmp_path / "python-stub"
    stub.write_text(
        "#!/usr/bin/env bash\nprintf 'cwd=%s\\n' \"$PWD\"\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    caller = tmp_path / "stale" / "OpenWAM"
    caller.mkdir(parents=True)
    environment = os.environ.copy()
    environment["ROBODOJO_PYTHON"] = str(stub)
    for key in ("OPENWAM_HOST", "OPENWAM_PORT", "OPENWAM_TIMEOUT"):
        environment.pop(key, None)

    inherited = subprocess.run(
        ["bash", str(script), "ping"],
        cwd=caller,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert f"cwd={repo_root}" in inherited
    assert "--host" not in inherited
    assert "--port" not in inherited
    assert "--timeout" not in inherited

    environment.update(
        OPENWAM_HOST="policy.example",
        OPENWAM_PORT="9988",
        OPENWAM_TIMEOUT="45",
    )
    overridden = subprocess.run(
        ["bash", str(script), "ping"],
        cwd=caller,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert overridden[overridden.index("--host") + 1] == "policy.example"
    assert overridden[overridden.index("--port") + 1] == "9988"
    assert overridden[overridden.index("--timeout") + 1] == "45"


def test_smoke_shell_resolves_relative_inputs_against_caller_cwd(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks" / "robodojo" / "run_smoke.sh"
    caller = tmp_path / "caller"
    caller.mkdir()
    stub = tmp_path / "path-stub"
    stub.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        ROBODOJO_PYTHON=str(stub),
        ROBODOJO_ROOT="external/RoboDojo",
    )

    output = subprocess.run(
        [
            "bash",
            str(script),
            "contract",
            "calibration/live.json",
            "datasets/robodojo",
            "stack_blocks",
        ],
        cwd=caller,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert output[output.index("--calibration") + 1] == str(
        caller / "calibration" / "live.json"
    )
    assert output[output.index("--dataset-root") + 1] == str(
        caller / "datasets" / "robodojo"
    )
    assert output[output.index("--robodojo-root") + 1] == str(
        caller / "external" / "RoboDojo"
    )


@pytest.mark.parametrize("retry_code", [99, 134, 139])
def test_isaac_smoke_preserves_run_id_and_retries_native_failures(
    tmp_path: Path,
    retry_code: int,
):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks" / "robodojo" / "run_smoke.sh"
    state = tmp_path / f"attempt-{retry_code}"
    stub = tmp_path / f"isaac-stub-{retry_code}"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "count=0\n"
        "[[ -f \"${STUB_STATE}\" ]] && count=\"$(<\"${STUB_STATE}\")\"\n"
        "count=$((count + 1))\n"
        "printf '%s' \"${count}\" >\"${STUB_STATE}\"\n"
        "printf 'attempt=%s run_id=%s cwd=%s\\n' "
        "\"${count}\" \"${ROBODOJO_RUN_ID:-}\" \"$PWD\"\n"
        "if [[ \"${count}\" -eq 1 ]]; then exit \"${STUB_RETRY_CODE}\"; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        ROBODOJO_PYTHON=str(stub),
        ROBODOJO_MAX_BASH_RETRIES="1",
        ROBODOJO_RETRY_DELAY_SECONDS="0",
        STUB_STATE=str(state),
        STUB_RETRY_CODE=str(retry_code),
    )
    environment.pop("ROBODOJO_RUN_ID", None)

    result = subprocess.run(
        ["bash", str(script), "isaac", "stack_blocks"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    attempts = [
        line for line in result.stdout.splitlines() if line.startswith("attempt=")
    ]
    assert len(attempts) == 2
    run_ids = {line.split("run_id=", 1)[1].split(" ", 1)[0] for line in attempts}
    assert len(run_ids) == 1
    assert run_ids != {""}
    assert all(f"cwd={repo_root}" in line for line in attempts)


@pytest.mark.parametrize(("retries", "expected_attempts"), [(0, 1), (2, 3)])
def test_isaac_smoke_retry_limit_counts_retries_after_initial_launch(
    tmp_path: Path,
    retries: int,
    expected_attempts: int,
):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks" / "robodojo" / "run_smoke.sh"
    state = tmp_path / f"retry-count-{retries}"
    stub = tmp_path / f"always-retry-{retries}"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "count=0\n"
        "[[ -f \"${STUB_STATE}\" ]] && count=\"$(<\"${STUB_STATE}\")\"\n"
        "printf '%s' \"$((count + 1))\" >\"${STUB_STATE}\"\n"
        "exit 99\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        ROBODOJO_PYTHON=str(stub),
        ROBODOJO_MAX_BASH_RETRIES=str(retries),
        ROBODOJO_RETRY_DELAY_SECONDS="0",
        STUB_STATE=str(state),
    )

    result = subprocess.run(
        ["bash", str(script), "isaac", "stack_blocks"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 99
    assert int(state.read_text(encoding="utf-8")) == expected_attempts


def test_non_isaac_smoke_failure_is_not_retried(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks" / "robodojo" / "run_smoke.sh"
    state = tmp_path / "attempt"
    stub = tmp_path / "failing-stub"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf x >>\"${STUB_STATE}\"\n"
        "exit 99\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        ROBODOJO_PYTHON=str(stub),
        ROBODOJO_MAX_BASH_RETRIES="5",
        ROBODOJO_RETRY_DELAY_SECONDS="0",
        STUB_STATE=str(state),
    )

    result = subprocess.run(
        ["bash", str(script), "ping"],
        cwd=tmp_path,
        env=environment,
        check=False,
    )
    assert result.returncode == 99
    assert state.read_text(encoding="utf-8") == "x"


def test_isaac_shell_smoke_defaults_to_three_steps_and_allows_override(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks" / "robodojo" / "run_smoke.sh"
    stub = tmp_path / "isaac-args-stub"
    stub.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    environment = os.environ.copy()
    environment["ROBODOJO_PYTHON"] = str(stub)
    environment.pop("ROBODOJO_ISAAC_STEPS", None)

    default_args = subprocess.run(
        ["bash", str(script), "isaac", "stack_blocks"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert default_args[default_args.index("--steps") + 1] == "3"

    environment["ROBODOJO_ISAAC_STEPS"] = "8"
    override_args = subprocess.run(
        ["bash", str(script), "isaac", "stack_blocks"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert override_args[override_args.index("--steps") + 1] == "8"


def test_isaac_smoke_forwards_positive_steps_as_max_steps(monkeypatch):
    captured = []
    monkeypatch.setattr(
        smoke_module,
        "run_eval",
        lambda config: captured.append(config) or 0,
    )

    assert smoke_module.isaac_smoke({"eval_count": 9}, steps=5) == 0
    assert captured == [{"eval_count": 1, "max_steps": 5}]


@pytest.mark.parametrize("steps", [True, False, 0, -1, 1.5])
def test_isaac_smoke_rejects_nonpositive_or_noninteger_steps(steps):
    with pytest.raises(ValueError, match="steps.*positive integer"):
        smoke_module.isaac_smoke({}, steps=steps)


def test_isaac_mode_cli_forwards_steps_without_changing_debug(monkeypatch):
    calls = []
    monkeypatch.setattr(smoke_module, "load_runner_config", lambda _path: {})
    monkeypatch.setattr(
        smoke_module,
        "isaac_smoke",
        lambda config, *, steps: calls.append((config, steps)) or 0,
    )

    assert smoke_module.main(["--mode", "isaac", "--steps", "6"]) == 0
    assert calls == [({}, 6)]


def test_isaac_smoke_converts_fatal_restart_signal_to_exit_99(monkeypatch):
    def fatal(_config):
        raise single_eval.PhysXRestartRequired(
            "fatal GPU",
            restart_count=1,
            shell_restart=True,
        )

    monkeypatch.setattr(smoke_module, "run_eval", fatal)
    assert smoke_module.isaac_smoke({"eval_count": 9}) == 99


def test_importing_runner_and_smoke_does_not_import_isaac():
    assert "isaaclab" not in sys.modules
    assert "isaacsim" not in sys.modules
    assert "src.eval_client.eval_env" not in sys.modules


def test_smoke_synthetic_observation_and_fake_transport_rollout_without_isaac():
    calibration = valid_calibration()

    class FakeDebugEnv:
        def __init__(self):
            self.actions = []
            self.resets = 0
            self.episode_step = 0

        def get_obs(self):
            obs = valid_observation(instruction=b"Debug instruction")
            obs["state"]["left_ee_pose"] = np.ones(7)
            obs["state"]["right_ee_pose"] = np.ones(7)
            return obs

        def take_action(self, action):
            self.actions.append(action)
            self.episode_step += 1

    raw = FakeDebugEnv().get_obs()
    synthetic = make_synthetic_debug_observation(raw, calibration, env_idx=0)
    assert synthetic["instruction"] == b"Debug instruction"
    for key in ("left_ee_pose", "right_ee_pose"):
        assert np.isclose(np.linalg.norm(synthetic["state"][key][3:7]), 1.0)

    transport = FakeTransport(
        resets=[{"type": "reset_ack"}],
        actions=[{"type": "action", "action": valid_action().tolist()}],
    )
    client, _, _ = make_client(
        transport,
        calibration=calibration,
        auto_reset=False,
    )
    env = FakeDebugEnv()
    result = run_debug_protocol_rollout(
        env,
        client,
        calibration,
        steps=1,
    )
    assert result == 1
    assert len(env.actions) == 1
    assert transport.calls == ["ping", "reset", "predict_once"]
