"""RoboTwin deploy proprio extraction must match training action spaces."""

import numpy as np
import pytest

from benchmarks.robotwin import openwam2robotwin_interface as iface
from benchmarks.utils import action_conversion
from openwam.dataloader.transforms.rotation import quat_xyzw_to_rotation_6d


class _Model:
    def __init__(self, action_type: str):
        self._action_type = action_type
        self._send_state = True


def test_robotwin_endpose_conversion_matches_dataset_eef_layout():
    left = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    right = np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    eef = action_conversion.robotwin_endpose_to_eef20d(left, right, 0.25, 0.75)
    expected = np.concatenate(
        [
            left[:3],
            quat_xyzw_to_rotation_6d(left[3:]),
            np.array([0.25], dtype=np.float32),
            right[:3],
            quat_xyzw_to_rotation_6d(right[3:]),
            np.array([0.75], dtype=np.float32),
        ]
    ).astype(np.float32)

    assert eef.shape == (20,)
    np.testing.assert_allclose(eef, expected, atol=1e-6)


def test_ee_action_type_sends_20d_eef_proprio():
    observation = {
        "endpose": {
            "left_endpose": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "right_endpose": np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "left_gripper": np.array(0.25, dtype=np.float32),
            "right_gripper": np.array(0.75, dtype=np.float32),
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }

    proprio = iface._extract_proprio(_Model("ee"), observation)

    assert proprio.shape == (20,)
    np.testing.assert_allclose(proprio[[0, 1, 2, 9, 10, 11, 12, 19]], [0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.75])


def test_qpos_action_type_sends_joint_vector():
    joint = np.arange(14, dtype=np.float32)
    observation = {"joint_action": {"vector": joint}}

    proprio = iface._extract_proprio(_Model("qpos"), observation)

    assert proprio.shape == (14,)
    np.testing.assert_allclose(proprio, joint)


def test_ee_action_type_rejects_joint_only_observation():
    observation = {"joint_action": {"vector": np.zeros(14, dtype=np.float32)}}

    try:
        iface._extract_proprio(_Model("ee"), observation)
    except KeyError as exc:
        assert "action_mode='eef'" in str(exc)
    else:
        raise AssertionError("expected ee action_type to require EEF proprio")


def _make_bypassed_client(*, send_state=True, state_dim=20):
    model = iface.ModelClient.__new__(iface.ModelClient)
    model._send_state = send_state
    model._state_dim = state_dim
    model._request_timeout = 1
    model._action_indices = None
    model._action_type = "ee"
    model._task_description = "pick"
    model._debug = False
    model._debug_dir = ""
    model._episode = 0
    model._step = 0
    model._server = "http://mock"
    return model


def test_step_forwards_state_payload_and_checks_dim(monkeypatch):
    model = _make_bypassed_client(send_state=True, state_dim=20)
    captured = {}

    monkeypatch.setattr(iface.client, "encode_numpy_b64", lambda img: "jpeg")

    def _post(server, endpoint, payload, timeout):
        captured.update(server=server, endpoint=endpoint, payload=payload, timeout=timeout)
        return {"action": [0.0] * 20}

    monkeypatch.setattr(iface.client, "post", _post)

    action = model.step(
        {
            "cams": {"head": np.zeros((2, 2, 3), dtype=np.uint8), "left": None, "right": None},
            "lang": "pick",
            "state": np.arange(20, dtype=np.float32),
        }
    )

    assert action.shape == (20,)
    assert captured["endpoint"] == "/predict"
    assert captured["payload"]["state"] == [float(v) for v in range(20)]


def test_step_rejects_wrong_state_dim_before_post(monkeypatch):
    model = _make_bypassed_client(send_state=True, state_dim=20)
    monkeypatch.setattr(iface.client, "encode_numpy_b64", lambda img: "jpeg")
    monkeypatch.setattr(
        iface.client,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("post called")),
    )

    with pytest.raises(ValueError, match="Extracted state_dim=14, expected 20"):
        model.step(
            {
                "cams": {"head": np.zeros((2, 2, 3), dtype=np.uint8), "left": None, "right": None},
                "lang": "pick",
                "state": np.arange(14, dtype=np.float32),
            }
        )


def test_string_bool_false_disables_state_payload(monkeypatch):
    assert iface._parse_bool("false", default=True) is False
    model = _make_bypassed_client(send_state=False, state_dim=20)
    captured = {}

    monkeypatch.setattr(iface.client, "encode_numpy_b64", lambda img: "jpeg")

    def _post(server, endpoint, payload, timeout):
        del server, endpoint, timeout
        captured["payload"] = payload
        return {"action": [0.0] * 20}

    monkeypatch.setattr(iface.client, "post", _post)

    model.step(
        {
            "cams": {"head": np.zeros((2, 2, 3), dtype=np.uint8), "left": None, "right": None},
            "lang": "pick",
            "state": np.arange(20, dtype=np.float32),
        }
    )

    assert "state" not in captured["payload"]


def test_eval_does_not_require_proprio_when_send_state_false(monkeypatch):
    model = _Model("ee")
    model._send_state = False

    class _TaskEnv:
        take_action_cnt = 0
        step_lim = 10

        def __init__(self):
            self.action = None
            self.action_type = None

        def get_instruction(self):
            return "pick"

        def take_action(self, action, action_type="qpos"):
            self.action = action
            self.action_type = action_type

    task_env = _TaskEnv()

    def _step(example, step=0):
        assert step == 0
        assert example["state"] is None
        return np.zeros(20, dtype=np.float32)

    model.step = _step

    iface.eval(
        task_env,
        model,
        {
            "observation": {
                "head_camera": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            },
        },
    )

    assert task_env.action_type == "ee"
    assert task_env.action.shape == (16,)
