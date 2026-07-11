"""Unit tests for the RoboCasa365 eval adapter.

Pure transforms + the policy class (with an injected fake client) + the pinned
upstream contract. No sim, no GPU, no live server. Mirrors robotwin's
``test_robotwin_interface_proprio.py`` in spirit (contract assertion + fail-fast
before predict + payload forwarding).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ADAPTER_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "robocasa365"
if str(_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_DIR))

import openwam2robocasa365_interface as adapter  # noqa: E402


def _make_obs():
    return {
        "state.base_position": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "state.base_rotation": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "state.end_effector_position_relative": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "state.end_effector_rotation_relative": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        "state.gripper_qpos": np.array([0.01, 0.02], dtype=np.float32),
        "video.robot0_agentview_left": np.zeros((256, 256, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.zeros((256, 256, 3), dtype=np.uint8),
        "video.robot0_agentview_right": np.zeros((256, 256, 3), dtype=np.uint8),
        "annotation.human.task_description": "open the drawer",
    }


# --- Pinned RoboCasa365 contract (guardrail vs accidental drift; CI-safe) ---
# Transcribed from upstream (NOT vendored here, so this is a drift guardrail, not an independent
# check) — re-verify by hand against these if upstream changes:
#   robocasa/robocasa/wrappers/gym_wrapper.py  PandaOmronKeyConverter.{unmap_action, map_obs}
#   robocasa/robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py
#   starVLA examples/Robocasa_365/eval_files/model2robocasa365_interface.py
_EXPECTED_ACTION_SLICES = {
    "action.end_effector_position": (0, 3),
    "action.end_effector_rotation": (3, 6),
    "action.gripper_close": (6, 7),
    "action.base_motion": (7, 11),
    "action.control_mode": (11, 12),
}
_EXPECTED_STATE_KEYS = [
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
]


def test_pinned_robocasa_contract():
    assert adapter.ACTION_SLICES == _EXPECTED_ACTION_SLICES
    assert adapter.ACTION_DIM == 12
    assert adapter.DEFAULT_STATE_KEYS == _EXPECTED_STATE_KEYS
    # Proprio is sent as the 20-D single-arm EEF the model trains on (NOT the raw 16-D);
    # built from eef_pos_rel + eef_rot_rel + gripper_qpos (base dropped).
    assert adapter.STATE_DIM == 20
    assert adapter.PROPRIO_EEF_KEYS == (
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    )


def test_default_camera_mapping_matches_robocasa():
    import inspect

    p = inspect.signature(adapter.OpenWAMRoboCasa365Policy.__init__).parameters
    assert p["head_camera_key"].default == "video.robot0_agentview_left"
    assert p["left_wrist_camera_key"].default == "video.robot0_eye_in_hand"  # the arm's real wrist cam
    assert p["right_wrist_camera_key"].default is None  # no 2nd wrist -> server black-fills
    assert p["image_transform"].default == "none"


# --- pure transforms ---

def test_assemble_state_order_and_dim():
    state = adapter.assemble_state(_make_obs(), adapter.DEFAULT_STATE_KEYS)
    assert len(state) == 16
    assert state[:3] == pytest.approx([1.0, 2.0, 3.0])
    assert state[3:7] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert state[7:10] == pytest.approx([4.0, 5.0, 6.0])
    assert state[-2:] == pytest.approx([0.01, 0.02])


def test_assemble_state_missing_key_raises():
    obs = _make_obs()
    del obs["state.gripper_qpos"]
    with pytest.raises(KeyError):
        adapter.assemble_state(obs, adapter.DEFAULT_STATE_KEYS)


def test_slice_action_splits_twelve():
    out = adapter.slice_action(np.arange(12, dtype=np.float32))
    assert list(out["action.end_effector_position"]) == [0, 1, 2]
    assert list(out["action.end_effector_rotation"]) == [3, 4, 5]
    assert list(out["action.gripper_close"]) == [6]
    assert list(out["action.base_motion"]) == [7, 8, 9, 10]
    assert list(out["action.control_mode"]) == [11]


def test_slice_action_wrong_dim_raises():
    with pytest.raises(ValueError):
        adapter.slice_action(np.arange(7, dtype=np.float32))


def test_transform_image_none_is_passthrough():
    img = np.arange(2 * 2 * 3).reshape(2, 2, 3).astype(np.uint8)
    out = adapter.transform_image(img, "none")
    assert np.array_equal(out, img)
    assert out.dtype == np.uint8


# --- policy class with a fake WS client ---

class _FakeClient:
    def __init__(self, action):
        self._action = list(action)
        self.last_payload = None
        self.reset_called = False

    def ping(self):
        return {"type": "pong"}

    def reset(self):
        self.reset_called = True
        return {"type": "reset_ack"}

    def predict(self, payload):
        self.last_payload = payload
        return {"action": list(self._action)}

    def close(self):
        pass


class _NoPredictClient:
    """ping/reset OK, but predict must never be called (fail-fast guard)."""

    def ping(self):
        return {"type": "pong"}

    def reset(self):
        return {"type": "reset_ack"}

    def predict(self, payload):
        raise AssertionError("predict must not be called when the obs is rejected")

    def close(self):
        pass


def test_policy_act_builds_payload_and_slices():
    fake = _FakeClient(action=range(12))
    policy = adapter.OpenWAMRoboCasa365Policy(_client=fake)
    action_dict = policy.act(_make_obs(), "open the drawer")

    p = fake.last_payload
    assert set(p["images"]) == {"head_camera", "left_wrist_camera", "right_wrist_camera"}
    # default mapping: head (agentview_left) + left_wrist (eye_in_hand) sent;
    # right_wrist stays None (single-arm has no 2nd wrist -> server black-fills it)
    assert p["images"]["head_camera"] is not None
    assert p["images"]["left_wrist_camera"] is not None
    assert p["images"]["right_wrist_camera"] is None
    assert p["prompt"] == "open the drawer"
    assert len(p["state"]) == 20  # 20-D EEF proprio (not raw 16-D)
    assert p["state"][10:] == [0.0] * 10  # right arm zero-padded

    assert set(action_dict) == set(adapter.ACTION_SLICES)
    assert list(action_dict["action.base_motion"]) == [7, 8, 9, 10]
    assert list(action_dict["action.control_mode"]) == [11]


def test_policy_act_wrong_action_dim_raises():
    policy = adapter.OpenWAMRoboCasa365Policy(_client=_FakeClient(action=range(7)))
    with pytest.raises(ValueError):
        policy.act(_make_obs(), "x")


def test_policy_act_wrong_state_dim_raises_before_predict():
    # state_dim mismatch must raise BEFORE the server is hit (predict not called).
    # The client now sends 20-D EEF proprio; set an inconsistent expected dim to trip the guard.
    policy = adapter.OpenWAMRoboCasa365Policy(_client=_NoPredictClient(), state_dim=14)
    with pytest.raises(ValueError, match="state dim"):
        policy.act(_make_obs(), "x")


def test_policy_reset_checks_ack():
    fake = _FakeClient(action=range(12))
    policy = adapter.OpenWAMRoboCasa365Policy(_client=fake)
    policy.reset()
    assert fake.reset_called


def test_policy_bad_image_transform_raises():
    with pytest.raises(ValueError):
        adapter.OpenWAMRoboCasa365Policy(_client=_FakeClient(action=range(12)), image_transform="flip")


# --- 20-D EEF -> 12-D OSC bridge (eef20d_to_robocasa12d) ---

# identity rot6d = first two columns of I3.
_IDENT_R6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_bridge_pos_delta_and_order():
    """target 0.025m ahead of current with pos_scale 0.05 -> +0.5 in slot 0; output
    is in slice_action order [eef_pos, eef_rot, grip, base, mode]."""
    from benchmarks.utils import eef20d_to_robocasa12d

    eef20d = np.zeros(20, np.float32)
    eef20d[0:3] = [0.025, 0.0, 0.0]      # absolute target pos
    eef20d[3:9] = _IDENT_R6D             # no rotation
    eef20d[9] = 0.7                      # gripper separation 0.7 (>>0.05) -> OPEN
    out = eef20d_to_robocasa12d(
        eef20d, proprio_eef_pos=[0.0, 0.0, 0.0], proprio_eef_rot6d=_IDENT_R6D,
        pos_scale=0.05, rot_scale=0.5,
    )
    assert out.shape == (12,)
    assert out[0:3] == pytest.approx([0.5, 0.0, 0.0], abs=1e-5)   # eef_pos delta/scale
    assert out[3:6] == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)   # eef_rot (identity)
    assert out[6] == pytest.approx(0.0)                          # gripper OPEN (sep 0.7 >= thresh)
    assert out[7:11] == pytest.approx([0.0, 0.0, 0.0, 0.0])      # base_motion default 0
    assert out[11] == pytest.approx(-1.0)                        # control_mode default -1
    # slices line up with the env adapter's ACTION_SLICES
    assert adapter.slice_action(out)["action.control_mode"][0] == pytest.approx(-1.0)


def test_bridge_gripper_binarize_and_invert():
    """Model gripper = finger separation (large=open); env binarizes gripper_close at 0.5
    (-1 open/+1 close). Bridge must map small sep -> close (>=0.5), large sep -> open (<0.5)."""
    from benchmarks.utils import eef20d_to_robocasa12d

    def _grip(sep):
        a = np.zeros(20, np.float32)
        a[3:9] = _IDENT_R6D
        a[9] = sep
        out = eef20d_to_robocasa12d(a, proprio_eef_pos=[0, 0, 0], proprio_eef_rot6d=_IDENT_R6D,
                                    pos_scale=0.05, rot_scale=0.5)
        return float(out[6])

    # default threshold 0.05: empirical closed-sep ~0.034 -> close; open-sep ~0.078 -> open
    assert _grip(0.034) >= 0.5    # closing -> env reads >=0.5 -> close
    assert _grip(0.078) < 0.5     # open -> env reads <0.5 -> open
    # raw pass-through (the old bug) would give 0.034/0.078 — both <0.5 -> gripper NEVER closes
    assert _grip(0.034) != pytest.approx(0.034)


def test_bridge_pos_clipped_to_unit():
    from benchmarks.utils import eef20d_to_robocasa12d

    eef20d = np.zeros(20, np.float32)
    eef20d[0:3] = [1.0, 0.0, 0.0]  # huge delta vs current 0 -> /0.05 = 20 -> clip to 1
    eef20d[3:9] = _IDENT_R6D
    out = eef20d_to_robocasa12d(eef20d, proprio_eef_pos=[0, 0, 0], proprio_eef_rot6d=_IDENT_R6D,
                                pos_scale=0.05, rot_scale=0.5)
    assert out[0] == pytest.approx(1.0)


def test_bridge_rotation_axis_angle():
    """90° about z, proprio identity, rot_scale=pi/2 -> rot_cmd ≈ [0,0,1]."""
    from benchmarks.utils import eef20d_to_robocasa12d

    eef20d = np.zeros(20, np.float32)
    eef20d[0:3] = [0, 0, 0]
    eef20d[3:9] = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]  # rot6d of 90°-about-z
    out = eef20d_to_robocasa12d(eef20d, proprio_eef_pos=[0, 0, 0], proprio_eef_rot6d=_IDENT_R6D,
                                pos_scale=0.05, rot_scale=np.pi / 2)
    assert out[3:6] == pytest.approx([0.0, 0.0, 1.0], abs=1e-4)


def test_client_proprio_matches_dataloader_repr():
    """The 20-D EEF proprio the client sends MUST match the dataloader's proprio
    representation (state_to_arm10 + left-pad) so train and deploy agree."""
    from openwam.dataloader.robocasa365 import state_to_arm10

    obs = _make_obs()
    sent = np.asarray(adapter.assemble_eef20d_proprio(obs), np.float32)
    assert sent.shape == (20,)
    state_row = np.zeros((1, 16), np.float32)
    state_row[0, 7:10] = obs["state.end_effector_position_relative"]
    state_row[0, 10:14] = obs["state.end_effector_rotation_relative"]
    state_row[0, 14:16] = obs["state.gripper_qpos"]
    arm10 = state_to_arm10(state_row)[0]
    assert sent[:10] == pytest.approx(arm10, abs=1e-5)  # left-10 == dataloader arm10
    assert (sent[10:] == 0).all()                       # right-10 zero-padded


def test_policy_act_bridges_20d_when_scales_set():
    eef20d = np.zeros(20, np.float32)
    eef20d[0:3] = [0.025, 0.0, 0.0]
    eef20d[3:9] = _IDENT_R6D
    eef20d[9] = 0.3
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=_FakeClient(action=eef20d.tolist()), osc_pos_scale=0.05, osc_rot_scale=0.5
    )
    action_dict = policy.act(_make_obs(), "x")
    assert set(action_dict) == set(adapter.ACTION_SLICES)  # bridged to 12-D env dict
    assert action_dict["action.control_mode"][0] == pytest.approx(-1.0)


def test_policy_act_20d_without_scales_raises():
    policy = adapter.OpenWAMRoboCasa365Policy(_client=_FakeClient(action=list(range(20))))
    with pytest.raises(ValueError, match="osc_pos_scale"):
        policy.act(_make_obs(), "x")


def _arm20(pos):
    a = np.zeros(20, np.float32)
    a[0:3] = pos
    a[3:9] = _IDENT_R6D
    a[9] = 0.7
    return a


def test_bridge_desired_mode_uses_prev_target_not_current():
    """control_mode=+1 (desired / "base mode"): robosuite OSC updates the arm goal from the last
    DESIRED goal (goal = last_goal + delta), so to reach the absolute target the bridge must form
    delta = target - previous_target, NOT target - current observed eef. Achieved (-1) uses the
    current eef. The bridge is stateful across steps (tracks the previous target)."""
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=_FakeClient(action=list(range(25))), osc_pos_scale=0.05, osc_rot_scale=0.5
    )
    obs = _make_obs()  # current eef pos = [4, 5, 6]

    # Step 1 (achieved, control_mode=-1): delta = (target1 - current_eef)/scale; sets prev_target=target1.
    # (small deltas so pos_cmd stays inside the [-1,1] clip and the two references stay distinguishable)
    out1 = policy._bridge_eef20d(obs, _arm20([4.02, 5.0, 6.0]), np.array([0, 0, 0, 0, -1.0], np.float32))
    assert out1[0:3] == pytest.approx([(4.02 - 4.0) / 0.05, 0.0, 0.0], abs=1e-4)  # (target1 - current)/scale = [0.4,0,0]

    # Step 2 (desired, control_mode=+1): delta MUST be (target2 - target1)/scale, NOT (target2 - current).
    out2 = policy._bridge_eef20d(obs, _arm20([4.03, 5.0, 6.0]), np.array([0.1, 0, 0, 0, 1.0], np.float32))
    assert out2[0:3] == pytest.approx([(4.03 - 4.02) / 0.05, 0.0, 0.0], abs=1e-4)  # (target2 - target1)/scale = [0.2,0,0]
    assert out2[0] != pytest.approx((4.03 - 4.0) / 0.05, abs=1e-3)  # would be 0.6 if it wrongly used current eef
    assert out2[11] == pytest.approx(1.0)  # control_mode forwarded


def test_bridge_reset_clears_prev_target():
    """reset() (called per-episode by single_eval) clears the tracked previous target, so the first
    desired-mode step of a new episode falls back to the current observed eef (matches the OSC's
    goal initialization at episode start)."""
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=_FakeClient(action=list(range(25))), osc_pos_scale=0.05, osc_rot_scale=0.5
    )
    obs = _make_obs()  # current eef = [4, 5, 6]
    des = np.array([0.1, 0, 0, 0, 1.0], np.float32)
    policy._bridge_eef20d(obs, _arm20([4.04, 5.0, 6.0]), des)  # sets prev_target
    policy.reset()
    # after reset, prev_target is None -> desired step uses current eef as reference again
    out = policy._bridge_eef20d(obs, _arm20([4.03, 5.0, 6.0]), des)
    assert out[0:3] == pytest.approx([(4.03 - 4.0) / 0.05, 0.0, 0.0], abs=1e-4)  # (target - current)/scale = [0.6,0,0]


# --- Task 2: mobile proprio (mobile_base ckpts send 25-D [arm20, base5=(vel3 A′-rescaled, 0, 0)]) ---


def _obs_base(pos, quat):
    o = _make_obs()
    o["state.base_position"] = np.asarray(pos, np.float32)
    o["state.base_rotation"] = np.asarray(quat, np.float32)
    return o


def test_base_velocity_cmd_matches_dataloader():
    """The client's pure-numpy base_velocity_cmd MUST equal the dataloader's base_velocity_cmd (train
    side) — same finite-diff AND same A′ rescale (× fps / _BASE_VEL_PHYS_MAX) — so a mobile ckpt sees
    the SAME command-space base velocity at eval as in training (no exposure bias / distribution shift)."""
    from openwam.dataloader.robocasa365 import base_velocity_cmd as train_cmd

    rng = np.random.RandomState(0)
    for _ in range(5):
        prev = rng.uniform(-1, 1, 7).astype(np.float32)
        cur = rng.uniform(-1, 1, 7).astype(np.float32)
        client = adapter.base_velocity_cmd(prev, cur)
        train = train_cmd(np.stack([prev, cur]))
        assert client == pytest.approx(train, abs=1e-5)


def test_policy_sends_25d_proprio_when_mobile():
    # mobile_base=True → the client sends 25-D proprio [arm20, base5]; the first step of an episode
    # has no previous base pose, so base5 = [0, 0, 0, 0, 0].
    fake = _FakeClient(action=list(range(25)))
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=fake, osc_pos_scale=0.05, osc_rot_scale=0.5, mobile_base=True
    )
    policy.reset()
    policy.act(_obs_base([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "x")
    state = fake.last_payload["state"]
    assert len(state) == 25                         # [arm20, base5]
    assert state[:20] == pytest.approx(list(adapter.assemble_eef20d_proprio(_make_obs())), abs=1e-5)
    assert state[20:25] == [0.0, 0.0, 0.0, 0.0, 0.0]  # first step: no prev -> zero base5


def test_mobile_base5_stateful_finite_diff():
    # step 2 base velocity = body-frame finite-diff of the two obs base poses (moved +0.1 in world x,
    # yaw 0 → body vx=+0.1), rescaled into the action command space (A′). torso + control_mode stay 0.
    fake = _FakeClient(action=list(range(25)))
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=fake, osc_pos_scale=0.05, osc_rot_scale=0.5, mobile_base=True
    )
    policy.reset()
    policy.act(_obs_base([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "x")  # step 1: prev set
    policy.act(_obs_base([0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "x")  # step 2: moved +x
    expected = adapter.base_velocity_cmd(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float32),
        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float32),
    )
    state = fake.last_payload["state"]
    assert state[20:23] == pytest.approx(list(expected), abs=1e-5)  # A′-rescaled velocity (not raw 0.1)
    assert state[23:25] == [0.0, 0.0]                               # torso + control_mode masked


def test_mobile_reset_clears_prev_base_pose():
    fake = _FakeClient(action=list(range(25)))
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=fake, osc_pos_scale=0.05, osc_rot_scale=0.5, mobile_base=True
    )
    policy.reset()
    policy.act(_obs_base([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "x")
    policy.act(_obs_base([0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "x")
    policy.reset()  # new episode: prev base pose cleared
    policy.act(_obs_base([0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0]), "x")  # first step -> zero again
    assert fake.last_payload["state"][20:25] == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_fixed_base_sends_20d():
    # default (mobile_base=False): proprio stays 20-D (no base5 appended).
    fake = _FakeClient(action=list(range(25)))
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=fake, osc_pos_scale=0.05, osc_rot_scale=0.5
    )
    policy.reset()
    policy.act(_make_obs(), "x")
    assert len(fake.last_payload["state"]) == 20


# --- Task 3: torso safety clamp (mask_torso_action zeros base_motion[3] at eval) ---
# torso is a LIVE JOINT_POSITION delta actuator (base_motion[3] → robot0_torso) but constant 0 in the
# data → masked out of the action loss, so the client must zero the (unconstrained) torso prediction.

_SERVER25_TORSO09 = list(np.arange(20, dtype=float)) + [0.5, -0.3, 0.2, 0.9, -1.0]  # base5 torso=0.9, mode=-1


def test_mask_torso_action_zeros_torso():
    # mask_torso_action=True (default): even if the server returns torso=0.9 in base5[3], the bridged
    # env action's base_motion[3] (torso) is 0; x_vel + control_mode pass through unchanged.
    fake = _FakeClient(action=_SERVER25_TORSO09)
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=fake, osc_pos_scale=0.05, osc_rot_scale=0.5, mobile_base=True  # mask_torso_action default True
    )
    policy.reset()
    act = policy.act(_obs_base([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "x")
    assert float(act["action.base_motion"][3]) == 0.0                 # torso zeroed despite server 0.9
    assert float(act["action.base_motion"][0]) == pytest.approx(0.5)  # x_vel passed through
    assert float(act["action.control_mode"][0]) == -1.0              # control_mode passed through


def test_mask_torso_action_false_passes_torso():
    # mask_torso_action=False: the model's torso prediction is passed through to the env.
    fake = _FakeClient(action=_SERVER25_TORSO09)
    policy = adapter.OpenWAMRoboCasa365Policy(
        _client=fake, osc_pos_scale=0.05, osc_rot_scale=0.5, mobile_base=True, mask_torso_action=False
    )
    policy.reset()
    act = policy.act(_obs_base([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]), "x")
    assert float(act["action.base_motion"][3]) == pytest.approx(0.9)  # torso passed through
