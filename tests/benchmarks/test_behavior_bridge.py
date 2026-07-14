"""Offline unit tests for the BEHAVIOR-1K (OmniGibson) eval bridge.

No OmniGibson, no Isaac Sim, no live OpenWAM server: the south policy client is
faked and the websocket is a stub. Covers the three things that must be correct
before a sim-capable box can validate the closed loop:

  1. the unified-80D ↔ R1Pro action / proprio conversions (pure numpy),
  2. the openpi-byte-compatible msgpack-numpy codec round-trip,
  3. the bridge dispatch contract (metadata-first, act→one reply, reset→no reply,
     prompt resolution, RAW-27 state assembly, 21-D action out).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from benchmarks.behavior import msgpack_numpy
from benchmarks.behavior.openwam2behavior_bridge import (
    HEAD_KEY,
    LEFT_WRIST_KEY,
    PROPRIO_KEY,
    RIGHT_WRIST_KEY,
    TASK_ID_KEY,
    BehaviorBridge,
    _make_handler,
)
from benchmarks.utils import transport
from benchmarks.utils.action_conversion import (
    R1PRO_IK_ACTION_DIM,
    R1PRO_RAW_DIM,
    quat_xyzw_to_axis_angle,
    quat_xyzw_to_rot6d,
    r1pro_proprio_to_raw27,
    raw27_to_r1pro_action,
    rot6d_to_axis_angle,
    rot6d_to_quat_xyzw,
)


def _unit_quat(rng):
    q = rng.uniform(-1, 1, size=4).astype(np.float64)
    return q / np.linalg.norm(q)


def _axisangle_to_quat(aa):
    """Local inverse of ``quat_xyzw_to_axis_angle`` (== OmniGibson axisangle2quat)."""
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = np.asarray(aa) / angle
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def _same_rotation(qa, qb):
    """Quaternions represent the same rotation iff |qa·qb| ≈ 1 (double cover)."""
    return abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)))) > 1 - 1e-5


# ── conversions: orientation ─────────────────────────────────────────────────


class TestAxisAngle:
    def test_identity_rot6d_is_zero(self):
        ident = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)  # identity rotation rot6d
        np.testing.assert_allclose(rot6d_to_axis_angle(ident), np.zeros(3), atol=1e-6)

    def test_quat_axisangle_roundtrip(self):
        rng = np.random.RandomState(0)
        for _ in range(50):
            q = _unit_quat(rng)
            aa = quat_xyzw_to_axis_angle(q)
            assert _same_rotation(q, _axisangle_to_quat(aa))
            assert np.linalg.norm(aa) <= np.pi + 1e-6  # minimal rotation vector

    def test_rot6d_axisangle_matches_rot6d_quat(self):
        # rot6d → axis-angle → quat must equal rot6d → quat (the orientation the
        # IK controller reconstructs via axisangle2quat).
        rng = np.random.RandomState(1)
        for _ in range(50):
            r6d = quat_xyzw_to_rot6d(_unit_quat(rng))
            aa = rot6d_to_axis_angle(r6d)
            assert _same_rotation(rot6d_to_quat_xyzw(r6d), _axisangle_to_quat(aa))


# ── conversions: RAW-27 → R1Pro 21-D action ──────────────────────────────────


class TestRaw27ToR1Pro:
    def _make_action(self, rng):
        a = np.zeros(R1PRO_RAW_DIM, dtype=np.float32)
        a[0:3] = [0.4, -0.2, 0.3]  # L pos (metric, can exceed 1)
        a[3:9] = quat_xyzw_to_rot6d(_unit_quat(rng))  # L rot6d
        a[9] = 0.7  # L grip
        a[10:13] = [0.5, 0.25, 0.31]  # R pos
        a[13:19] = quat_xyzw_to_rot6d(_unit_quat(rng))  # R rot6d
        a[19] = -0.9  # R grip
        a[20:23] = [0.6, -0.4, 0.2]  # base vel
        a[23:27] = [0.1, -0.3, 0.25, -0.15]  # trunk
        return a

    def test_shape_and_layout(self):
        rng = np.random.RandomState(2)
        a = self._make_action(rng)
        out = raw27_to_r1pro_action(a)
        assert out.shape == (R1PRO_IK_ACTION_DIM,) == (21,)
        # controller order [base3, trunk4, armL(pos3+aa3), gripL1, armR(pos3+aa3), gripR1]
        np.testing.assert_allclose(out[0:3], a[20:23], atol=1e-6)  # base
        np.testing.assert_allclose(out[3:7], a[23:27], atol=1e-6)  # trunk
        np.testing.assert_allclose(out[7:10], a[0:3], atol=1e-6)  # L arm pos (metric, unclipped)
        np.testing.assert_allclose(out[10:13], rot6d_to_axis_angle(a[3:9]), atol=1e-6)  # L arm aa
        assert out[13] == pytest.approx(0.7)  # L grip
        np.testing.assert_allclose(out[14:17], a[10:13], atol=1e-6)  # R arm pos
        np.testing.assert_allclose(out[17:20], rot6d_to_axis_angle(a[13:19]), atol=1e-6)  # R arm aa
        assert out[20] == pytest.approx(-0.9)  # R grip

    def test_passthrough_clipped_arms_not(self):
        a = np.zeros(R1PRO_RAW_DIM, dtype=np.float32)
        a[20:23] = [3.0, -2.0, 1.5]  # base out of range → clipped to [-1, 1]
        a[23:27] = [5.0, -5.0, 0.2, -9.0]  # trunk → clipped
        a[9], a[19] = 4.0, -4.0  # grippers → clipped
        a[0:3] = [2.5, -3.1, 4.2]  # L arm pos → NOT clipped (raw metric)
        out = raw27_to_r1pro_action(a)
        np.testing.assert_allclose(out[0:3], [1.0, -1.0, 1.0])  # base clipped
        np.testing.assert_allclose(out[3:7], [1.0, -1.0, 0.2, -1.0])  # trunk clipped
        assert out[13] == 1.0 and out[20] == -1.0  # grippers clipped
        np.testing.assert_allclose(out[7:10], [2.5, -3.1, 4.2], atol=1e-6)  # arm pos unclipped

    def test_no_clip_option(self):
        a = np.zeros(R1PRO_RAW_DIM, dtype=np.float32)
        a[20:23] = [3.0, -2.0, 1.5]
        out = raw27_to_r1pro_action(a, clip_passthrough=False)
        np.testing.assert_allclose(out[0:3], [3.0, -2.0, 1.5], atol=1e-6)

    def test_wrong_width_raises(self):
        with pytest.raises(ValueError, match="width 27"):
            raw27_to_r1pro_action(np.zeros(80, dtype=np.float32))


# ── conversions: R1Pro 256-D proprio → RAW-27 ────────────────────────────────


class TestProprioToRaw27:
    def _make_proprio(self, rng):
        """A realistic R1Pro 256-D measured proprio (only the fields the renderer reads)."""
        p = rng.uniform(-1, 1, size=256).astype(np.float32)
        p[186:189] = [0.41, -0.22, 0.33]  # L eef pos
        p[189:193] = _unit_quat(rng)  # L eef quat
        p[225:228] = [0.52, 0.21, 0.30]  # R eef pos
        p[228:232] = _unit_quat(rng)  # R eef quat
        p[193:195] = rng.uniform(0.0, 0.05, size=2)  # L gripper: 2 finger qpos (m)
        p[232:234] = rng.uniform(0.0, 0.05, size=2)  # R gripper: 2 finger qpos
        p[236:240] = [0.12, -0.34, 0.56, -0.78]  # trunk qpos (rad)
        p[253:256] = [0.1, -0.05, 0.2]  # base_qvel (WORLD frame)
        p[246] = 0.7  # base yaw (world)
        return p

    def test_shape_and_eef_placement(self):
        rng = np.random.RandomState(3)
        p = self._make_proprio(rng)
        out = r1pro_proprio_to_raw27(p)
        assert out.shape == (R1PRO_RAW_DIM,)
        np.testing.assert_allclose(out[0:3], p[186:189], atol=1e-6)  # L pos
        np.testing.assert_allclose(out[3:9], quat_xyzw_to_rot6d(p[189:193]), atol=1e-5)  # L rot6d
        np.testing.assert_allclose(out[10:13], p[225:228], atol=1e-6)  # R pos
        np.testing.assert_allclose(out[13:19], quat_xyzw_to_rot6d(p[228:232]), atol=1e-5)  # R rot6d
        np.testing.assert_allclose(out[23:27], p[236:240], atol=1e-6)  # trunk qpos

    def test_gripper_open_scale_and_base_local_frame(self):
        # Grippers: mean of the 2 finger qpos → open-scale 2*mean/0.05-1 ∈ [-1,1].
        # Base: world base_qvel rotated by -yaw into the base frame, /[0.75,0.75,1.0].
        rng = np.random.RandomState(4)
        p = self._make_proprio(rng)
        out = r1pro_proprio_to_raw27(p)
        exp_l = np.clip(2.0 * p[193:195].mean() / 0.05 - 1.0, -1.0, 1.0)
        exp_r = np.clip(2.0 * p[232:234].mean() / 0.05 - 1.0, -1.0, 1.0)
        assert out[9] == pytest.approx(exp_l, abs=1e-6)  # L grip open-scale
        assert out[19] == pytest.approx(exp_r, abs=1e-6)  # R grip open-scale
        yaw = float(p[246])
        c, s = np.cos(yaw), np.sin(yaw)
        qv = p[253:256]
        exp_base = np.array([c * qv[0] + s * qv[1], -s * qv[0] + c * qv[1], qv[2]]) / np.array([0.75, 0.75, 1.0])
        np.testing.assert_allclose(out[20:23], exp_base, atol=1e-6)  # base-frame velocity

    def test_matches_trainer_rendering(self):
        # The deploy renderer MUST equal the trainer's _state_to_raw_proprio_eef
        # byte-for-byte on the same 256-D state — otherwise train/deploy proprio skew.
        behavior = pytest.importorskip("openwam.dataloader.behavior")
        rng = np.random.RandomState(6)
        p = self._make_proprio(rng)
        deploy = r1pro_proprio_to_raw27(p)
        trainer = behavior._state_to_raw_proprio_eef(p[None])[0]
        np.testing.assert_allclose(deploy, trainer, atol=1e-6)

    def test_wrong_width_raises(self):
        with pytest.raises(ValueError, match="expected R1Pro proprio of width 256"):
            r1pro_proprio_to_raw27(np.zeros(80, dtype=np.float32))


# ── msgpack-numpy codec (openpi byte layout) ─────────────────────────────────


class TestMsgpackCodec:
    def test_roundtrip_arrays_and_keys(self):
        obs = {
            HEAD_KEY: np.zeros((8, 6, 3), dtype=np.uint8),
            "observation/state": np.arange(80, dtype=np.float32),
            TASK_ID_KEY: np.array([7], dtype=np.int64),
            "prompt": "turn on the radio",
        }
        back = msgpack_numpy.unpackb(msgpack_numpy.packb(obs), strict_map_key=False)
        assert back["prompt"] == "turn on the radio"
        assert back[HEAD_KEY].dtype == np.uint8 and back[HEAD_KEY].shape == (8, 6, 3)
        np.testing.assert_array_equal(back["observation/state"], obs["observation/state"])
        assert back["observation/state"].dtype == np.float32
        assert int(back[TASK_ID_KEY][0]) == 7

    def test_ndarray_uses_bytes_keys(self):
        # The wire layout must use the openpi bytes-key sentinel (decode-independent).
        import msgpack

        raw = msgpack_numpy.packb({"x": np.ones(2, dtype=np.float32)})
        plain = msgpack.unpackb(raw, raw=True)  # no numpy hook → see the raw map
        assert b"__ndarray__" in plain[b"x"]
        assert plain[b"x"][b"dtype"] == b"<f4"

    def test_rejects_complex(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            msgpack_numpy.packb({"z": np.ones(2, dtype=np.complex64)})


# ── bridge dispatch (faked south server) ─────────────────────────────────────


class _FakeSouth:
    """Stand-in for WSPolicyClient: records payloads, returns a canned RAW-27 action."""

    def __init__(self):
        self.payloads = []
        self.reset_calls = 0
        self.action_raw = np.zeros(27, dtype=np.float32)
        self.action_raw[0:3] = [0.3, 0.1, 0.2]  # L pos
        self.action_raw[3:9] = [1, 0, 0, 0, 1, 0]  # L identity rot6d
        self.action_raw[10:13] = [0.4, -0.1, 0.25]  # R pos
        self.action_raw[13:19] = [1, 0, 0, 0, 1, 0]  # R identity rot6d
        self.action_raw[20:23] = [0.2, -0.1, 0.05]  # base vel
        self.action_raw[23:27] = [0.1, 0.0, -0.1, 0.2]  # trunk

    def predict(self, payload):
        self.payloads.append(payload)
        return {"type": transport.ACTION, "action": self.action_raw.tolist()}

    def reset(self):
        self.reset_calls += 1
        return {"type": transport.RESET_ACK}

    def ping(self):
        return {"type": transport.PONG}

    def close(self):
        pass


def _make_obs(rng, *, with_prompt=False, with_wrists=True):
    obs = {
        HEAD_KEY: rng.randint(0, 255, size=(12, 10, 3), dtype=np.uint8),
        PROPRIO_KEY: rng.uniform(-1, 1, size=256).astype(np.float32),
        TASK_ID_KEY: np.array([0], dtype=np.int64),
    }
    obs[PROPRIO_KEY][189:193] = _unit_quat(rng)
    obs[PROPRIO_KEY][228:232] = _unit_quat(rng)
    if with_wrists:
        obs[LEFT_WRIST_KEY] = rng.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
        obs[RIGHT_WRIST_KEY] = rng.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
    if with_prompt:
        obs["prompt"] = "explicit instruction"
    return obs


def _bridge(south, **kw):
    kw.setdefault("task_names", {0: "turning_on_radio"})
    return BehaviorBridge(south_client=south, **kw)


class TestBridgeDispatch:
    def test_act_returns_21d_action(self):
        south = _FakeSouth()
        b = _bridge(south)
        out = b.handle_message(_make_obs(np.random.RandomState(6)))
        assert set(out) == {"action"}
        assert out["action"].shape == (21,)
        # base / trunk passed through from the canned south action
        np.testing.assert_allclose(out["action"][0:3], [0.2, -0.1, 0.05], atol=1e-6)
        np.testing.assert_allclose(out["action"][3:7], [0.1, 0.0, -0.1, 0.2], atol=1e-6)

    def test_payload_forwarded_to_south(self):
        south = _FakeSouth()
        b = _bridge(south)
        b.handle_message(_make_obs(np.random.RandomState(7)))
        (payload,) = south.payloads
        assert payload["images"]["head_camera"] is not None
        assert payload["images"]["left_wrist_camera"] is not None
        assert payload["images"]["right_wrist_camera"] is not None
        assert payload["prompt"] == "turning on radio"  # task_id 0 de-underscored
        assert len(payload["state"]) == 27  # RAW-27 proprio assembled

    def test_reset_does_not_reply(self):
        south = _FakeSouth()
        b = _bridge(south)
        assert b.handle_message({"reset": True}) is None
        assert south.reset_calls == 1
        assert south.payloads == []

    def test_explicit_prompt_wins(self):
        south = _FakeSouth()
        b = _bridge(south)
        b.handle_message(_make_obs(np.random.RandomState(8), with_prompt=True))
        assert south.payloads[0]["prompt"] == "explicit instruction"

    def test_unmapped_task_id_raises(self):
        south = _FakeSouth()
        b = _bridge(south, task_names={})  # no mapping, no default
        with pytest.raises(ValueError, match="could not resolve a task prompt"):
            b.handle_message(_make_obs(np.random.RandomState(9)))

    def test_default_prompt_fallback(self):
        south = _FakeSouth()
        b = _bridge(south, task_names={}, default_prompt="do the task")
        b.handle_message(_make_obs(np.random.RandomState(10)))
        assert south.payloads[0]["prompt"] == "do the task"

    def test_missing_head_camera_raises(self):
        south = _FakeSouth()
        b = _bridge(south)
        obs = _make_obs(np.random.RandomState(11))
        del obs[HEAD_KEY]
        with pytest.raises(KeyError, match="head camera"):
            b.handle_message(obs)

    def test_no_send_state(self):
        south = _FakeSouth()
        b = _bridge(south, send_state=False)
        b.handle_message(_make_obs(np.random.RandomState(12), with_wrists=False))
        assert south.payloads[0].get("state") is None

    def test_missing_proprio_raises_when_state_required(self):
        south = _FakeSouth()
        b = _bridge(south)
        obs = _make_obs(np.random.RandomState(13))
        del obs[PROPRIO_KEY]
        with pytest.raises(KeyError, match="proprio"):
            b.handle_message(obs)


# ── server framing (stub websocket) ──────────────────────────────────────────


class _FakeWS:
    """Minimal websockets.sync stub: feeds queued frames, records sends/close."""

    def __init__(self, incoming):
        self._incoming = list(incoming)
        self.sent = []
        self.closed = None

    def recv(self):
        if not self._incoming:
            raise ConnectionError("client gone")  # ends the handler loop
        return self._incoming.pop(0)

    def send(self, data):
        self.sent.append(data)

    def close(self, code=None, reason=None):
        self.closed = (code, reason)


class TestServerFraming:
    def test_metadata_first_then_one_reply_per_act(self):
        south = _FakeSouth()
        bridge = _bridge(south)
        packer = msgpack_numpy.Packer()
        frames = [
            packer.pack({"reset": True}),
            packer.pack(_make_obs(np.random.RandomState(14))),
        ]
        ws = _FakeWS(frames)
        _make_handler(bridge)(ws)

        # frame 0 = metadata ({}), then exactly ONE reply for the act frame
        # (reset produced no reply): 2 sends total.
        assert len(ws.sent) == 2
        assert msgpack_numpy.unpackb(ws.sent[0], strict_map_key=False) == {}
        reply = msgpack_numpy.unpackb(ws.sent[1], strict_map_key=False)
        assert reply["action"].shape == (21,)
        assert south.reset_calls == 1

    def test_error_sends_text_frame_and_closes(self):
        south = _FakeSouth()
        bridge = _bridge(south)
        packer = msgpack_numpy.Packer()
        bad = _make_obs(np.random.RandomState(15))
        del bad[HEAD_KEY]  # triggers KeyError inside infer
        ws = _FakeWS([packer.pack(bad)])
        _make_handler(bridge)(ws)
        # metadata frame + a TEXT (str) traceback frame, then close 1011.
        assert isinstance(ws.sent[-1], str)
        assert "head camera" in ws.sent[-1]
        assert ws.closed is not None and ws.closed[0] == 1011


class TestPromptLoggingAndFallback:
    """The round-2 additions: per-episode `resolved prompt` logging and loud fallbacks."""

    def test_resolved_prompt_logged_on_change_and_after_reset(self, caplog):
        south = _FakeSouth()
        b = _bridge(south)
        rng = np.random.RandomState(20)
        with caplog.at_level(logging.INFO, logger="behavior_bridge"):
            b.handle_message(_make_obs(rng))
            b.handle_message(_make_obs(rng))  # same task → no second line
            assert [r for r in caplog.records if "resolved prompt" in r.message]
            n_before = sum("resolved prompt" in r.message for r in caplog.records)
            assert n_before == 1
            b.handle_message({"reset": True})  # new episode → log again
            b.handle_message(_make_obs(rng))
            n_after = sum("resolved prompt" in r.message for r in caplog.records)
            assert n_after == 2

    def test_default_prompt_fallback_warns_when_map_misses_id(self, caplog):
        south = _FakeSouth()
        b = _bridge(south, default_prompt="fallback text")
        obs = _make_obs(np.random.RandomState(21))
        obs[TASK_ID_KEY] = np.array([42], dtype=np.int64)  # not in {0: ...}
        with caplog.at_level(logging.WARNING, logger="behavior_bridge"):
            b.handle_message(obs)
        (payload,) = south.payloads
        assert payload["prompt"] == "fallback text"
        assert any("task_id 42 missing from --task-names" in r.message for r in caplog.records)

    def test_load_task_names_warns_on_underscored_values(self, tmp_path, caplog):
        import json as _json

        from benchmarks.behavior.openwam2behavior_bridge import _load_task_names

        p = tmp_path / "map.json"
        p.write_text(_json.dumps({"0": "clean sentence", "1": "turning_on_radio"}))
        with caplog.at_level(logging.WARNING, logger="behavior_bridge"):
            mapping = _load_task_names(str(p))
        assert mapping == {0: "clean sentence", 1: "turning_on_radio"}
        assert any("de-underscored" in r.message for r in caplog.records)
