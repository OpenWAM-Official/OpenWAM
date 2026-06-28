"""Offline unit tests for the BEHAVIOR-1K (OmniGibson) eval bridge.

No OmniGibson, no Isaac Sim, no live OpenWAM server: the south policy client is
faked and the websocket is a stub. Covers the three things that must be correct
before a sim-capable box can validate the closed loop:

  1. the unified-80D ↔ R1Pro action / proprio conversions (pure numpy),
  2. the openpi-byte-compatible msgpack-numpy codec round-trip,
  3. the bridge dispatch contract (metadata-first, act→one reply, reset→no reply,
     prompt resolution, 80-D state assembly, 21-D action out).
"""

from __future__ import annotations

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
    UNIFIED_ACTION_DIM,
    quat_xyzw_to_axis_angle,
    quat_xyzw_to_rot6d,
    r1pro_proprio_to_unified80d,
    rot6d_to_axis_angle,
    rot6d_to_quat_xyzw,
    unified80d_to_r1pro_action,
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


# ── conversions: unified 80-D → R1Pro 21-D action ────────────────────────────


class TestUnifiedToR1Pro:
    def _make_action(self, rng):
        a = np.zeros(UNIFIED_ACTION_DIM, dtype=np.float32)
        a[0:3] = [0.4, -0.2, 0.3]  # L pos (metric, can exceed 1)
        a[3:9] = quat_xyzw_to_rot6d(_unit_quat(rng))  # L rot6d
        a[9] = 0.7  # L grip
        a[34:37] = [0.5, 0.25, 0.31]  # R pos
        a[37:43] = quat_xyzw_to_rot6d(_unit_quat(rng))  # R rot6d
        a[43] = -0.9  # R grip
        a[68:71] = [0.6, -0.4, 0.2]  # base vel
        a[71:75] = [0.1, -0.3, 0.25, -0.15]  # trunk
        return a

    def test_shape_and_layout(self):
        rng = np.random.RandomState(2)
        a = self._make_action(rng)
        out = unified80d_to_r1pro_action(a)
        assert out.shape == (R1PRO_IK_ACTION_DIM,) == (21,)
        # controller order [base3, trunk4, armL(pos3+aa3), gripL1, armR(pos3+aa3), gripR1]
        np.testing.assert_allclose(out[0:3], a[68:71], atol=1e-6)  # base
        np.testing.assert_allclose(out[3:7], a[71:75], atol=1e-6)  # trunk
        np.testing.assert_allclose(out[7:10], a[0:3], atol=1e-6)  # L arm pos (metric, unclipped)
        np.testing.assert_allclose(out[10:13], rot6d_to_axis_angle(a[3:9]), atol=1e-6)  # L arm aa
        assert out[13] == pytest.approx(0.7)  # L grip
        np.testing.assert_allclose(out[14:17], a[34:37], atol=1e-6)  # R arm pos
        np.testing.assert_allclose(out[17:20], rot6d_to_axis_angle(a[37:43]), atol=1e-6)  # R arm aa
        assert out[20] == pytest.approx(-0.9)  # R grip

    def test_passthrough_clipped_arms_not(self):
        a = np.zeros(UNIFIED_ACTION_DIM, dtype=np.float32)
        a[68:71] = [3.0, -2.0, 1.5]  # base out of range → clipped to [-1, 1]
        a[71:75] = [5.0, -5.0, 0.2, -9.0]  # trunk → clipped
        a[9], a[43] = 4.0, -4.0  # grippers → clipped
        a[0:3] = [2.5, -3.1, 4.2]  # L arm pos → NOT clipped (raw metric)
        out = unified80d_to_r1pro_action(a)
        np.testing.assert_allclose(out[0:3], [1.0, -1.0, 1.0])  # base clipped
        np.testing.assert_allclose(out[3:7], [1.0, -1.0, 0.2, -1.0])  # trunk clipped
        assert out[13] == 1.0 and out[20] == -1.0  # grippers clipped
        np.testing.assert_allclose(out[7:10], [2.5, -3.1, 4.2], atol=1e-6)  # arm pos unclipped

    def test_no_clip_option(self):
        a = np.zeros(UNIFIED_ACTION_DIM, dtype=np.float32)
        a[68:71] = [3.0, -2.0, 1.5]
        out = unified80d_to_r1pro_action(a, clip_passthrough=False)
        np.testing.assert_allclose(out[0:3], [3.0, -2.0, 1.5], atol=1e-6)

    def test_wrong_width_raises(self):
        with pytest.raises(ValueError, match="width 80"):
            unified80d_to_r1pro_action(np.zeros(23, dtype=np.float32))


# ── conversions: R1Pro 256-D proprio → unified 80-D ──────────────────────────


class TestProprioToUnified:
    def _make_proprio(self, rng):
        p = rng.uniform(-1, 1, size=256).astype(np.float32)
        p[186:189] = [0.41, -0.22, 0.33]  # L pos
        p[189:193] = _unit_quat(rng)  # L quat
        p[225:228] = [0.52, 0.21, 0.30]  # R pos
        p[228:232] = _unit_quat(rng)  # R quat
        p[236:240] = [0.12, -0.34, 0.56, -0.78]  # trunk
        return p

    def test_shape_and_eef_placement(self):
        rng = np.random.RandomState(3)
        p = self._make_proprio(rng)
        out = r1pro_proprio_to_unified80d(p)
        assert out.shape == (UNIFIED_ACTION_DIM,)
        np.testing.assert_allclose(out[0:3], p[186:189], atol=1e-6)  # L pos
        np.testing.assert_allclose(out[3:9], quat_xyzw_to_rot6d(p[189:193]), atol=1e-5)  # L rot6d
        np.testing.assert_allclose(out[34:37], p[225:228], atol=1e-6)  # R pos
        np.testing.assert_allclose(out[37:43], quat_xyzw_to_rot6d(p[228:232]), atol=1e-5)  # R rot6d
        np.testing.assert_allclose(out[71:75], p[236:240], atol=1e-6)  # trunk

    def test_unset_offsets_zero_filled(self):
        # base_vel / grippers default to None offsets → those unified slots stay 0.
        rng = np.random.RandomState(4)
        out = r1pro_proprio_to_unified80d(self._make_proprio(rng))
        assert (out[68:71] == 0).all()  # base vel (None offset)
        assert out[9] == 0.0 and out[43] == 0.0  # grippers (None offset)
        # dex + reserved-tail slots are always zero (never populated)
        assert (out[10:34] == 0).all() and (out[44:68] == 0).all() and (out[75:80] == 0).all()

    def test_offset_override(self):
        rng = np.random.RandomState(5)
        p = self._make_proprio(rng)
        p[250:253] = [0.1, 0.2, 0.3]  # pretend base vel lives here on a real build
        offs = dict(
            l_pos=slice(186, 189), l_quat=slice(189, 193), r_pos=slice(225, 228), r_quat=slice(228, 232),
            trunk=slice(236, 240), base_vel=slice(250, 253), l_grip=None, r_grip=None,
        )
        out = r1pro_proprio_to_unified80d(p, offsets=offs)
        np.testing.assert_allclose(out[68:71], [0.1, 0.2, 0.3], atol=1e-6)


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
    """Stand-in for WSPolicyClient: records payloads, returns a canned 80-D action."""

    def __init__(self):
        self.payloads = []
        self.reset_calls = 0
        self.action80 = np.zeros(80, dtype=np.float32)
        self.action80[0:3] = [0.3, 0.1, 0.2]  # L pos
        self.action80[3:9] = [1, 0, 0, 0, 1, 0]  # identity rot6d
        self.action80[34:37] = [0.4, -0.1, 0.25]
        self.action80[37:43] = [1, 0, 0, 0, 1, 0]
        self.action80[68:71] = [0.2, -0.1, 0.05]
        self.action80[71:75] = [0.1, 0.0, -0.1, 0.2]

    def predict(self, payload):
        self.payloads.append(payload)
        return {"type": transport.ACTION, "action": self.action80.tolist()}

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
        assert len(payload["state"]) == 80  # unified proprio assembled

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
