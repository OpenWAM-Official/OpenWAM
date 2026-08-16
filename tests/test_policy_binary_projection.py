"""WAMPolicy final legality projection for two-point command dims.

The normalizer normally emits exact ±1 for ``binary_action_dims``. The policy
boundary still enforces that wire contract for raw or legacy engine outputs,
driven by ``architecture.binary_command_dims``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from openwam.deploy.policy import WAMPolicy

DIM = 25
MODE = 24  # raw control_mode dim


class _FakeEngine:
    """Emits queued (T, 25) chunks; carries a fake architecture with binary_command_dims."""

    def __init__(self, chunks, binary_dims=(MODE,)):
        self._chunks = list(chunks)
        self.architecture = SimpleNamespace(binary_command_dims=tuple(binary_dims))

    def generate(self, conditions):
        return {"actions": np.array(self._chunks.pop(0), dtype=np.float32)}


def _chunk(t, mode_value):
    c = np.zeros((t, DIM), np.float32)
    c[:, MODE] = mode_value
    return c


def _policy(engine, inference_horizon=None):
    p = WAMPolicy(
        engine=engine,
        cfg=SimpleNamespace(),
        execution_config={"mode": "sync", "inference_horizon": inference_horizon},
    )
    p._build_conditions = lambda obs: obs  # bypass image/prompt assembly — not under test
    return p


def test_replan_uses_fresh_chunk_without_temporal_mixing():
    policy = _policy(_FakeEngine([_chunk(4, +1.0), _chunk(4, -1.0)]), inference_horizon=2)
    a0 = policy.predict_action({})
    a1 = policy.predict_action({})
    a2 = policy.predict_action({})
    for a in (a0, a1, a2):
        assert float(a[MODE]) in (-1.0, 1.0)
    assert a0[MODE] == 1.0 and a1[MODE] == 1.0
    assert a2[MODE] == -1.0


def test_projection_noop_without_binary_dims():
    engine = _FakeEngine([_chunk(4, -0.333)], binary_dims=())
    policy = _policy(engine)
    a = policy.predict_action({})
    assert a[MODE] == pytest.approx(-0.333)  # untouched: projection only governs declared dims


def test_projection_independent_of_normalizer():
    """The projection reads architecture.binary_command_dims only — it must work for a
    normalize_mode=null ckpt (no normalizer anywhere), closing the B3 contract."""
    engine = _FakeEngine([_chunk(2, 0.97)])  # raw-space model output, never normalized
    policy = _policy(engine)
    assert policy.predict_action({})[MODE] == 1.0


def test_projection_width_mismatch_raises():
    engine = _FakeEngine([np.zeros((2, 12), np.float32)], binary_dims=(24,))
    policy = _policy(engine)
    with pytest.raises(ValueError, match="binary_command_dims"):
        policy.predict_action({})


def test_projection_missing_architecture_is_noop():
    """Engines without an architecture (require_architecture=False paths) must not break."""
    engine = _FakeEngine([_chunk(2, -0.4)])
    engine.architecture = None
    policy = _policy(engine)
    assert policy.predict_action({})[MODE] == pytest.approx(-0.4)
