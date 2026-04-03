"""Tests for WAMPolicy receding-horizon execution and temporal ensembling."""

import numpy as np
import pytest
from types import SimpleNamespace

from open_wam.evaluation.policy import WAMPolicy


class MockEngine:
    """Mock inference engine that returns predictable action chunks."""

    def __init__(self, chunk_size: int = 10, action_dim: int = 7):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.call_count = 0

    def generate(self, conditions: dict) -> dict:
        """Return a chunk of actions where each action = [call_count] * action_dim."""
        self.call_count += 1
        actions = np.full((self.chunk_size, self.action_dim), self.call_count, dtype=np.float32)
        return {"actions": actions, "video": None}


def make_cfg(**kwargs):
    """Create a config namespace."""
    defaults = {"history_len": 5}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_greedy_mode():
    """Without execute_horizon, consume full chunk before re-generating."""
    engine = MockEngine(chunk_size=4)
    cfg = make_cfg(execute_horizon=None, temporal_ensemble=False)
    policy = WAMPolicy(engine, cfg)

    obs = {"image": np.zeros(3)}

    # First 4 calls use chunk 1
    for _ in range(4):
        a = policy.predict_action(obs)
        np.testing.assert_allclose(a, 1.0)

    # Next call triggers chunk 2
    a = policy.predict_action(obs)
    np.testing.assert_allclose(a, 2.0)

    assert engine.call_count == 2


def test_receding_horizon_triggers_regeneration():
    """With execute_horizon=2, re-generate every 2 steps."""
    engine = MockEngine(chunk_size=6)
    cfg = make_cfg(execute_horizon=2, temporal_ensemble=False)
    policy = WAMPolicy(engine, cfg)

    obs = {"image": np.zeros(3)}

    # Steps 0,1 from generation 1
    a0 = policy.predict_action(obs)
    a1 = policy.predict_action(obs)
    assert engine.call_count == 1

    # Step 2 triggers generation 2
    a2 = policy.predict_action(obs)
    assert engine.call_count == 2


def test_temporal_ensemble_fuses_predictions():
    """Overlapping predictions are fused via weighted average."""
    engine = MockEngine(chunk_size=4, action_dim=1)
    cfg = make_cfg(execute_horizon=2, temporal_ensemble=True, ensemble_decay=0.5)
    policy = WAMPolicy(engine, cfg)

    obs = {"image": np.zeros(1)}

    # Generation 1: actions = [1, 1, 1, 1] for steps 0,1,2,3
    a0 = policy.predict_action(obs)  # step 0, only gen1 -> 1.0
    a1 = policy.predict_action(obs)  # step 1, only gen1 -> 1.0
    assert engine.call_count == 1
    np.testing.assert_allclose(a0, 1.0)
    np.testing.assert_allclose(a1, 1.0)

    # Generation 2: actions = [2, 2, 2, 2] for steps 2,3,4,5
    # Step 2 has: gen1 (decayed) value 1 + gen2 value 2
    a2 = policy.predict_action(obs)  # step 2, ensemble of gen1 + gen2
    assert engine.call_count == 2
    # a2 should be between 1.0 and 2.0 (weighted average)
    assert 1.0 < float(a2.item()) < 2.0 + 1e-6


def test_ensemble_newer_predictions_weighted_higher():
    """Newer predictions should have higher effective weight."""
    engine = MockEngine(chunk_size=6, action_dim=1)
    cfg = make_cfg(execute_horizon=2, temporal_ensemble=True, ensemble_decay=0.3)
    policy = WAMPolicy(engine, cfg)

    obs = {"image": np.zeros(1)}

    # Gen 1: all 1.0
    policy.predict_action(obs)
    policy.predict_action(obs)

    # Gen 2 at step 2: all 2.0. Step 2 has overlap with gen1.
    a2 = policy.predict_action(obs)

    # With decay=0.3, gen1 weight=0.3, gen2 weight=1.0
    # Expected: (0.3*1 + 1.0*2) / (0.3+1.0) = 2.3/1.3 ≈ 1.77
    expected = (0.3 * 1.0 + 1.0 * 2.0) / (0.3 + 1.0)
    np.testing.assert_allclose(float(a2.item()), expected, atol=1e-5)


def test_reset_clears_state():
    """Reset should clear all buffers and counters."""
    engine = MockEngine(chunk_size=4)
    cfg = make_cfg(execute_horizon=2, temporal_ensemble=True)
    policy = WAMPolicy(engine, cfg)

    obs = {"image": np.zeros(1)}
    policy.predict_action(obs)
    policy.predict_action(obs)

    policy.reset()

    assert len(policy._action_buffer) == 0
    assert len(policy._ensemble_buffer) == 0
    assert len(policy.obs_history) == 0
    assert policy._current_step == 0
    assert policy._steps_since_generate == 0

    # After reset, next call should trigger fresh generation
    policy.predict_action(obs)
    assert engine.call_count == 2  # One before reset, one after


def test_greedy_backward_compatible():
    """Default config (no execute_horizon) should behave like old WAMPolicy."""
    engine = MockEngine(chunk_size=3)
    cfg = make_cfg()  # No execute_horizon -> None
    policy = WAMPolicy(engine, cfg)

    obs = {"image": np.zeros(1)}

    actions = [policy.predict_action(obs) for _ in range(9)]

    # 3 chunks consumed
    assert engine.call_count == 3
    # First 3 actions from gen1, next 3 from gen2, etc.
    for a in actions[:3]:
        np.testing.assert_allclose(a, 1.0)
    for a in actions[3:6]:
        np.testing.assert_allclose(a, 2.0)
    for a in actions[6:9]:
        np.testing.assert_allclose(a, 3.0)


def test_observation_history():
    """Observations should be accumulated in history."""
    engine = MockEngine(chunk_size=5)
    cfg = make_cfg(history_len=3, execute_horizon=None)
    policy = WAMPolicy(engine, cfg)

    for i in range(5):
        policy.predict_action({"step": i})

    assert len(policy.obs_history) == 3  # maxlen=3
    assert policy.obs_history[-1]["step"] == 4
