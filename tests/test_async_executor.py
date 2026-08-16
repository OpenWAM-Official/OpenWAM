"""Tests for async inference executor."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from openwam.deploy.executors.async_executor import AsyncInferenceExecutor, normalize_execution_config
from openwam.deploy.executors.sync_executor import SyncInferenceExecutor


class MockEngine:
    """Mock inference engine that returns deterministic actions."""

    def __init__(self, action_dim=7, num_frames=10, latency=0.01):
        self.action_dim = action_dim
        self.num_frames = num_frames
        self.latency = latency
        self.call_count = 0

    def generate(self, conditions):
        time.sleep(self.latency)
        self.call_count += 1
        actions = np.ones((self.num_frames, self.action_dim), dtype=np.float32) * self.call_count
        return {"video": None, "actions": actions}


class BlockingSecondCallEngine(MockEngine):
    """Mock engine whose second call blocks until released."""

    def __init__(self):
        super().__init__(num_frames=4, latency=0.0)
        self.second_call_started = threading.Event()
        self.release_second_call = threading.Event()

    def generate(self, conditions):
        self.call_count += 1
        if self.call_count == 2:
            self.second_call_started.set()
            self.release_second_call.wait(timeout=2.0)
        actions = np.ones((self.num_frames, self.action_dim), dtype=np.float32) * self.call_count
        return {"video": None, "actions": actions}


class FailingSecondCallEngine(MockEngine):
    """Mock engine whose background call fails once and then recovers."""

    def __init__(self):
        super().__init__(num_frames=4, latency=0.0)

    def generate(self, conditions):
        self.call_count += 1
        if self.call_count == 2:
            raise RuntimeError("background failure")
        actions = np.ones((self.num_frames, self.action_dim), dtype=np.float32) * self.call_count
        return {"video": None, "actions": actions}


class IndexedEngine(MockEngine):
    """Mock engine whose actions encode call number and action index."""

    def __init__(self, action_dim=1, num_frames=8, latency=0.0):
        super().__init__(action_dim=action_dim, num_frames=num_frames, latency=latency)

    def generate(self, conditions):
        time.sleep(self.latency)
        self.call_count += 1
        values = self.call_count * 100 + np.arange(self.num_frames, dtype=np.float32)
        actions = np.repeat(values[:, None], self.action_dim, axis=1)
        return {"video": None, "actions": actions}


def test_async_executor_basic():
    engine = MockEngine(num_frames=5, latency=0.0)
    executor = AsyncInferenceExecutor(engine, inference_horizon=5, inference_delay_steps=0)

    action = executor.predict_action({"obs": "dummy"})
    assert action.shape == (7,)
    assert engine.call_count == 1
    executor.shutdown()


def test_async_executor_buffer_exhaustion_without_background():
    engine = MockEngine(num_frames=3, latency=0.0)
    executor = AsyncInferenceExecutor(engine, inference_horizon=3, inference_delay_steps=0)
    executor._background_enabled = False  # deterministic: no background generation

    for _ in range(3):
        action = executor.predict_action({"obs": "dummy"})
        np.testing.assert_allclose(action, np.ones(7) * 1.0)
    assert engine.call_count == 1

    action = executor.predict_action({"obs": "dummy"})
    np.testing.assert_allclose(action, np.ones(7) * 2.0)
    assert engine.call_count == 2

    executor.shutdown()


def test_async_executor_inference_horizon_discards_tail():
    engine = MockEngine(num_frames=5, latency=0.0)
    executor = AsyncInferenceExecutor(engine, inference_horizon=2, inference_delay_steps=0)
    executor._background_enabled = False  # deterministic: no background generation

    for _ in range(2):
        action = executor.predict_action({"obs": "dummy"})
        np.testing.assert_allclose(action, np.ones(7) * 1.0)

    action = executor.predict_action({"obs": "dummy"})
    np.testing.assert_allclose(action, np.ones(7) * 2.0)
    assert engine.call_count == 2

    executor.shutdown()


def test_sync_executor_inference_horizon_discards_tail_without_mixing():
    engine = IndexedEngine(num_frames=5, latency=0.0)
    executor = SyncInferenceExecutor(engine, inference_horizon=2)

    np.testing.assert_allclose(executor.predict_action({"obs": "step0"}), [100.0])
    np.testing.assert_allclose(executor.predict_action({"obs": "step1"}), [101.0])
    np.testing.assert_allclose(executor.predict_action({"obs": "step2"}), [200.0])
    assert engine.call_count == 2


def test_sync_executor_full_chunk_when_inference_horizon_is_none():
    engine = IndexedEngine(num_frames=3, latency=0.0)
    executor = SyncInferenceExecutor(engine, inference_horizon=None)

    np.testing.assert_allclose(executor.predict_action({"obs": "step0"}), [100.0])
    np.testing.assert_allclose(executor.predict_action({"obs": "step1"}), [101.0])
    np.testing.assert_allclose(executor.predict_action({"obs": "step2"}), [102.0])
    np.testing.assert_allclose(executor.predict_action({"obs": "step3"}), [200.0])


def test_async_executor_starts_background_at_delay_threshold():
    engine = MockEngine(num_frames=5, latency=0.01)
    executor = AsyncInferenceExecutor(engine, inference_horizon=4, inference_delay_steps=2)

    executor.predict_action({"obs": "step0"})
    assert executor.stats["num_background_inferences"] == 0

    executor.predict_action({"obs": "step1"})
    assert executor.stats["num_background_inferences"] == 0

    executor.predict_action({"obs": "step2"})
    time.sleep(0.05)

    stats = executor.stats
    assert stats["pending"]
    assert stats["num_background_inferences"] == 1
    assert engine.call_count == 2

    executor.shutdown()


def test_async_executor_reset():
    engine = MockEngine(num_frames=5, latency=0.0)
    executor = AsyncInferenceExecutor(engine, inference_horizon=5, inference_delay_steps=0)
    executor._background_enabled = False  # deterministic: no background generation

    executor.predict_action({"obs": "dummy"})
    executor.reset()
    assert len(executor._action_buffer) == 0

    executor.shutdown()


def test_async_executor_reset_drains_running_future_before_reuse():
    engine = BlockingSecondCallEngine()
    executor = AsyncInferenceExecutor(engine, inference_horizon=2, inference_delay_steps=1)

    executor.predict_action({"obs": "first"})
    executor.predict_action({"obs": "second"})
    assert engine.second_call_started.wait(timeout=1.0)

    timer = threading.Timer(0.05, engine.release_second_call.set)
    timer.start()
    t0 = time.monotonic()
    executor.reset()
    elapsed = time.monotonic() - t0
    timer.join(timeout=1.0)

    assert elapsed >= 0.03
    assert executor.stats["pending"] is False
    assert executor.stats["buffer_size"] == 0

    action = executor.predict_action({"obs": "after-reset"})
    np.testing.assert_allclose(action, np.ones(7) * 3.0)

    executor.shutdown()


def test_async_executor_clears_failed_pending_future_before_reuse():
    engine = FailingSecondCallEngine()
    executor = AsyncInferenceExecutor(engine, inference_horizon=2, inference_delay_steps=1)

    np.testing.assert_allclose(executor.predict_action({"obs": "step0"}), np.ones(7) * 1.0)
    np.testing.assert_allclose(executor.predict_action({"obs": "step1"}), np.ones(7) * 1.0)

    with pytest.raises(RuntimeError, match="background failure"):
        executor.predict_action({"obs": "step2"})

    assert executor.stats["pending"] is False
    action = executor.predict_action({"obs": "after-failure"})
    np.testing.assert_allclose(action, np.ones(7) * 3.0)

    executor.shutdown()


def test_async_executor_switches_at_inference_horizon_and_skips_stale_prefix():
    engine = IndexedEngine(num_frames=10, latency=0.0)
    executor = AsyncInferenceExecutor(engine, inference_horizon=6, inference_delay_steps=3)

    np.testing.assert_allclose(executor.predict_action({"obs": "step0"}), [100.0])
    np.testing.assert_allclose(executor.predict_action({"obs": "step1"}), [101.0])
    np.testing.assert_allclose(executor.predict_action({"obs": "step2"}), [102.0])
    np.testing.assert_allclose(executor.predict_action({"obs": "step3"}), [103.0])
    time.sleep(0.05)

    np.testing.assert_allclose(executor.predict_action({"obs": "step4"}), [104.0])
    np.testing.assert_allclose(executor.predict_action({"obs": "step5"}), [105.0])

    action = executor.predict_action({"obs": "step6"})
    np.testing.assert_allclose(action, [203.0])
    assert executor.stats["last_skip_steps"] == 3
    assert executor.stats["current_step"] == 7
    assert executor.stats["lead_time_steps"] == 3

    executor.shutdown()


def test_async_executor_stats():
    engine = MockEngine(num_frames=3, latency=0.0)
    executor = AsyncInferenceExecutor(engine, inference_horizon=3, inference_delay_steps=0)
    executor._background_enabled = False  # deterministic: no background generation

    executor.predict_action({"obs": "dummy"})
    stats = executor.stats
    assert stats["num_inferences"] == 1
    assert stats["num_sync_inferences"] == 1
    assert stats["buffer_size"] == 2
    assert stats["inference_horizon"] == 3
    assert stats["resolved_inference_delay_steps"] == 0

    executor.shutdown()


def test_async_executor_auto_delay_uses_half_inference_horizon():
    engine = MockEngine(num_frames=8, latency=0.0)
    executor = AsyncInferenceExecutor(engine, inference_horizon=6, inference_delay_steps=None)
    executor._background_enabled = False  # deterministic: no background generation

    executor.predict_action({"obs": "dummy"})
    assert executor.stats["resolved_inference_delay_steps"] == 3

    executor.shutdown()


def test_async_executor_rejects_inference_horizon_larger_than_action_horizon():
    engine = MockEngine(num_frames=3, latency=0.0)
    executor = AsyncInferenceExecutor(engine, inference_horizon=4, inference_delay_steps=0)

    with pytest.raises(ValueError, match="inference_horizon"):
        executor.predict_action({"obs": "dummy"})

    executor.shutdown()


@pytest.mark.parametrize("delay_steps", [4, 5])
def test_async_config_rejects_delay_at_or_larger_than_inference_horizon(delay_steps):
    cfg = {"mode": "async", "inference_horizon": 4, "inference_delay_steps": delay_steps}

    with pytest.raises(ValueError, match="inference_delay_steps"):
        normalize_execution_config(cfg)


@pytest.mark.parametrize("value", [1.2, "1.2"])
def test_execution_config_rejects_non_integral_horizon(value):
    cfg = {"mode": "async", "inference_horizon": value, "inference_delay_steps": 0}

    with pytest.raises(ValueError, match="inference_horizon must be an integer"):
        normalize_execution_config(cfg)


def test_unrelated_config_resolves_to_sync():
    from openwam.deploy.executors.async_executor import resolve_execution_config

    resolved = resolve_execution_config({"deploy": {"async_execution": {"enabled": True}}})
    assert resolved.enabled is False
    assert resolved.mode == "sync"


def test_sync_config_accepts_inference_horizon():
    cfg = normalize_execution_config({"mode": "sync", "inference_horizon": 10})

    assert cfg.inference_horizon == 10
    assert cfg.inference_delay_steps is None


def test_sync_config_rejects_async_delay():
    cfg = {"mode": "sync", "inference_delay_steps": 1}

    with pytest.raises(ValueError, match="inference_mode='async'"):
        normalize_execution_config(cfg)


def test_async_config_dataclass_roundtrip_preserves_delay_settings():
    cfg = normalize_execution_config({"mode": "async", "inference_horizon": 8, "inference_delay_steps": 4})

    roundtrip = normalize_execution_config(cfg)
    assert roundtrip.inference_horizon == 8
    assert roundtrip.inference_delay_steps == 4


def test_wam_policy_mode_none_keeps_sync_buffer_path():
    from openwam.deploy.policy import WAMPolicy

    engine = MockEngine(num_frames=3, latency=0.0)
    cfg = SimpleNamespace()
    policy = WAMPolicy(engine=engine, cfg=cfg, execution_config={"mode": "sync"})

    assert policy._async is False
    first = policy.predict_action({"prompt": "test"})
    second = policy.predict_action({"prompt": "test"})

    np.testing.assert_allclose(first, np.ones(7) * 1.0)
    np.testing.assert_allclose(second, np.ones(7) * 1.0)
    assert engine.call_count == 1


def test_wam_policy_sync_reads_inference_horizon_from_execution_config():
    from openwam.deploy.policy import WAMPolicy

    engine = MockEngine(num_frames=4, latency=0.0)
    policy = WAMPolicy(
        engine=engine,
        cfg=SimpleNamespace(),
        execution_config={"mode": "sync", "inference_horizon": 2},
    )

    policy.predict_action({"prompt": "step0"})
    policy.predict_action({"prompt": "step1"})
    policy.predict_action({"prompt": "step2"})
    assert engine.call_count == 2
