"""Tests for async inference executor."""

import time
import numpy as np

from open_wam.inference.optimizations.async_executor import AsyncInferenceExecutor


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
        actions = np.ones((self.num_frames, self.action_dim)) * self.call_count
        return {"video": None, "actions": actions}


def test_async_executor_basic():
    engine = MockEngine(num_frames=5, latency=0.0)
    executor = AsyncInferenceExecutor(engine, chunk_size=5, prefetch=False)

    action = executor.predict_action({"obs": "dummy"})
    assert action.shape == (7,)
    assert engine.call_count == 1
    executor.shutdown()


def test_async_executor_buffer_exhaustion():
    engine = MockEngine(num_frames=3, latency=0.0)
    executor = AsyncInferenceExecutor(engine, chunk_size=3, prefetch=False)

    # First 3 actions from chunk 1
    for i in range(3):
        action = executor.predict_action({"obs": "dummy"})
        np.testing.assert_allclose(action, np.ones(7) * 1.0)
    assert engine.call_count == 1

    # 4th action triggers new inference (chunk 2)
    action = executor.predict_action({"obs": "dummy"})
    np.testing.assert_allclose(action, np.ones(7) * 2.0)
    assert engine.call_count == 2

    executor.shutdown()


def test_async_executor_reset():
    engine = MockEngine(num_frames=5, latency=0.0)
    executor = AsyncInferenceExecutor(engine, chunk_size=5, prefetch=False)

    executor.predict_action({"obs": "dummy"})
    executor.reset()
    assert len(executor._action_buffer) == 0

    executor.shutdown()


def test_async_executor_stats():
    engine = MockEngine(num_frames=3, latency=0.0)
    executor = AsyncInferenceExecutor(engine, chunk_size=3, prefetch=False)

    executor.predict_action({"obs": "dummy"})
    stats = executor.stats
    assert stats["num_inferences"] == 1
    assert stats["buffer_size"] == 2  # 3 generated, 1 consumed

    executor.shutdown()


def test_async_executor_prefetch():
    """With prefetch, next chunk should start computing before buffer empties."""
    engine = MockEngine(num_frames=4, latency=0.01)
    executor = AsyncInferenceExecutor(engine, chunk_size=4, prefetch=True)

    # Consume first 3 actions (half buffer = 2, so prefetch triggers at action 3)
    for _ in range(3):
        executor.predict_action({"obs": "dummy"})

    # Give prefetch thread time to start
    time.sleep(0.05)

    # Next action should come from buffer (possibly from prefetched chunk)
    action = executor.predict_action({"obs": "dummy"})
    assert action.shape == (7,)

    executor.shutdown()
