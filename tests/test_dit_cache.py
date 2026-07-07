"""Tests for DiT velocity caching."""

import torch

from openwam.deploy.optimizations.dit_cache import DiTVelocityCache


def test_cache_init():
    cache = DiTVelocityCache(cosine_threshold=0.99)
    assert cache.cosine_threshold == 0.99
    assert cache.skip_rate == 0.0


def test_first_step_always_recomputes():
    cache = DiTVelocityCache()
    assert cache.should_recompute(0.9) is True


def test_second_step_always_recomputes():
    """Need two velocity samples before comparison is possible."""
    cache = DiTVelocityCache()
    assert cache.should_recompute(0.9) is True
    cache.update(torch.randn(1, 4, 8, 8), 0.9)
    # Second step: prev_velocity is None (only cached_velocity set)
    assert cache.should_recompute(0.8) is True


def test_cache_hit_with_similar_velocities():
    cache = DiTVelocityCache(cosine_threshold=0.9)
    v = torch.randn(1, 4, 8, 8)

    cache.update(v, 0.9)  # step 1
    cache.update(v * 1.01, 0.85)  # step 2: very similar to step 1

    # step 3: should skip (v and v*1.01 have cosine sim ≈ 1.0)
    assert cache.should_recompute(0.8) is False
    cached = cache.get_cached()
    assert cached.shape == v.shape


def test_cache_hit_can_reuse_joint_action_prediction():
    cache = DiTVelocityCache(cosine_threshold=0.9)
    v = torch.randn(1, 4, 8, 8)
    a = torch.randn(1, 8, 7)

    cache.update(v, 0.9, a)
    cache.update(v * 1.01, 0.85, a * 1.01)

    assert cache.should_recompute(0.8, require_action=True) is False
    assert torch.allclose(cache.get_cached(), v * 1.01)
    assert torch.allclose(cache.get_cached_action(), a * 1.01)


def test_cache_clones_forward_outputs():
    cache = DiTVelocityCache(cosine_threshold=0.9)
    v = torch.ones(1, 4, 8, 8)
    a = torch.ones(1, 8, 7)

    cache.update(v, 0.9, a)
    cached_v = cache.get_cached()
    cached_a = cache.get_cached_action()

    assert cached_v.data_ptr() != v.data_ptr()
    assert cached_a is not None
    assert cached_a.data_ptr() != a.data_ptr()

    v.zero_()
    a.zero_()

    assert torch.all(cached_v == 1)
    assert torch.all(cached_a == 1)


def test_cache_recomputes_when_action_velocity_diverges():
    cache = DiTVelocityCache(cosine_threshold=0.9)
    v = torch.randn(1, 4, 8, 8)
    a = torch.randn(1, 8, 7)

    cache.update(v, 0.9, a)
    cache.update(v * 1.01, 0.85, -a)

    assert cache.should_recompute(0.8, require_action=True) is True
    assert cache.stats["total_skips"] == 0


def test_cache_requires_action_when_joint_caller_needs_it():
    cache = DiTVelocityCache(cosine_threshold=0.9)
    v = torch.randn(1, 4, 8, 8)

    cache.update(v, 0.9)
    cache.update(v * 1.01, 0.85)

    assert cache.should_recompute(0.8, require_action=True) is True


def test_cache_miss_with_different_velocities():
    cache = DiTVelocityCache(cosine_threshold=0.99)
    v1 = torch.randn(1, 4, 8, 8)
    v2 = torch.randn(1, 4, 8, 8)  # totally different

    cache.update(v1, 0.9)
    cache.update(v2, 0.8)

    # v1 and v2 are random → cosine sim ≈ 0, should recompute
    assert cache.should_recompute(0.7) is True


def test_max_consecutive_skips():
    cache = DiTVelocityCache(cosine_threshold=0.5, max_consecutive_skips=2)
    v = torch.ones(1, 4, 8, 8)

    cache.update(v, 0.9)
    cache.update(v, 0.85)

    # Skip 1
    assert cache.should_recompute(0.8) is False
    # Skip 2
    assert cache.should_recompute(0.75) is False
    # Forced recompute after max_consecutive_skips
    assert cache.should_recompute(0.7) is True


def test_reset_clears_state():
    cache = DiTVelocityCache()
    cache.update(torch.randn(1, 4, 8, 8), 0.9)
    cache.reset()
    assert cache._cached_velocity is None
    assert cache._cached_action_velocity is None
    assert cache._prev_action_velocity is None
    assert cache.skip_rate == 0.0


def test_stats():
    cache = DiTVelocityCache(cosine_threshold=0.5)
    v = torch.ones(1, 4)

    cache.should_recompute(0.9)  # step 1: recompute
    cache.update(v, 0.9)
    cache.should_recompute(0.85)  # step 2: recompute (only 1 cached)
    cache.update(v, 0.85)
    cache.should_recompute(0.8)  # step 3: skip (similar)

    stats = cache.stats
    assert stats["total_steps"] == 3
    assert stats["total_skips"] == 1
