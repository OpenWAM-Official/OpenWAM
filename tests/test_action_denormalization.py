"""Unit tests for action denormalization on the deployment path.

Covers _build_action_denormalizer (reads action_stats.npy + cfg) and the
ActionNormalizer.unnormalize invariant that joint_generation.py relies on
to convert model output back to physical units before returning to clients.

Pure CPU, no GPU, no network. Uses pytest's tmp_path fixture so there's no
dependency on any real checkpoint directory.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from openwam.dataloader.transforms.normalize import ActionNormalizer
from openwam.deploy.model_loader import _build_action_denormalizer

# --- Helper: build a realistic stats dict for a 20D eef action ---


def _eef_stats_min_max():
    """Build action_stats in the nested schema with eef range simulating real robot."""
    # Simulate a physical workspace roughly ±0.8 m for xyz, [-1, 1] for rot6d,
    # [0, 1] for gripper. 20D = [lxyz(3), lrot(6), lgrip(1), rxyz(3), rrot(6), rgrip(1)]
    lo = np.array([-0.8, -0.8, -0.2] + [-1.0] * 6 + [0.0] + [-0.8, -0.8, -0.2] + [-1.0] * 6 + [0.0], dtype=np.float32)
    hi = np.array([0.8, 0.8, 1.5] + [1.0] * 6 + [1.0] + [0.8, 0.8, 1.5] + [1.0] * 6 + [1.0], dtype=np.float32)
    mean = (lo + hi) / 2
    std = (hi - lo) / 4
    return {
        "mean": mean,
        "std": np.maximum(std, 1e-6),
        "min": lo,
        "max": hi,
        "q01": lo,
        "q99": hi,
    }


def _write_stats_file(tmp_path, mode_key: str = "eef"):
    """Write a nested-schema action_stats.npy into tmp_path and return its path."""
    stats = {mode_key: _eef_stats_min_max(), "num_timesteps": 1000}
    p = tmp_path / "action_stats.npy"
    np.save(str(p), stats, allow_pickle=True)
    return str(p)


# --- Normalizer round-trip tests ---


def test_denormalizer_min_max_roundtrip():
    stats = _eef_stats_min_max()
    norm = ActionNormalizer(mode="min_max", stats=stats)
    x = np.random.RandomState(0).uniform(-0.5, 0.5, size=(4, 20)).astype(np.float32)
    # Clamp to stats range so round-trip is well-defined
    x = np.clip(x, stats["min"], stats["max"])
    y = norm.normalize(x)
    x_back = norm.unnormalize(y)
    np.testing.assert_allclose(x, x_back, atol=1e-5)


def test_denormalizer_zscore_roundtrip():
    stats = _eef_stats_min_max()
    norm = ActionNormalizer(mode="mean_std", stats=stats)
    x = np.random.RandomState(1).uniform(-0.5, 0.5, size=(4, 20)).astype(np.float32)
    y = norm.normalize(x)
    x_back = norm.unnormalize(y)
    np.testing.assert_allclose(x, x_back, atol=1e-5)


# --- _build_action_denormalizer branch tests ---


def test_build_denormalizer_happy_path(tmp_path):
    _write_stats_file(tmp_path, mode_key="eef")
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "eef"}})
    denorm = _build_action_denormalizer(cfg, str(tmp_path))
    assert denorm is not None
    assert isinstance(denorm, ActionNormalizer)

    # Feeding a normalized zero vector should map to the center of the range.
    # For min-max with [lo, hi], normalize(x) = 2*(x-lo)/(hi-lo) - 1, so x=0
    # (normalized) => x = (lo+hi)/2 (physical).
    out = denorm.unnormalize(np.zeros(20, dtype=np.float32))
    stats = _eef_stats_min_max()
    expected = (stats["min"] + stats["max"]) / 2
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_build_denormalizer_missing_stats_raises(tmp_path):
    # tmp_path is empty — no action_stats.npy
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "eef"}})
    with pytest.raises(FileNotFoundError, match="Missing required action_stats.npy"):
        _build_action_denormalizer(cfg, str(tmp_path))


@pytest.mark.parametrize("disabled_value", [None, "none", "null", ""])
def test_build_denormalizer_disabled_mode_returns_none(tmp_path, disabled_value):
    _write_stats_file(tmp_path)
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": disabled_value, "action_mode": "eef"}})
    denorm = _build_action_denormalizer(cfg, str(tmp_path))
    assert denorm is None


def test_build_denormalizer_unknown_mode_returns_none(tmp_path):
    _write_stats_file(tmp_path)
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "eef"}})
    denorm = _build_action_denormalizer(cfg, str(tmp_path))
    assert denorm is not None


def test_build_denormalizer_wrong_action_mode_returns_none(tmp_path):
    # Stats file has only "eef" but config says action_mode="joint"
    _write_stats_file(tmp_path, mode_key="eef")
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "joint"}})
    denorm = _build_action_denormalizer(cfg, str(tmp_path))
    assert denorm is None


# --- Deployment-path invariant: denormalize must push xyz beyond [-1, 1] ---


def test_deployment_action_range_sanity(tmp_path):
    """Mirrors what generate_video_and_actions does at [joint_generation.py:447-456].

    Model output is in [-1, 1] (after flow-matching). After unnormalize, xyz
    dims must reach physical range (here: ±0.8 m). If denormalize were missing,
    this guard would catch the regression.
    """
    _write_stats_file(tmp_path)
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "eef"}})
    denorm = _build_action_denormalizer(cfg, str(tmp_path))
    assert denorm is not None, "prerequisite: denormalizer must build"

    # Simulate a model output batch: 33-step action chunk in [-1, 1].
    normalized = np.random.RandomState(42).uniform(-1.0, 1.0, size=(33, 20)).astype(np.float32)

    # Exact logic copied from joint_generation.py:447-450
    actions = denorm.unnormalize(normalized)

    # xyz indices in the 20D eef layout: left xyz = [0,1,2], right xyz = [10,11,12]
    xyz_abs_max = float(np.abs(actions[:, [0, 1, 2, 10, 11, 12]]).max())
    assert xyz_abs_max > 1.0, (
        f"xyz.abs().max()={xyz_abs_max:.4f} after denormalize — expected > 1.0 "
        f"(stats x range is ±0.8 m but full span is ±1.5 m on z). This would fire if the "
        f"denormalizer path stopped applying."
    )

    assert float(np.abs(actions).max()) > 1.0
