"""Stats-related tests for the LeRobot v3 dataloader.

Covers (no GPU / real dataset required):
  - _normalize / denormalize_action: none / min-max / z-score + roundtrip
  - LeRobot3Dataset startup guards: normalize_mode + missing stats raises
  - load_action_stats: post-transform single "action" key + .npy {eef: ...} schema
  - MultiTaskLeRobot3Dataset per-subset stats routing
  - Union-stats / shared-stats path aggregation, deploy schema round-trip
  - compute_multitask_lerobot_v3_stats / save_union_stats_npy
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests.test_lerobot_v3_helpers import (
    _create_fake_task,
    _fake_task_and_ds,
    _make_lerobot3,
    _mkdir_subs,
    _write_union_stats_npy,
)

# ---------------------------------------------------------------------------
# _normalize / denormalize_action
# ---------------------------------------------------------------------------


def test_normalize_none_is_noop(tmp_path):
    ds = _fake_task_and_ds(tmp_path, normalize_mode="none")
    actions = np.array([[0.5] * 14], dtype=np.float32)
    out = ds._normalize(actions)
    np.testing.assert_array_equal(out, actions)


def test_normalize_minmax_range(tmp_path):
    ds = _fake_task_and_ds(tmp_path, normalize_mode="min-max")
    # With min=-1, max=1 stats and input in [-1, 1], output must be in [-1, 1]
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1.0, 1.0, (10, 14)).astype(np.float32)
    out = ds._normalize(actions)
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_normalize_zscore_shifts_mean(tmp_path):
    ds = _fake_task_and_ds(tmp_path, normalize_mode="z-score")
    # With mean=0, std=1 stats, the output should equal the input
    actions = np.array([[1.0, 2.0, -1.0] + [0.0] * 11], dtype=np.float32)
    out = ds._normalize(actions)
    np.testing.assert_allclose(out, actions, atol=1e-5)


@pytest.mark.parametrize("normalize_mode", ["min-max", "z-score"])
def test_denormalize_roundtrip(tmp_path, normalize_mode):
    ds = _fake_task_and_ds(tmp_path, normalize_mode=normalize_mode)
    rng = np.random.default_rng(2 if normalize_mode == "min-max" else 3)
    if normalize_mode == "min-max":
        original = rng.uniform(-1.0, 1.0, (5, 14)).astype(np.float32)
    else:
        original = rng.standard_normal((5, 14)).astype(np.float32)
    normed = ds._normalize(original)
    recovered = ds.denormalize_action(normed)
    np.testing.assert_allclose(recovered, original, atol=1e-5)


# ---------------------------------------------------------------------------
# Stats loading
# ---------------------------------------------------------------------------


def test_load_action_stats_post_transform_single_action_key(tmp_path):
    """Regression: OXE's eef_stats.json has a single top-level "action" key with
    the post-transform 20-D vector, but raw action_fields are
    ["action.cartesian_position", "action.gripper_position"] (7-D total).
    The loader must detect the post-transform format and load directly,
    instead of iterating raw fields and falling back to zeros (which produced
    a 7-D vector and crashed the trainer with a 20 vs 7 size mismatch)."""
    raw_fields = [("action.cartesian_position", [6]), ("action.gripper_position", [1])]
    out_dim = 20
    target_mean = np.linspace(-0.5, 0.5, out_dim, dtype=np.float32)
    target_std = np.linspace(0.1, 1.5, out_dim, dtype=np.float32)
    target_min = -np.ones(out_dim, dtype=np.float32)
    target_max = np.ones(out_dim, dtype=np.float32)

    _create_fake_task(str(tmp_path), n_episodes=2, ep_len=20, action_fields=raw_fields)
    # Overwrite the auto-generated meta/stats.json layout: write the
    # post-transform stats file the way lerobot_v3_stats_computation emits
    # them when action_transform is set.
    eef_stats = {
        "num_timesteps": 40,
        "action": {
            "mean": target_mean.tolist(), "std": target_std.tolist(),
            "min": target_min.tolist(), "max": target_max.tolist(),
        },
    }
    with open(os.path.join(str(tmp_path), "meta", "eef_stats.json"), "w") as f:
        json.dump(eef_stats, f)

    ds = _make_lerobot3(
        tmp_path,
        action_fields=[f for f, _ in raw_fields],
        normalize_mode="min-max",
        action_transform=lambda x: np.zeros((len(x), out_dim), dtype=np.float32),
        action_out_dim=out_dim,
        action_stats_path=None,
    )

    stats = ds.action_stats
    assert stats is not None
    assert stats["mean"].shape == (out_dim,), f"expected {out_dim}-D, got {stats['mean'].shape}"
    np.testing.assert_allclose(stats["mean"], target_mean)
    np.testing.assert_allclose(stats["std"], target_std)
    np.testing.assert_allclose(stats["min"], target_min)
    np.testing.assert_allclose(stats["max"], target_max)


def test_normalize_mode_raises_when_eef_stats_missing(tmp_path):
    """normalize_mode + action_transform with no eef_stats.json must raise at init.
    Auto-compute was removed to prevent DDP multi-rank file write races."""
    _create_fake_task(str(tmp_path), n_episodes=2, ep_len=20)
    with pytest.raises(RuntimeError, match="action_stats"):
        _make_lerobot3(
            tmp_path,
            normalize_mode="min-max",
            action_transform=lambda x: x,
            action_out_dim=14,
        )


# ---------------------------------------------------------------------------
# Per-subset stats routing (MultiTaskLeRobot3Dataset)
# ---------------------------------------------------------------------------


def _build_two_subset_minmax(tmp_path):
    """Two LeRobot3 subsets with min-max stats; subset_b overridden to [0,10] range."""
    d1, d2 = _mkdir_subs(tmp_path, "d1", "d2")
    _create_fake_task(str(d1), n_episodes=2, ep_len=20)
    _create_fake_task(str(d2), n_episodes=2, ep_len=20)
    ds1 = _make_lerobot3(d1, normalize_mode="min-max", task_name="subset_a")
    ds2 = _make_lerobot3(d2, normalize_mode="min-max", task_name="subset_b")
    ds2._action_stats = {
        "mean": np.full(14, 5.0, dtype=np.float32),
        "std": np.full(14, 1.0, dtype=np.float32),
        "min": np.full(14, 0.0, dtype=np.float32),
        "max": np.full(14, 10.0, dtype=np.float32),
    }
    return ds1, ds2


def test_multitask_subset_stats_are_per_dataset(tmp_path):
    """subset_stats should be keyed by task_name and reflect each subset's own stats."""
    from openwam.dataloader.lerobot_v3_base import MultiTaskLeRobot3Dataset

    ds1, ds2 = _build_two_subset_minmax(tmp_path)
    multi = MultiTaskLeRobot3Dataset([ds1, ds2])
    per_subset = multi.subset_stats
    assert set(per_subset.keys()) == {"subset_a", "subset_b"}
    assert per_subset["subset_a"] is ds1.action_stats
    assert per_subset["subset_b"] is ds2.action_stats
    # Backward-compat: action_stats still returns the first subset's stats.
    assert multi.action_stats is ds1.action_stats


def test_multitask_denormalize_routes_by_task_name(tmp_path):
    """denormalize_action(task_name=...) should use the matching subset's stats."""
    from openwam.dataloader.lerobot_v3_base import MultiTaskLeRobot3Dataset

    ds1, ds2 = _build_two_subset_minmax(tmp_path)
    multi = MultiTaskLeRobot3Dataset([ds1, ds2])
    normed = np.zeros((2, 14), dtype=np.float32)  # min-max midpoint

    # Routing by task_name picks each subset's stats.
    out_a = multi.denormalize_action(normed, task_name="subset_a")
    out_b = multi.denormalize_action(normed, task_name="subset_b")
    np.testing.assert_allclose(out_a, np.zeros_like(out_a))  # midpoint of [-1, 1]
    np.testing.assert_allclose(out_b, np.full_like(out_b, 5.0))  # midpoint of [0, 10]

    # H3: multi-subset + omitted task_name must fail loudly. Silently
    # using _datasets[0]'s stats would send wrong-magnitude actions to
    # the robot at inference time — see denormalize_action docstring.
    with pytest.raises(ValueError, match="task_name is required"):
        multi.denormalize_action(normed)

    # Unknown task_name surfaces as KeyError, not silent wrong-stats.
    with pytest.raises(KeyError, match="unknown task_name"):
        multi.denormalize_action(normed, task_name="ghost")


# ---------------------------------------------------------------------------
# Union stats (RoboTwin-pattern: shared .npy across sub-datasets, deploy-ready)
# ---------------------------------------------------------------------------


def _make_eef_sub_dataset(task_dir, stats_path, task_name):
    """Build a LeRobot3Dataset wired for AGIBOT EEF stats — used by union-stats tests."""
    from openwam.dataloader.agibot import AGIBOT_ACTION_DIM_EEF, make_eef_transform

    return _make_lerobot3(
        task_dir,
        normalize_mode="min-max",
        action_stats_path=stats_path,
        action_transform=make_eef_transform("xyzw"),
        action_out_dim=AGIBOT_ACTION_DIM_EEF,
        task_name=task_name,
    )


def _build_eef_sub_datasets(tmp_path, eef_dim, *, shared=True):
    """Create two LeRobot3Dataset subsets (sub_0, sub_1) for union-stats tests.

    - shared=True: both subsets point at one tmp_path/"union.npy" (returned as 2nd item).
    - shared=False: each subset gets its own stats_i.npy (returned path is None).
    """
    if shared:
        union_path = os.path.join(str(tmp_path), "union.npy")
        _write_union_stats_npy(union_path, dim=eef_dim)
    sub_datasets = []
    for i, sub in enumerate(_mkdir_subs(tmp_path, "sub_0", "sub_1")):
        task_dir = _create_fake_task(str(sub))
        if shared:
            stats_path = union_path
        else:
            stats_path = os.path.join(str(tmp_path), f"stats_{i}.npy")
            _write_union_stats_npy(stats_path, dim=eef_dim)
        sub_datasets.append(_make_eef_sub_dataset(task_dir, stats_path, f"sub_{i}"))
    return sub_datasets, (union_path if shared else None)


def test_load_action_stats_reads_npy_eef_schema(tmp_path):
    """LeRobot3Dataset._load_action_stats now supports .npy with {eef: ...} schema."""
    from openwam.dataloader.agibot import AGIBOT_ACTION_DIM_EEF, make_eef_transform

    task_dir = _create_fake_task(str(tmp_path))
    union_path = os.path.join(str(tmp_path), "union.npy")
    _write_union_stats_npy(union_path, dim=AGIBOT_ACTION_DIM_EEF)

    ds = _make_lerobot3(
        task_dir,
        normalize_mode="min-max",
        action_stats_path=union_path,
        action_transform=make_eef_transform("xyzw"),
        action_out_dim=AGIBOT_ACTION_DIM_EEF,
    )

    assert ds._action_stats is not None
    assert ds._action_stats["mean"].shape == (AGIBOT_ACTION_DIM_EEF,)
    # The exposed action_stats_path should point at the npy we loaded
    assert ds.action_stats_path == union_path


def test_lerobot3_exposes_action_stats_path(tmp_path):
    """``self.action_stats_path`` reflects the actually-loaded stats file (json case)."""
    task_dir = _create_fake_task(str(tmp_path))
    ds = _make_lerobot3(task_dir, normalize_mode="min-max")
    # Falls back to meta/stats.json (the only one written by _create_fake_task)
    assert ds.action_stats_path == os.path.join(task_dir, "meta", "stats.json")


def test_lerobot3_action_stats_path_none_when_normalize_disabled(tmp_path):
    """When normalize_mode is null and no stats are loaded, action_stats_path is None."""
    task_dir = _create_fake_task(str(tmp_path))
    # Remove stats.json so nothing loads even at normalize_mode=null
    os.remove(os.path.join(task_dir, "meta", "stats.json"))
    ds = _make_lerobot3(task_dir, normalize_mode=None)
    assert ds.action_stats_path is None


def test_multitask_aggregates_shared_action_stats_path(tmp_path):
    """All sub-datasets sharing one stats path → wrapper exposes it + sets _stats_are_shared."""
    from openwam.dataloader.agibot import AGIBOT_ACTION_DIM_EEF
    from openwam.dataloader.lerobot_v3_base import MultiTaskLeRobot3Dataset

    sub_datasets, union_path = _build_eef_sub_datasets(tmp_path, AGIBOT_ACTION_DIM_EEF, shared=True)
    wrapper = MultiTaskLeRobot3Dataset(sub_datasets)
    assert wrapper.action_stats_path == union_path
    assert wrapper._stats_are_shared is True


def test_multitask_per_subset_stats_keeps_strict_mode(tmp_path):
    """Different stats paths across subsets → _stats_are_shared=False, action_stats_path=None."""
    from openwam.dataloader.agibot import AGIBOT_ACTION_DIM_EEF
    from openwam.dataloader.lerobot_v3_base import MultiTaskLeRobot3Dataset

    sub_datasets, _ = _build_eef_sub_datasets(tmp_path, AGIBOT_ACTION_DIM_EEF, shared=False)
    wrapper = MultiTaskLeRobot3Dataset(sub_datasets)
    assert wrapper.action_stats_path is None
    assert wrapper._stats_are_shared is False


def test_multitask_denormalize_no_task_name_succeeds_with_shared_stats(tmp_path):
    """When stats are shared, denormalize_action(x) without task_name works (RoboTwin pattern)."""
    from openwam.dataloader.agibot import AGIBOT_ACTION_DIM_EEF
    from openwam.dataloader.lerobot_v3_base import MultiTaskLeRobot3Dataset

    sub_datasets, _ = _build_eef_sub_datasets(tmp_path, AGIBOT_ACTION_DIM_EEF, shared=True)
    wrapper = MultiTaskLeRobot3Dataset(sub_datasets)
    action = np.zeros((5, AGIBOT_ACTION_DIM_EEF), dtype=np.float32)
    # Should NOT raise: shared stats are unambiguous
    result = wrapper.denormalize_action(action)
    assert result.shape == (5, AGIBOT_ACTION_DIM_EEF)


def test_multitask_denormalize_no_task_name_raises_with_per_subset_stats(tmp_path):
    """Per-subset stats with no task_name still raises (preserves safety guard)."""
    from openwam.dataloader.agibot import AGIBOT_ACTION_DIM_EEF
    from openwam.dataloader.lerobot_v3_base import MultiTaskLeRobot3Dataset

    sub_datasets, _ = _build_eef_sub_datasets(tmp_path, AGIBOT_ACTION_DIM_EEF, shared=False)
    wrapper = MultiTaskLeRobot3Dataset(sub_datasets)
    action = np.zeros((5, AGIBOT_ACTION_DIM_EEF), dtype=np.float32)
    with pytest.raises(ValueError, match="task_name is required"):
        wrapper.denormalize_action(action)


def test_compute_multitask_lerobot_v3_stats_aggregates_across_roots(tmp_path):
    """compute_multitask_lerobot_v3_stats reads many task roots and emits one block."""
    from openwam.dataloader.lerobot_v3_stats_computation import (
        compute_multitask_lerobot_v3_stats,
    )

    roots = [
        _create_fake_task(str(sub), n_episodes=2, ep_len=10)
        for sub in _mkdir_subs(tmp_path, "task_0", "task_1", "task_2")
    ]

    result = compute_multitask_lerobot_v3_stats(
        task_roots=roots,
        action_fields=["action"],
        field_dims=[14],
        action_transform=None,
        action_out_dim=None,
        action_mode="eef",
        show_progress=False,
    )

    assert "eef" in result
    assert "num_timesteps" in result
    assert result["num_timesteps"] == 3 * 2 * 10  # 3 tasks * 2 eps * 10 frames
    for k in ("mean", "std", "min", "max"):
        assert k in result["eef"]
    assert len(result["eef"]["mean"]) == 14


def test_compute_multitask_requires_field_dims_when_no_transform():
    from openwam.dataloader.lerobot_v3_stats_computation import (
        compute_multitask_lerobot_v3_stats,
    )

    with pytest.raises(ValueError, match="field_dims is required"):
        compute_multitask_lerobot_v3_stats(
            task_roots=["/dummy"],
            action_fields=["action"],
            field_dims=None,
            action_transform=None,
        )


def test_compute_multitask_empty_task_roots_raises():
    from openwam.dataloader.lerobot_v3_stats_computation import (
        compute_multitask_lerobot_v3_stats,
    )

    with pytest.raises(ValueError, match="task_roots is empty"):
        compute_multitask_lerobot_v3_stats(
            task_roots=[],
            action_fields=["action"],
            field_dims=[14],
        )


def test_save_union_stats_npy_writes_deploy_schema(tmp_path):
    """save_union_stats_npy round-trips through np.load with allow_pickle."""
    from openwam.dataloader.lerobot_v3_stats_computation import save_union_stats_npy

    path = os.path.join(str(tmp_path), "union.npy")
    stats = {
        "eef": {
            "mean": [0.0] * 20,
            "std": [1.0] * 20,
            "min": [-1.0] * 20,
            "max": [1.0] * 20,
        },
        "num_timesteps": 1234,
    }
    save_union_stats_npy(path, stats)
    assert os.path.exists(path)

    # Deploy-compatible: np.load + .item() recovers the dict
    loaded = np.load(path, allow_pickle=True).item()
    assert "eef" in loaded
    assert loaded["num_timesteps"] == 1234
    assert len(loaded["eef"]["mean"]) == 20
