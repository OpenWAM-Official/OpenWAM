"""Discovery, filtering, and OXE-registry tests for the LeRobot v3 dataloader.

Covers (no GPU / real dataset required):
  - discover_lerobot_v3_roots: single-root, parent mode, skip-incomplete, strict, empty
  - filter_discovered_roots: passthrough, train allowlist, holdout, prefix mode
  - discover_agibot_tasks / discover_galaxea_tasks
  - MultiTaskLeRobot3Dataset __init__ rejections (heterogeneous action_dim,
    duplicate task_name)
  - OXE registry end-to-end: build_dataset over yaml-like configs, subset
    merging, disable-skip, error paths, action_format="full"
"""

from __future__ import annotations

import os

import pytest

from tests.test_lerobot_v3_helpers import (
    _OXE_CAM,
    _create_fake_task,
    _make_lerobot3,
    _make_oxe_task,
    _mkdir_subs,
    _mock_video,
    _oxe_config,
    _oxe_subset,
)

# ---------------------------------------------------------------------------
# discover_lerobot_v3_roots
# ---------------------------------------------------------------------------


def test_discover_lerobot_v3_roots_single_root(tmp_path):
    """When dataset_dir IS a LeRobot v3 root, return one entry."""
    from openwam.dataloader.lerobot_v3_base import discover_lerobot_v3_roots

    tmpdir = str(tmp_path)
    _create_fake_task(tmpdir, n_episodes=1, ep_len=5)
    roots = discover_lerobot_v3_roots(tmpdir, max_depth=1)
    assert len(roots) == 1
    display_name, path = roots[0]
    assert display_name == os.path.basename(os.path.normpath(tmpdir))
    assert path == tmpdir


def test_discover_lerobot_v3_roots_parent_mode(tmp_path):
    """When dataset_dir contains multiple roots, return them all."""
    from openwam.dataloader.lerobot_v3_base import discover_lerobot_v3_roots

    parent = str(tmp_path)
    # Two complete subsets side-by-side under parent.
    for sub in ("droid", "bridge"):
        _create_fake_task(os.path.join(parent, sub), n_episodes=1, ep_len=5)
    roots = discover_lerobot_v3_roots(parent, max_depth=1)
    names = sorted(name for name, _ in roots)
    assert names == ["bridge", "droid"]
    for _, path in roots:
        assert os.path.isfile(os.path.join(path, "meta", "info.json"))


def test_discover_lerobot_v3_roots_skips_incomplete(tmp_path):
    """Incomplete child dirs (no meta/info.json) are skipped with INFO log."""
    from openwam.dataloader.lerobot_v3_base import discover_lerobot_v3_roots

    parent = str(tmp_path)
    _create_fake_task(os.path.join(parent, "droid"), n_episodes=1, ep_len=5)
    # Half-prepared sibling — has a top-level dir but no meta/info.json.
    os.makedirs(os.path.join(parent, "bridge_pending"), exist_ok=True)

    roots = discover_lerobot_v3_roots(parent, max_depth=1, skip_incomplete=True)
    names = [name for name, _ in roots]
    assert names == ["droid"]


def test_discover_lerobot_v3_roots_strict_raises_on_incomplete(tmp_path):
    """skip_incomplete=False makes incomplete entries hard-fail the scan."""
    from openwam.dataloader.lerobot_v3_base import discover_lerobot_v3_roots

    parent = str(tmp_path)
    _create_fake_task(os.path.join(parent, "droid"), n_episodes=1, ep_len=5)
    os.makedirs(os.path.join(parent, "bridge_pending"), exist_ok=True)

    with pytest.raises(RuntimeError, match="missing meta/info.json"):
        discover_lerobot_v3_roots(parent, max_depth=1, skip_incomplete=False)


def test_discover_lerobot_v3_roots_no_roots_raises(tmp_path):
    """A parent with no LeRobot v3 root inside should raise."""
    from openwam.dataloader.lerobot_v3_base import discover_lerobot_v3_roots

    parent = str(tmp_path)
    # Only an empty subdirectory; no info.json anywhere.
    os.makedirs(os.path.join(parent, "empty"), exist_ok=True)
    with pytest.raises(RuntimeError, match="No LeRobot v3 roots"):
        discover_lerobot_v3_roots(parent, max_depth=1, skip_incomplete=True)


# ---------------------------------------------------------------------------
# filter_discovered_roots
# ---------------------------------------------------------------------------


def test_filter_no_filters_passthrough():
    """train_tasks=None, holdout_tasks=None should be a no-op."""
    from openwam.dataloader.lerobot_v3_base import filter_discovered_roots

    roots = [("a", "/p/a"), ("b", "/p/b"), ("c", "/p/c")]
    assert filter_discovered_roots(roots) == roots


def test_filter_train_tasks_allowlist_single_and_prefix():
    """train_tasks should match by full name and (with match_prefix) by top-level."""
    from openwam.dataloader.lerobot_v3_base import filter_discovered_roots

    # Single-level: only "b" survives.
    roots = [("a", "/p/a"), ("b", "/p/b"), ("c", "/p/c")]
    out = filter_discovered_roots(roots, train_tasks=["b"])
    assert out == [("b", "/p/b")]

    # Two-level with match_prefix: "task_a" should admit every ep_range under task_a.
    nested = [
        ("task_a/0_50", "/p/task_a/0_50"),
        ("task_a/50_100", "/p/task_a/50_100"),
        ("task_b/0_50", "/p/task_b/0_50"),
    ]
    out = filter_discovered_roots(nested, train_tasks=["task_a"], match_prefix=True)
    assert [n for n, _ in out] == ["task_a/0_50", "task_a/50_100"]

    # Two-level: full-name allowlist still works alongside prefix matching.
    out = filter_discovered_roots(nested, train_tasks=["task_b/0_50"], match_prefix=True)
    assert [n for n, _ in out] == ["task_b/0_50"]


def test_filter_holdout_tasks_excludes_by_full_or_prefix():
    """holdout_tasks should drop matches by full name AND (with prefix) by top-level."""
    from openwam.dataloader.lerobot_v3_base import filter_discovered_roots

    nested = [
        ("task_a/0_50", "/p/task_a/0_50"),
        ("task_a/50_100", "/p/task_a/50_100"),
        ("task_b/0_50", "/p/task_b/0_50"),
    ]
    # Drop everything under task_a via prefix.
    out = filter_discovered_roots(nested, holdout_tasks=["task_a"], match_prefix=True)
    assert [n for n, _ in out] == ["task_b/0_50"]

    # Drop a single ep_range via full-name match.
    out = filter_discovered_roots(nested, holdout_tasks=["task_a/0_50"], match_prefix=True)
    assert [n for n, _ in out] == ["task_a/50_100", "task_b/0_50"]


def test_filter_prefix_match_off_by_default():
    """With match_prefix=False, names containing '/' must be matched in full only."""
    from openwam.dataloader.lerobot_v3_base import filter_discovered_roots

    nested = [
        ("task_a/0_50", "/p/task_a/0_50"),
        ("task_b/0_50", "/p/task_b/0_50"),
    ]
    # "task_a" should NOT match "task_a/0_50" when match_prefix=False.
    out = filter_discovered_roots(nested, train_tasks=["task_a"])
    assert out == []
    out = filter_discovered_roots(nested, holdout_tasks=["task_a"])
    assert out == nested


# ---------------------------------------------------------------------------
# AgiBot / Galaxea discovery
# ---------------------------------------------------------------------------


def test_discover_agibot_two_level_with_holdout(tmp_path):
    """AgiBot's two-level layout: holdout on a top-level folder drops every ep_range under it."""
    from openwam.dataloader.agibot import discover_agibot_tasks

    base = str(tmp_path)
    # task_a/{0_50, 50_100}, task_b/0_50  — three complete v3 roots in a 2-level tree.
    for top, ep in [("task_a", "0_50"), ("task_a", "50_100"), ("task_b", "0_50")]:
        _create_fake_task(os.path.join(base, top, ep), n_episodes=1, ep_len=5)

    # No filter: all three.
    names = sorted(name for name, _ in discover_agibot_tasks(base))
    assert names == ["task_a/0_50", "task_a/50_100", "task_b/0_50"]

    # Holdout on top-level "task_a": drops every ep_range under task_a, keeps task_b.
    names = sorted(name for name, _ in discover_agibot_tasks(base, holdout_tasks=["task_a"]))
    assert names == ["task_b/0_50"]

    # Holdout on a single full name: only that ep_range drops.
    names = sorted(
        name for name, _ in discover_agibot_tasks(base, holdout_tasks=["task_a/0_50"])
    )
    assert names == ["task_a/50_100", "task_b/0_50"]


def test_discover_galaxea_single_level_with_holdout(tmp_path):
    """Galaxea's flat layout: single-level allowlist + holdout."""
    from openwam.dataloader.galaxea import discover_galaxea_tasks

    base = str(tmp_path)
    for entry in ("task_a", "task_b", "task_c"):
        _create_fake_task(os.path.join(base, entry), n_episodes=1, ep_len=5)

    names = sorted(name for name, _ in discover_galaxea_tasks(base))
    assert names == ["task_a", "task_b", "task_c"]

    names = sorted(name for name, _ in discover_galaxea_tasks(base, holdout_tasks=["task_b"]))
    assert names == ["task_a", "task_c"]

    names = sorted(
        name for name, _ in discover_galaxea_tasks(base, train_tasks=["task_a", "task_c"])
    )
    assert names == ["task_a", "task_c"]


def test_discover_agibot_skips_partial_root(tmp_path):
    """A leaf with data/ but missing meta/info.json is now silently skipped, not crashing."""
    from openwam.dataloader.agibot import discover_agibot_tasks

    base = str(tmp_path)
    # Complete root.
    _create_fake_task(os.path.join(base, "task_a", "0_50"), n_episodes=1, ep_len=5)
    # Partial sibling: has data/ but no meta/info.json — old behaviour returned it
    # then crashed in LeRobot3Dataset.__init__; new behaviour skips with INFO log.
    partial = os.path.join(base, "task_a", "50_100")
    os.makedirs(os.path.join(partial, "data"), exist_ok=True)

    names = sorted(name for name, _ in discover_agibot_tasks(base))
    assert names == ["task_a/0_50"]


# ---------------------------------------------------------------------------
# MultiTaskLeRobot3Dataset rejection paths
# ---------------------------------------------------------------------------


def test_multitask_rejects_heterogeneous_action_dim(tmp_path):
    """Mixing subsets with different action_dim must fail at __init__, not silently."""
    from openwam.dataloader.lerobot_v3_base import MultiTaskLeRobot3Dataset

    d14, d40 = _mkdir_subs(tmp_path, "d14", "d40")
    _create_fake_task(str(d14), n_episodes=1, ep_len=5, action_fields=[("action", [14])])
    _create_fake_task(str(d40), n_episodes=1, ep_len=5, action_fields=[("action", [40])])
    ds14 = _make_lerobot3(d14, num_frames=5, task_name="dim14")
    ds40 = _make_lerobot3(d40, num_frames=5, task_name="dim40")

    with pytest.raises(ValueError, match="heterogeneous action_dim"):
        MultiTaskLeRobot3Dataset([ds14, ds40])


def test_multitask_rejects_duplicate_task_name(tmp_path):
    """Subsets sharing a task_name must fail at __init__: dict-comprehension
    dedup would silently drop earlier subsets and route denormalize_action
    to the wrong stats."""
    from openwam.dataloader.lerobot_v3_base import MultiTaskLeRobot3Dataset

    d1, d2 = _mkdir_subs(tmp_path, "d1", "d2")
    _create_fake_task(str(d1), n_episodes=1, ep_len=5)
    _create_fake_task(str(d2), n_episodes=1, ep_len=5)
    ds1 = _make_lerobot3(d1, num_frames=5, task_name="duplicate_name")
    ds2 = _make_lerobot3(d2, num_frames=5, task_name="duplicate_name")

    with pytest.raises(ValueError, match="duplicate task_name"):
        MultiTaskLeRobot3Dataset([ds1, ds2])


# ---------------------------------------------------------------------------
# OXE end-to-end (registry → build_dataset → MultiTaskLeRobot3Dataset)
# ---------------------------------------------------------------------------


def test_registry_builds_oxe_dataset(tmp_path):
    """build_dataset(type=oxe, action_format=eef) → action_dim=20, sample carries mask."""
    from openwam.dataloader.registry import build_dataset

    _make_oxe_task(tmp_path)
    config = _oxe_config(subsets=[_oxe_subset("droid_1.0.1", tmp_path, _OXE_CAM)])

    ds = build_dataset(config, split="train")
    # OXEDataset is a MultiTaskLeRobot3Dataset; the dim_mask lives on the
    # one inner LeRobot3Dataset built from the single subset entry.
    assert ds.action_dim == 20
    inner = ds._datasets[0]
    assert inner.task_name == "droid_1.0.1"
    assert inner._action_dim_mask is not None
    assert inner._action_dim_mask.shape == (20,)

    with _mock_video():
        sample = ds[0]
    assert sample["action"].shape[-1] == 20
    assert "action_dim_mask" in sample
    assert sample["action_dim_mask"].shape == (20,)
    assert sample["action_dim_mask"][:10].all()
    assert not sample["action_dim_mask"][10:].any()


def test_oxe_subsets_loads_two_subsets(tmp_path):
    """Two enabled subsets in yaml → two inner LeRobot3Dataset wrapped in MultiTask."""
    from openwam.dataloader.registry import build_dataset

    droid_dir, bridge_dir = _mkdir_subs(tmp_path, "droid", "bridge")
    _make_oxe_task(droid_dir)
    _make_oxe_task(bridge_dir)

    config = _oxe_config(subsets=[
        _oxe_subset("droid_1.0.1", droid_dir, _OXE_CAM),
        _oxe_subset("bridge_v2", bridge_dir, _OXE_CAM),
    ])

    ds = build_dataset(config, split="train")
    assert len(ds._datasets) == 2
    assert {d.task_name for d in ds._datasets} == {"droid_1.0.1", "bridge_v2"}
    assert ds.action_dim == 20

    with _mock_video():
        sample = ds[0]
    assert sample["task_name"] in {"droid_1.0.1", "bridge_v2"}
    assert sample["action_dim_mask"].shape == (20,)
    assert sample["action_dim_mask"][:10].all()
    assert not sample["action_dim_mask"][10:].any()


def test_oxe_subsets_merged_with_defaults(tmp_path):
    """defaults supply shared values; per-subset overrides take precedence."""
    from openwam.dataloader.registry import build_dataset

    a_dir, b_dir = _mkdir_subs(tmp_path, "a", "b")
    _make_oxe_task(a_dir)
    _make_oxe_task(b_dir)

    # defaults.fps=15; subset 'a' inherits, subset 'b' overrides to 5.
    config = _oxe_config(
        defaults=dict(fps=15, num_frames=7),
        subsets=[
            _oxe_subset("a", a_dir, _OXE_CAM),
            _oxe_subset("b", b_dir, _OXE_CAM, fps=5),
        ],
    )

    ds = build_dataset(config, split="train")
    by_name = {d.task_name: d for d in ds._datasets}
    assert by_name["a"].fps == 15
    assert by_name["b"].fps == 5
    # num_frames flowed through defaults to both subsets (LeRobot3Dataset stores it).
    assert by_name["a"].num_frames == 7
    assert by_name["b"].num_frames == 7


def test_oxe_disabled_subset_skipped(tmp_path):
    """enabled=false subsets are silently skipped; only enabled ones are built."""
    from openwam.dataloader.registry import build_dataset

    a_dir = tmp_path / "a"
    _make_oxe_task(a_dir)
    # bridge_dir does NOT exist on disk; enabled=false should mean we never look
    # at it (otherwise the dataset_dir validation would raise).
    bridge_dir = tmp_path / "bridge_pending_no_meta"

    config = _oxe_config(subsets=[
        _oxe_subset("a", a_dir, _OXE_CAM),
        _oxe_subset("bridge", bridge_dir, _OXE_CAM, enabled=False),
    ])

    ds = build_dataset(config, split="train")
    assert len(ds._datasets) == 1
    assert ds._datasets[0].task_name == "a"


def test_oxe_all_disabled_raises():
    from openwam.dataloader.registry import build_dataset

    config = _oxe_config(subsets=[
        _oxe_subset("a", "/nonexistent", "cam", enabled=False),
        _oxe_subset("b", "/nonexistent", "cam", enabled=False),
    ])
    with pytest.raises(RuntimeError, match="enabled=false"):
        build_dataset(config, split="train")


def test_oxe_empty_subsets_raises():
    from openwam.dataloader.registry import build_dataset

    config = _oxe_config(subsets=[])
    with pytest.raises(ValueError, match="non-empty 'subsets' list"):
        build_dataset(config, split="train")


def test_oxe_duplicate_subset_names_raises():
    from openwam.dataloader.registry import build_dataset

    config = _oxe_config(subsets=[
        _oxe_subset("dup", "/x", "cam"),
        _oxe_subset("dup", "/y", "cam"),
    ])
    with pytest.raises(ValueError, match="duplicate subset names"):
        build_dataset(config, split="train")


def test_oxe_dataset_dir_not_v3_root_raises(tmp_path):
    """dataset_dir without meta/info.json gets a clear error pointing at enabled=false."""
    from openwam.dataloader.registry import build_dataset

    # Directory exists but is empty — no meta/info.json.
    (empty_dir,) = _mkdir_subs(tmp_path, "incomplete")
    config = _oxe_config(subsets=[_oxe_subset("incomplete", empty_dir, _OXE_CAM)])
    with pytest.raises(FileNotFoundError) as excinfo:
        build_dataset(config, split="train")
    msg = str(excinfo.value)
    assert "LeRobot v3 root" in msg
    assert "enabled" in msg  # hint about setting enabled: false


def test_oxe_action_format_full_skips_transform(tmp_path):
    """When action_format is null/'full', no transform is applied and no mask attached."""
    from openwam.dataloader.registry import build_dataset

    _make_oxe_task(tmp_path)
    config = _oxe_config(
        defaults=dict(action_format=None),  # raw 7-D, no transform
        subsets=[_oxe_subset("droid", tmp_path, _OXE_CAM)],
    )

    ds = build_dataset(config, split="train")
    assert ds.action_dim == 7
    inner = ds._datasets[0]
    assert inner._action_dim_mask is None

    with _mock_video():
        sample = ds[0]
    assert sample["action"].shape[-1] == 7
    # No transform -> 7-D real action with no padding; fallback mask is all-True
    # at the dataset's actual action_dim.
    assert "action_dim_mask" in sample
    assert sample["action_dim_mask"].shape == (7,)
    assert sample["action_dim_mask"].all()
