"""Tests for the InternData-A1 v3.0 dataloader (openwam/dataloader/interndata_a1.py).

Covers the A1-specific behavior the shared LeRobotV3Reader base does not:
arm-layout auto-detection (bimanual vs unprefixed franka), recursive
variable-depth bucket discovery, the wxyz->xyzw quaternion reorder, 20-D
xyz+rot6d+gripper assembly with single-arm left-half placement, the
episode-boundary action drop, and per-embodiment stats loading with rot6d
identity pinning.

Video decode is monkeypatched throughout, so no mp4 is needed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from openwam.dataloader.interndata_a1 import (
    ROBOT_TYPE_TO_EMBODIMENT,
    InternDataA1Dataset,
    MultiInternDataA1Dataset,
    detect_arm_layout,
    discover_a1_buckets,
    embodiment_key,
    iter_data_shards,
    load_excluded_episodes,
    parse_shard_path,
    resolve_trim_bounds,
)
from openwam.dataloader.utils.eef import (
    ARM10_DIM,
    EEF_DIM,
    LEFT_ARM_DIM_MASK,
    quat_wxyz_to_rot6d,
    quat_xyzw_to_rot6d,
)
from openwam.dataloader.utils.lerobotv3 import DataContractError
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20

HEAD = "images.rgb.head"
HAND = "images.rgb.hand"
HAND_L = "images.rgb.hand_left"
HAND_R = "images.rgb.hand_right"

_POSE7 = {"dtype": "float32", "shape": [7]}
_SCALAR = {"dtype": "float32", "shape": [1]}
_VIDEO = {"dtype": "video", "shape": [360, 640, 3]}

BIMANUAL_FEATURES = {
    HEAD: _VIDEO,
    HAND_L: _VIDEO,
    HAND_R: _VIDEO,
    "states.left_ee_to_robot_pose": _POSE7,
    "states.left_gripper.position": _SCALAR,
    "states.right_ee_to_robot_pose": _POSE7,
    "states.right_gripper.position": _SCALAR,
    "actions.left_ee_to_robot_pose": _POSE7,
    "actions.left_gripper.position": _SCALAR,
    "actions.right_ee_to_robot_pose": _POSE7,
    "actions.right_gripper.position": _SCALAR,
}
SINGLE_ARM_FEATURES = {
    HEAD: _VIDEO,
    HAND: _VIDEO,
    "states.ee_to_robot_pose": _POSE7,
    "states.gripper.position": _SCALAR,
    "actions.ee_to_robot_pose": _POSE7,
    "actions.gripper.position": _SCALAR,
}


def _unit_quats(n: int, seed: int) -> np.ndarray:
    """(n, 4) random unit quaternions in wxyz order."""
    rng = np.random.RandomState(seed)
    q = rng.randn(n, 4).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _make_bucket(
    root: Path,
    rel: str,
    *,
    layout: str = "bimanual",
    robot_type: str = "AgileX Split Aloha",
    n_eps: int = 2,
    ep_len: int = 40,
    seed: int = 0,
    fixed_quat: np.ndarray | None = None,
) -> Path:
    """Write a synthetic LeRobot v3 A1 bucket at ``root/rel``.

    Actions are written as the exact next state (``actions[t] == states[t+1]``,
    last row clamped) to mirror the real dataset's relabeling.

    ``fixed_quat`` writes one known (4,) **wxyz** quaternion into every row of
    every arm instead of random ones, so a test can assert the exact rot6d the
    reader must emit for it.
    """
    d = root / rel
    (d / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (d / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    bimanual = layout == "bimanual"
    features = BIMANUAL_FEATURES if bimanual else SINGLE_ARM_FEATURES
    cams = [HEAD, HAND_L, HAND_R] if bimanual else [HEAD, HAND]
    sides = ["left", "right"] if bimanual else [None]

    (d / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": robot_type,
                "fps": 30.0,
                "total_episodes": n_eps,
                "splits": {"train": f"0:{n_eps}"},
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": features,
            }
        )
    )
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame({"task_index": [0]}, index=["Close the microwave"])),
        d / "meta" / "tasks.parquet",
    )

    total = n_eps * ep_len
    rng = np.random.RandomState(seed)
    cols: dict = {
        "task_index": np.zeros(total, dtype=np.int64),
        # Real A1 shards carry this; the cleaned-view filters key on it.
        "episode_index": np.repeat(np.arange(n_eps, dtype=np.int64), ep_len),
    }
    for i, side in enumerate(sides):
        pfx = f"{side}_" if side else ""
        pos = rng.randn(total, 3).astype(np.float32)
        if fixed_quat is None:
            quat = _unit_quats(total, seed + i)
        else:
            quat = np.tile(np.asarray(fixed_quat, dtype=np.float32), (total, 1))
        state = np.concatenate([pos, quat], axis=-1)
        grip = rng.rand(total, 1).astype(np.float32)
        # actions[t] = states[t+1] within each episode; final row clamped.
        act, act_grip = state.copy(), grip.copy()
        for e in range(n_eps):
            lo, hi = e * ep_len, (e + 1) * ep_len
            act[lo : hi - 1] = state[lo + 1 : hi]
            act[hi - 1] = state[hi - 1]
            act_grip[lo : hi - 1] = grip[lo + 1 : hi]
            act_grip[hi - 1] = grip[hi - 1]
        cols[f"states.{pfx}ee_to_robot_pose"] = list(state)
        cols[f"states.{pfx}gripper.position"] = list(grip)
        cols[f"actions.{pfx}ee_to_robot_pose"] = list(act)
        cols[f"actions.{pfx}gripper.position"] = list(act_grip)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(cols)), d / "data" / "chunk-000" / "file-000.parquet")

    rows = []
    for e in range(n_eps):
        row = {
            "episode_index": e,
            "length": ep_len,
            "tasks": ["Close the microwave"],
            "dataset_from_index": e * ep_len,
            "dataset_to_index": (e + 1) * ep_len,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for c in cams:
            row[f"videos/{c}/chunk_index"] = 0
            row[f"videos/{c}/file_index"] = 0
            # Real manifests carry these; the reader derives each camera's frame
            # offset from from_timestamp rather than a cumsum over surviving rows.
            row[f"videos/{c}/from_timestamp"] = (e * ep_len) / 30.0
            row[f"videos/{c}/to_timestamp"] = ((e + 1) * ep_len) / 30.0
        rows.append(row)
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(rows)),
        d / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    return d


@pytest.fixture
def patch_decode(monkeypatch):
    def fake_decode(path, frame_indices, height, width):
        return [Image.new("RGB", (width, height)) for _ in frame_indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", fake_decode)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestLayoutDetection:
    def test_bimanual(self):
        assert detect_arm_layout(BIMANUAL_FEATURES) == "bimanual"

    def test_single_arm_unprefixed(self):
        assert detect_arm_layout(SINGLE_ARM_FEATURES) == "single_arm"

    def test_bimanual_wins_when_both_shapes_present(self):
        """A bucket exposing both must not be read as single-arm (that would
        silently drop the right arm)."""
        assert detect_arm_layout({**BIMANUAL_FEATURES, "states.ee_to_robot_pose": _POSE7}) == "bimanual"

    def test_unknown_schema_raises(self):
        with pytest.raises(ValueError, match="neither the bimanual"):
            detect_arm_layout({"states.joint.position": {"dtype": "float32", "shape": [7]}})


class TestEmbodimentKey:
    @pytest.mark.parametrize("robot_type,expected", sorted(ROBOT_TYPE_TO_EMBODIMENT.items()))
    def test_known_types(self, robot_type, expected):
        assert embodiment_key(robot_type, "bimanual") == expected

    def test_unknown_type_slugs_rather_than_borrowing(self, caplog):
        assert embodiment_key("Some New Bot v2", "bimanual") == "some_new_bot_v2"
        assert "unrecognized robot_type" in caplog.text

    def test_franka_maps_to_the_franka_stats_key(self):
        """Named for what it checks: the robot_type -> stats-file-suffix mapping.
        Franka's single-arm-ness is NOT a property of this table — it is detected
        from info.features, and asserted in TestSingleArmFranka."""
        assert ROBOT_TYPE_TO_EMBODIMENT["Franka"] == "franka"


class TestQuaternionConvention:
    """The dataset stores quaternion.w FIRST; feeding wxyz into the xyzw helper
    produces a wrong-but-unit rotation that no norm check can catch."""

    def test_matches_explicit_rotation_matrix_columns(self):
        # 90 deg about z: wxyz = [cos45, 0, 0, sin45]
        s = np.sqrt(0.5)
        rot6d = quat_wxyz_to_rot6d(np.array([[s, 0.0, 0.0, s]], dtype=np.float32))[0]
        # R = [[0,-1,0],[1,0,0],[0,0,1]] -> col0 = (0,1,0), col1 = (-1,0,0)
        np.testing.assert_allclose(rot6d[:3], [0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(rot6d[3:], [-1, 0, 0], atol=1e-6)

    def test_identity_quaternion(self):
        rot6d = quat_wxyz_to_rot6d(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))[0]
        np.testing.assert_allclose(rot6d, [1, 0, 0, 0, 1, 0], atol=1e-6)

    def test_reorder_is_load_bearing(self):
        """Regression guard: the two conventions must NOT agree, and the wrong
        one must still look orthonormal — that is exactly why it needs a test."""
        q = _unit_quats(16, seed=3)
        right, wrong = quat_wxyz_to_rot6d(q), quat_xyzw_to_rot6d(q)
        assert not np.allclose(right, wrong, atol=1e-3)
        np.testing.assert_allclose(np.linalg.norm(wrong[:, :3], axis=1), 1.0, atol=1e-5)

    def test_output_columns_are_orthonormal(self):
        r = quat_wxyz_to_rot6d(_unit_quats(32, seed=7))
        np.testing.assert_allclose(np.linalg.norm(r[:, :3], axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(r[:, 3:], axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose((r[:, :3] * r[:, 3:]).sum(axis=1), 0.0, atol=1e-5)

    def test_rejects_wrong_width(self):
        with pytest.raises(ValueError, match="4-D wxyz"):
            quat_wxyz_to_rot6d(np.zeros((4, 3), dtype=np.float32))


class TestBucketDiscovery:
    def test_finds_both_nesting_depths(self, tmp_path):
        _make_bucket(tmp_path, "articulation_tasks/split_aloha/close_microwave")
        _make_bucket(tmp_path, "pick_and_place_tasks/franka/single_pick/google_scan-book", layout="single_arm")
        found = {p.relative_to(tmp_path).as_posix() for p in discover_a1_buckets(tmp_path)}
        assert found == {
            "articulation_tasks/split_aloha/close_microwave",
            "pick_and_place_tasks/franka/single_pick/google_scan-book",
        }

    def test_does_not_descend_into_a_bucket(self, tmp_path):
        """A bucket's own data/ and videos/ must never be reported as buckets —
        and the walk must not pay to traverse them."""
        b = _make_bucket(tmp_path, "cat/emb/task")
        (b / "data" / "chunk-000" / "meta").mkdir(parents=True, exist_ok=True)
        (b / "data" / "chunk-000" / "meta" / "info.json").write_text("{}")
        assert discover_a1_buckets(tmp_path) == [b]

    def test_empty_root(self, tmp_path):
        assert discover_a1_buckets(tmp_path) == []

    def test_skips_killed_tar_staging_dirs(self, tmp_path):
        """Archive extraction stages into <cat>/<emb>/.partial_<name>/.
        SIGKILL/OOM/preemption bypasses the extractor cleanup, so a half-extracted tree
        that already has meta/ must not be mistaken for a complete bucket — it
        would construct fine and then die at __getitem__ mid-training."""
        good = _make_bucket(tmp_path, "cat/emb/good")
        _make_bucket(tmp_path, "cat/emb/.partial_halfdone/halfdone")
        (tmp_path / ".extract_logs" / "sentinels").mkdir(parents=True)
        assert discover_a1_buckets(tmp_path) == [good]

    def test_follows_symlinked_buckets(self, tmp_path):
        """Symlinking a subset instead of copying is the realistic way to carve a
        slice out of a 2.1 TiB tree, and the base reader's root mode follows
        symlinks — os.walk's followlinks=False default would report an empty
        tree and raise the misleading 'did the archives get extracted?' error."""
        elsewhere = _make_bucket(tmp_path / "store", "real_task")
        farm = tmp_path / "farm" / "cat" / "emb"
        farm.mkdir(parents=True)
        (farm / "linked").symlink_to(elsewhere)
        found = discover_a1_buckets(tmp_path / "farm")
        assert [p.relative_to(tmp_path / "farm").as_posix() for p in found] == ["cat/emb/linked"]

    def test_multiply_reachable_bucket_keeps_a_readdir_independent_alias(self, tmp_path, monkeypatch):
        """When a bucket is reachable by two paths, WHICH alias survives must not
        depend on raw readdir order — that is a filesystem-instance property (ext4
        htree hashing is seeded per mkfs), so the same tree on two machines would
        otherwise keep different aliases. That shifts dataset_id, every later
        bucket's index, the per-bucket subsample seeds in build_multibucket, and
        the stats merge order (hence q01/q99).

        The host's own readdir order is not trusted here: this reverses scandir,
        so the assertion only holds if the walk sorts. Without the sort the
        reversed order makes the 'zfarm/alias' path win instead.
        """
        real = _make_bucket(tmp_path, "astore/task1")
        (tmp_path / "zfarm").mkdir(parents=True)
        (tmp_path / "zfarm" / "alias").symlink_to(real)

        _real_scandir = os.scandir

        class _ReverseScandir:
            def __init__(self, path="."):
                with _real_scandir(path) as it:
                    self._it = iter(sorted(it, key=lambda e: e.name, reverse=True))

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._it)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def close(self):
                pass

        monkeypatch.setattr(os, "scandir", _ReverseScandir)
        found = discover_a1_buckets(tmp_path)
        assert [p.relative_to(tmp_path).as_posix() for p in found] == ["astore/task1"]

    def test_symlink_cycle_terminates(self, tmp_path):
        """followlinks=True re-walks a cycle forever without the inode guard."""
        good = _make_bucket(tmp_path, "cat/emb/good")
        loop = tmp_path / "cat" / "loop"
        loop.mkdir(parents=True, exist_ok=True)
        (loop / "back").symlink_to(tmp_path)
        assert discover_a1_buckets(tmp_path) == [good]


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class TestBimanualReader:
    def test_all_20_dims_supervised(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds.arm_layout == "bimanual"
        assert ds.embodiment == "split_aloha"
        assert ds.ACTION_DIM_MASK is None
        s = ds[0]
        assert s["action"].shape == (8, EEF_DIM)
        assert s["proprio"].shape == (1, EEF_DIM)
        assert bool(s["action_mask"][0].all())
        assert bool(s["proprio_mask"].all())
        assert s["prompt"] == "Close the microwave"

    def test_resolves_three_cameras(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert (ds._head_camera, ds._left_wrist_camera, ds._right_wrist_camera) == (HEAD, HAND_L, HAND_R)

    def test_rot6d_slots_are_orthonormal(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        a = ds[0]["action"].numpy()
        for lo in (3, 13):
            c0, c1 = a[:, lo : lo + 3], a[:, lo + 3 : lo + 6]
            np.testing.assert_allclose(np.linalg.norm(c0, axis=1), 1.0, atol=1e-5)
            np.testing.assert_allclose((c0 * c1).sum(axis=1), 0.0, atol=1e-5)


class TestReaderQuaternionConvention:
    """Pin the wxyz convention through the REAL ``__getitem__`` path.

    ``TestQuaternionConvention`` pins the helper, but nothing there stops the
    reader from calling the *other* helper: swapping ``_arm10``'s
    ``quat_wxyz_to_rot6d`` for ``quat_xyzw_to_rot6d`` leaves every other test in
    this file green (orthonormality holds for the wrong rotation, and the
    row-alignment test compares two outputs of the same ``_eef20``, so it is
    self-consistent under the flip). Every rotation in every training batch would
    be silently wrong. So these assert exact, hand-computed rot6d values.

    Planted quaternion: wxyz ``[s, 0, 0, s]``, s = sqrt(1/2) — 90 deg about z.
        R = [[0,-1,0],[1,0,0],[0,0,1]]  ->  rot6d = col0 ++ col1 = [0,1,0, -1,0,0]
    Read as xyzw the SAME four numbers are 90 deg about x, giving [1,0,0, 0,0,1]:
    a different, equally unit-norm, equally orthonormal answer.
    """

    S = float(np.sqrt(0.5))
    WXYZ = np.array([S, 0.0, 0.0, S], dtype=np.float32)
    EXPECTED = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
    # What the xyzw helper would produce from the same bytes (must NOT appear).
    WRONG = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def test_expected_and_wrong_are_what_the_two_helpers_give(self):
        """Guard the constants above against a helper change, so a failure below
        is unambiguously the reader wiring and not a stale expectation here."""
        q = self.WXYZ[None, :]
        np.testing.assert_allclose(quat_wxyz_to_rot6d(q)[0], self.EXPECTED, atol=1e-6)
        np.testing.assert_allclose(quat_xyzw_to_rot6d(q)[0], self.WRONG, atol=1e-6)

    def test_bimanual_action_and_proprio_rot6d_are_the_wxyz_answer(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", fixed_quat=self.WXYZ)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        s = ds[0]
        for arr in (s["action"].numpy(), s["proprio"].numpy()):
            for lo in (3, 13):  # left and right rot6d slots
                block = arr[:, lo : lo + 6]
                np.testing.assert_allclose(block, np.tile(self.EXPECTED, (len(arr), 1)), atol=1e-6)
                assert not np.allclose(block, self.WRONG, atol=1e-3)

    def test_single_arm_action_rot6d_is_the_wxyz_answer(self, tmp_path, patch_decode):
        """The franka path builds its left arm through the same ``_arm10``, but
        via a different column set — cover it so neither branch can drift."""
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka", fixed_quat=self.WXYZ)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        a = ds[0]["action"].numpy()
        np.testing.assert_allclose(a[:, 3:9], np.tile(self.EXPECTED, (len(a), 1)), atol=1e-6)
        assert not np.allclose(a[:, 3:9], self.WRONG, atol=1e-3)


class TestSingleArmFranka:
    """Franka fills the LEFT half; the right half is zero padding, masked out."""

    def test_left_half_placement_and_mask(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds.arm_layout == "single_arm"
        assert ds.embodiment == "franka"
        np.testing.assert_array_equal(ds.ACTION_DIM_MASK, LEFT_ARM_DIM_MASK)

        s = ds[0]
        a, p = s["action"].numpy(), s["proprio"].numpy()
        # right half is exactly zero, left half carries real data
        np.testing.assert_array_equal(a[:, ARM10_DIM:], 0.0)
        np.testing.assert_array_equal(p[:, ARM10_DIM:], 0.0)
        assert np.abs(a[:, :ARM10_DIM]).sum() > 0
        # mask excludes the padding
        assert int(s["action_mask"][0].sum()) == ARM10_DIM
        assert int(s["proprio_mask"].sum()) == ARM10_DIM
        assert not bool(s["action_mask"][:, ARM10_DIM:].any())

    def test_single_wrist_camera_takes_the_left_slot(self, tmp_path, patch_decode):
        """The wrist view must sit on the same side as the arm's action slots."""
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert (ds._head_camera, ds._left_wrist_camera, ds._right_wrist_camera) == (HEAD, HAND, None)


class TestTemporalAlignment:
    def test_action_is_row_aligned_next_state(self, tmp_path, patch_decode):
        """actions[t] == states[t+1] in the source; the reader reads actions.*
        row-aligned, so no extra shift may be introduced."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task", ep_len=40)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        win = ds._load_data_table(0, 0).slice(0, 10).to_pandas()
        action = ds._eef20(win, "action", 9)
        state = ds._eef20(win, "state", 10)
        np.testing.assert_allclose(action[:-1], state[1:9], atol=1e-6)

    def test_full_window_keeps_every_step(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._n_supervised_action_steps(9) == 9

    def test_episode_truncated_window_drops_the_clamped_last_action(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._n_supervised_action_steps(4) == 3
        assert ds._n_supervised_action_steps(1) == 0

    def test_last_window_of_episode_masks_the_fabricated_target(self, tmp_path, patch_decode):
        """ep_len=12, num_frames=9 -> starts at offsets 0..10 (T_action=8).

        offset 3 fits exactly: it spans rows 3..11, but T_action=8 already stops
        at row 10, so the clamped row 11 is never used as a target and all 8
        steps stay supervised. offset 10 is truncated to 2 rows (10, 11), and row
        11 IS the clamped duplicate -> only 1 supervised step survives.
        """
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=1, ep_len=12)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert len(ds) == 11
        assert int(ds[3]["action_mask"].any(dim=1).sum()) == 8
        assert int(ds[len(ds) - 1]["action_mask"].any(dim=1).sum()) == 1

    def test_min_window_len_keeps_every_train_window_supervised(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=1, ep_len=6)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._train_min_window_len() == 2
        for i in range(len(ds)):
            assert int(ds[i]["action_mask"].any(dim=1).sum()) >= 1


class TestGripperHarmonization:
    """gripper.position is published on two different scales; the reader must
    map every bucket onto a normalized [0, 1] aperture before assembly."""

    def _write_bucket_stats(self, bucket: Path, cols: dict):
        (bucket / "meta" / "stats.json").write_text(
            json.dumps({c: {"min": [0.0], "max": [m]} for c, m in cols.items()})
        )

    def test_metric_bucket_is_divided_by_the_stroke(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
        self._write_bucket_stats(d, {"states.gripper.position": 0.08})
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._grip_scale[0] == pytest.approx(0.08)
        raw = np.stack(ds._load_data_table(0, 0).slice(0, 9).to_pandas()["actions.gripper.position"].values)
        np.testing.assert_allclose(ds[0]["action"].numpy()[:, 9], raw.ravel()[:8] / 0.08, atol=1e-5)

    def test_binary_openness_bucket_is_left_alone(self, tmp_path, patch_decode):
        """A normalized Franka bucket stores 0/1; applying the metric stroke
        again would incorrectly amplify its values."""
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
        self._write_bucket_stats(d, {"states.gripper.position": 1.0})
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._grip_scale[0] == 1.0
        assert float(np.abs(ds[0]["action"].numpy()[:, 9]).max()) <= 1.0

    def test_two_scales_land_on_the_same_aperture(self, tmp_path, patch_decode):
        """A half-open gripper must read ~0.5 whichever scale its bucket used."""
        metric = _make_bucket(tmp_path, "cat/franka/metric", layout="single_arm", robot_type="Franka", seed=1)
        norm = _make_bucket(tmp_path, "cat/franka/norm", layout="single_arm", robot_type="Franka", seed=1)
        self._write_bucket_stats(metric, {"states.gripper.position": 0.08})
        self._write_bucket_stats(norm, {"states.gripper.position": 1.0})
        # rewrite the metric bucket's gripper as the normalized one * 0.08
        for bucket, factor in ((metric, 0.08), (norm, 1.0)):
            p = bucket / "data" / "chunk-000" / "file-000.parquet"
            t = pq.read_table(p).to_pandas()
            for c in ("states.gripper.position", "actions.gripper.position"):
                t[c] = [np.array([0.5 * factor], dtype=np.float32)] * len(t)
            pq.write_table(pa.Table.from_pandas(t), p)
        a = InternDataA1Dataset(str(metric), normalize_mode=None, num_frames=9, video_stride=4)[0]
        b = InternDataA1Dataset(str(norm), normalize_mode=None, num_frames=9, video_stride=4)[0]
        np.testing.assert_allclose(a["action"].numpy()[:, 9], 0.5, atol=1e-5)
        np.testing.assert_allclose(b["action"].numpy()[:, 9], 0.5, atol=1e-5)

    def test_missing_bucket_stats_falls_back_to_the_embodiment_stroke(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._grip_scale == (pytest.approx(0.1), pytest.approx(0.1))

    def test_single_gripper_embodiments_always_use_their_one_stroke(self, tmp_path, patch_decode):
        """lift2 ships ONE gripper, so no per-bucket variant detection may fire —
        even for a side whose observed max coincidentally looks like 1.0."""
        d = _make_bucket(tmp_path, "cat/lift2/task", robot_type="ARX Lift-2")
        self._write_bucket_stats(d, {"states.left_gripper.position": 0.088, "states.right_gripper.position": 1.0})
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._grip_scale == (pytest.approx(0.088), pytest.approx(0.088))

    def test_genie1_uses_the_documented_574_stroke(self, tmp_path, patch_decode):
        """genie1's full-open is 5.74, NOT 1.0 — most episodes never fully open,
        so a naive 'observed max ~ 1' read of this embodiment is wrong."""
        d = _make_bucket(tmp_path, "cat/genie1/task", robot_type="Genie-1")
        self._write_bucket_stats(d, {"states.left_gripper.position": 1.16, "states.right_gripper.position": 5.74})
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._grip_scale == (pytest.approx(5.74), pytest.approx(5.74))

    def test_half_open_franka_still_resolves_to_the_panda_stroke(self, tmp_path, patch_decode):
        """Variant matching is in LOG space: 0.04 is 2x from 0.08 but 25x from
        1.0, so a panda bucket that only ever half-opens must not flip to Robotiq."""
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
        self._write_bucket_stats(d, {"states.gripper.position": 0.04})
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._grip_scale[0] == pytest.approx(0.08)

    def test_out_of_range_outlier_warns(self, tmp_path, patch_decode, caplog):
        """An extreme synthetic outlier is surfaced as a warning, not made fatal."""
        d = _make_bucket(tmp_path, "cat/genie1/task", robot_type="Genie-1")
        self._write_bucket_stats(d, {"states.left_gripper.position": 100.0, "states.right_gripper.position": 1.0})
        with caplog.at_level("WARNING"):
            ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._grip_scale == (pytest.approx(5.74), pytest.approx(5.74))
        assert "the assumed full-open stroke" in caplog.text

    def test_in_range_bucket_does_not_warn(self, tmp_path, patch_decode, caplog):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        self._write_bucket_stats(d, {"states.left_gripper.position": 0.1, "states.right_gripper.position": 0.1})
        with caplog.at_level("WARNING"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert "full-open stroke" not in caplog.text

    def test_alt_stroke_pick_is_never_silent(self, tmp_path, patch_decode, caplog):
        """A corroborated Robotiq bucket still logs — the pick rescales the whole
        bucket's gripper dim by 12.5x off one order statistic.

        The level is asserted, not just the text: `caplog.at_level("INFO")`
        captures WARNING too, so a text-only assert would stay green if this
        branch were collapsed into `logger.warning`, which would create warning
        fatigue for every legitimate alternate-stroke bucket.
        """
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
        (d / "meta" / "stats.json").write_text(
            json.dumps({"states.gripper.position": {"min": [0.0], "max": [1.0], "mean": [0.45]}})
        )
        with caplog.at_level("INFO"):
            ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._grip_scale[0] == pytest.approx(1.0)
        picks = [r for r in caplog.records if "alt (second-variant) stroke" in r.getMessage()]
        assert len(picks) == 1
        assert picks[0].levelno == logging.INFO
        # A corroborated pick must NOT also trip the uncorroborated warning.
        assert "does not clear the primary stroke" not in caplog.text

    def test_glitch_max_flipping_a_panda_bucket_warns(self, tmp_path, patch_decode, caplog):
        """A synthetic in-range maximum above the log-space decision boundary
        can misclassify a metric bucket as the alternate variant while evading
        the out-of-range guard. The mean must still escalate that decision."""
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
        (d / "meta" / "stats.json").write_text(
            json.dumps({"states.gripper.position": {"min": [0.0], "max": [0.4], "mean": [0.04]}})
        )
        with caplog.at_level("WARNING"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert "does not clear the primary stroke" in caplog.text

    def test_missing_mean_does_not_silently_reassure(self, tmp_path, patch_decode, caplog):
        """A stats.json without `mean` cannot corroborate, so the alt pick must
        warn rather than pass unremarked."""
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
        self._write_bucket_stats(d, {"states.gripper.position": 1.0})  # min/max only
        with caplog.at_level("WARNING"):
            ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._grip_scale[0] == pytest.approx(1.0)
        assert "does not clear the primary stroke" in caplog.text

    def test_single_gripper_embodiment_never_logs_a_variant_pick(self, tmp_path, patch_decode, caplog):
        """No alt stroke declared -> no detection, so no pick to report."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        self._write_bucket_stats(d, {"states.left_gripper.position": 0.1, "states.right_gripper.position": 0.1})
        with caplog.at_level("INFO"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert "second-variant" not in caplog.text

    def test_never_opened_gripper_falls_back_to_the_stroke_and_stays_zero(self, tmp_path, patch_decode):
        """max ~ 0 carries no scale information, but 0 / anything == 0."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        self._write_bucket_stats(d, {"states.left_gripper.position": 0.0, "states.right_gripper.position": 0.0})
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._grip_scale[0] == pytest.approx(0.1)


class TestStats:
    def _write_stats(
        self,
        root: Path,
        embodiment: str,
        *,
        pin_rot6d: bool = True,
        buckets=("cat/split_aloha/task",),
        split="train",
        trim_active=False,
        min_keep=2,
    ):
        (root / "meta").mkdir(parents=True, exist_ok=True)
        eef = {
            "mean": [0.0] * EEF_DIM,
            "std": [1.0] * EEF_DIM,
            "min": [-2.0] * EEF_DIM,
            "max": [2.0] * EEF_DIM,
            "q01": [-2.0] * EEF_DIM,
            "q99": [2.0] * EEF_DIM,
        }
        if pin_rot6d:
            for dim in ROT6D_DIMS_EEF20:
                eef["q01"][dim] = -1.0
                eef["q99"][dim] = 1.0
        (root / "meta" / f"stats_{embodiment}.json").write_text(
            json.dumps(
                {
                    "eef": eef,
                    "population": {
                        "split": split,
                        "trim_active": trim_active,
                        "min_keep": min_keep,
                        "buckets": list(buckets),
                    },
                }
            )
        )

    def test_missing_stats_file_raises_actionable_error(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        with pytest.raises(FileNotFoundError, match="interndata_a1_stats_computation"):
            InternDataA1Dataset(str(d), a1_stats_root=str(tmp_path), normalize_mode="quantile")

    def test_missing_stats_error_prescribes_the_stats_root_it_looked_in(self, tmp_path, patch_decode):
        """The suggested command must carry --stats_root, pointing at the SAME
        directory the failed lookup used. Otherwise the one scenario stats_root
        exists for — a read-only dataset mount — hands the user a command that
        writes where this lookup does not read, reproducing the same error."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        elsewhere = tmp_path / "scratch"
        with pytest.raises(FileNotFoundError) as e:
            InternDataA1Dataset(str(d), a1_stats_root=str(elsewhere), normalize_mode="quantile")
        msg = str(e.value)
        assert f"--stats_root {elsewhere}" in msg
        # And it names the path it actually looked for, so the two agree.
        assert str(elsewhere / "meta" / "stats_split_aloha.json") in msg

    def test_normalize_mode_null_needs_no_stats(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        assert ds._normalization_stats is None

    def test_shared_stats_root_is_used_not_the_bucket_dir(self, tmp_path, patch_decode):
        """Buckets sit at variable depth; stats live once at the dataset root."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        self._write_stats(tmp_path, "split_aloha")
        ds = InternDataA1Dataset(
            str(d), a1_stats_root=str(tmp_path), normalize_mode="quantile", num_frames=9, video_stride=4
        )
        assert ds._normalization_stats is not None

    def test_rot6d_dims_pass_through_normalization(self, tmp_path, patch_decode):
        """Pinned rot6d stats must leave the rotation representation untouched,
        while pos/gripper dims are rescaled."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        self._write_stats(tmp_path, "split_aloha")
        raw = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=9, video_stride=4)
        norm = InternDataA1Dataset(
            str(d), a1_stats_root=str(tmp_path), normalize_mode="quantile", num_frames=9, video_stride=4
        )
        a_raw = raw[0]["action"].numpy()
        a_norm = norm[0]["action"].numpy()
        np.testing.assert_allclose(a_norm[:, list(ROT6D_DIMS_EEF20)], a_raw[:, list(ROT6D_DIMS_EEF20)], atol=1e-6)
        # xyz dims used q01/q99 = +-2 -> genuinely rescaled
        assert not np.allclose(a_norm[:, 0:3], a_raw[:, 0:3], atol=1e-6)

    def test_wrong_width_stats_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
        (tmp_path / "meta" / "stats_split_aloha.json").write_text(
            json.dumps(
                {
                    "eef": {k: [0.0] * 10 for k in ("mean", "std", "min", "max", "q01", "q99")},
                    "population": {
                        "split": "train",
                        "trim_active": False,
                        "min_keep": 2,
                        "buckets": ["cat/split_aloha/task"],
                    },
                }
            )
        )
        with pytest.raises(ValueError, match="!= expected 20"):
            InternDataA1Dataset(str(d), a1_stats_root=str(tmp_path), normalize_mode="quantile")


class TestUnifyScatter:
    def test_single_arm_padding_stays_masked_after_scatter(self, tmp_path, patch_decode):
        """The 80-D scatter must not resurrect franka's zero-padded right arm."""
        d = _make_bucket(tmp_path, "cat/franka/task", layout="single_arm", robot_type="Franka")
        ds = InternDataA1Dataset(
            str(d),
            normalize_mode=None,
            num_frames=9,
            video_stride=4,
            unify_action=True,
            unify_action_map=["0-9", "34-43"],
        )
        s = ds[0]
        assert ds.action_dim == 80
        assert s["action"].shape == (8, 80)
        # left arm -> slots 0-9 valid; right-arm destinations 34-43 masked out
        assert int(s["action_mask"][0].sum()) == ARM10_DIM
        assert bool(s["action_mask"][0, :ARM10_DIM].all())
        assert not bool(s["action_mask"][0, 34:44].any())

    def test_bimanual_maps_both_arms(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        ds = InternDataA1Dataset(
            str(d),
            normalize_mode=None,
            num_frames=9,
            video_stride=4,
            unify_action=True,
            unify_action_map=["0-9", "34-43"],
        )
        m = ds[0]["action_mask"][0]
        assert int(m.sum()) == EEF_DIM
        assert bool(m[:10].all()) and bool(m[34:44].all())


class TestFromConfig:
    def test_root_mode_discovers_and_aggregates_mixed_embodiments(self, tmp_path, patch_decode):
        _make_bucket(tmp_path, "cat/split_aloha/taskA")
        _make_bucket(tmp_path, "cat/franka/taskB/obj", layout="single_arm", robot_type="Franka")
        _make_bucket(tmp_path, "cat/lift2/taskC", robot_type="ARX Lift-2")
        ds = InternDataA1Dataset.from_config(
            {"dataset_dir": str(tmp_path), "normalize_mode": None, "num_frames": 9, "video_stride": 4},
            split="train",
        )
        assert isinstance(ds, MultiInternDataA1Dataset)
        assert ds.embodiment_bucket_counts == {"franka": 1, "lift2": 1, "split_aloha": 1}
        assert ds.action_dim == EEF_DIM
        assert len(ds) == sum(len(b) for b in ds.buckets)
        assert ds[0]["action"].shape == (8, EEF_DIM)

    def test_root_mode_rejects_removed_pooled_contributor(self, tmp_path, patch_decode):
        _make_bucket(tmp_path, "cat/split_aloha/current")
        (tmp_path / "meta").mkdir(exist_ok=True)
        (tmp_path / "meta" / "stats_split_aloha.json").write_text(
            json.dumps(
                {
                    "population": {
                        "buckets": ["cat/split_aloha/current", "cat/split_aloha/removed"],
                        "empty_buckets": [],
                    }
                }
            )
        )

        with pytest.raises(DataContractError, match="contributor set no longer matches"):
            InternDataA1Dataset.from_config(
                {"dataset_dir": str(tmp_path), "normalize_mode": "quantile"},
                split="train",
            )

    def test_single_bucket_mode(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task")
        ds = InternDataA1Dataset.from_config(
            {"dataset_dir": str(d), "normalize_mode": None, "num_frames": 9, "video_stride": 4},
            split="train",
        )
        assert isinstance(ds, InternDataA1Dataset)

    def test_bucket_ids_are_root_relative_paths(self, tmp_path, patch_decode):
        """Bucket dir names repeat across tasks, so ids must disambiguate."""
        _make_bucket(tmp_path, "cat/franka/taskA/google_scan-book", layout="single_arm", robot_type="Franka")
        _make_bucket(tmp_path, "cat/franka/taskB/google_scan-book", layout="single_arm", robot_type="Franka")
        ds = InternDataA1Dataset.from_config(
            {"dataset_dir": str(tmp_path), "normalize_mode": None, "num_frames": 9, "video_stride": 4},
            split="train",
        )
        ids = sorted(b._dataset_id for b in ds.buckets)
        assert ids == ["cat/franka/taskA/google_scan-book", "cat/franka/taskB/google_scan-book"]

    def test_empty_root_raises_pointing_at_extraction(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="extracted"):
            InternDataA1Dataset.from_config({"dataset_dir": str(tmp_path)}, split="train")

    def test_registered_under_interndata_a1(self):
        from openwam.dataloader.registry import DATASET_REGISTRY

        assert DATASET_REGISTRY["interndata_a1"] is InternDataA1Dataset


# ---------------------------------------------------------------------------
# Cleaned-view / trim regression coverage
# ---------------------------------------------------------------------------


class TestCleanedViewOffsets:
    """A cleaned view must not express deletions by dropping meta/episodes rows.

    ``_data_row_offset`` is a ``groupby(chunk, file).cumsum()`` over the rows
    currently in ``eps_df``. A cleaned view symlinks ``data/`` at the untouched
    source shards, so a shortened manifest makes every deleted episode's length
    vanish from that sum and silently slides later episodes onto earlier frames.
    ``meta/excluded_episodes.json`` is applied after the offsets are computed,
    so it does not have this failure mode.
    """

    def test_excluded_first_episode_leaves_the_second_at_its_physical_offset(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0]}))

        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

        assert ds._eps_df["episode_index"].tolist() == [1]
        # Episode 1 physically starts at row 4 of the shard; the exclusion must
        # not renumber it to 0.
        assert int(ds._ep_data_row_offset[0]) == 4
        assert ds._ep_video_frame_offsets, "no camera offsets resolved"
        for cam, off in ds._ep_video_frame_offsets.items():
            assert int(off[0]) == 4, f"{cam} offset collapsed to {int(off[0])}"

    def test_dropping_a_manifest_row_no_longer_shifts_the_survivor(self, tmp_path, patch_decode):
        """The failure mode this class was written for, now closed at the source.

        `_add_data_offsets` rebuilds each offset from `dataset_from_index` and the
        shards' real row counts, so it no longer depends on a cumsum over the
        surviving rows — a shortened manifest cannot displace anything. This once
        asserted the WRONG offset (0 instead of 4) to pin the hazard; it now
        asserts the right one, so the guarantee is pinned rather than the bug.

        excluded_episodes.json remains the correct way to express deletions —
        it keeps meta/episodes intact and self-describing — but offsets are no
        longer the reason why.
        """
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        t = pq.read_table(man)
        pq.write_table(t.slice(1, 1), man)  # keep only episode 1

        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

        assert ds._eps_df["episode_index"].tolist() == [1]
        assert int(ds._ep_data_row_offset[0]) == 4  # physical start, not 0
        # And EVERY camera, not just the parquet rows. Checking only the data
        # offset once let a half-fix advertise alignment safety while the video
        # offsets still collapsed to 0 — pairing episode 1's actions with
        # episode 0's frames, which is worse than either error alone.
        assert ds._ep_video_frame_offsets, "no camera offsets resolved"
        for cam, off in ds._ep_video_frame_offsets.items():
            assert int(off[0]) == 4, f"{cam} offset collapsed to {int(off[0])}"


class TestTrimTooShortIsLeftWhole:
    """A trim that would leave less than one window keeps the episode intact —
    and the stats path must make the identical call (they share
    :func:`resolve_trim_bounds`), or the normalizer describes episodes the
    reader never emits that way."""

    def test_reader_leaves_a_too_short_trim_whole(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=1, ep_len=4)
        trim = tmp_path / "trim.csv"
        trim.write_text(
            "dataset,episode_index,total_frames,trim_head_to,trim_tail_from\n"
            "cat/split_aloha/task,0,4,3,\n"  # would leave 1 frame < min_len 2
        )
        ds = InternDataA1Dataset(
            str(d),
            dataset_id="cat/split_aloha/task",
            normalize_mode=None,
            trim_csv=str(trim),
            num_frames=2,
            video_stride=1,
        )
        assert int(ds._eps_df["length"].iloc[0]) == 4
        assert int(ds._ep_data_row_offset[0]) == 0

    def test_resolve_trim_bounds_rejects_the_short_case_and_accepts_a_valid_one(self):
        assert resolve_trim_bounds((3, None, 4), 4, 2) is None  # leaves 1 < 2
        assert resolve_trim_bounds((1, None, 4), 4, 2) == (1, 4)  # leaves 3
        assert resolve_trim_bounds((1, None, 99), 4, 2) is None  # stale total_frames
        assert resolve_trim_bounds((0, None, 4), 4, 2) is None  # no-op


class TestStaleShardIndex:
    """`data/file_index` goes stale at shard boundaries: the episode that starts a
    new shard keeps the previous file's index.

    In affected multi-shard buckets, the base
    `groupby(chunk,file).cumsum()` points boundary episodes into the previous
    shard — a valid row range that reads back real numbers, so it pairs an
    episode with another one's frames and raises nothing.
    """

    def _two_shard_bucket(self, tmp_path):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        src = d / "data" / "chunk-000" / "file-000.parquet"
        t = pq.read_table(src)
        pq.write_table(t.slice(0, 4), src)
        pq.write_table(t.slice(4, 4), d / "data" / "chunk-000" / "file-001.parquet")
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        m["data/file_index"] = [0, 0]  # ep1 really lives in file-001
        m["dataset_from_index"] = [0, 4]
        m["dataset_to_index"] = [4, 8]
        pq.write_table(pa.Table.from_pydict(m), man)
        return d

    def test_the_boundary_episode_resolves_to_its_real_shard(self, tmp_path, patch_decode):
        d = self._two_shard_bucket(tmp_path)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)
        i = ds._eps_df.index[ds._eps_df["episode_index"] == 1][0]
        pos = list(ds._eps_df.index).index(i)
        assert int(ds._eps_df["data/file_index"].loc[i]) == 1
        assert int(ds._ep_data_row_offset[pos]) == 0

    def test_the_unaffected_episode_is_untouched(self, tmp_path, patch_decode):
        d = self._two_shard_bucket(tmp_path)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)
        i = ds._eps_df.index[ds._eps_df["episode_index"] == 0][0]
        pos = list(ds._eps_df.index).index(i)
        assert int(ds._eps_df["data/file_index"].loc[i]) == 0
        assert int(ds._ep_data_row_offset[pos]) == 0


class TestShardCompleteness:
    """A missing shard must fail loudly rather than resolve onto another episode.

    The cumulative boundaries close over whatever files exist, so with a shard
    gone every episode start still lands inside the total found on disk, and each
    later episode maps onto a plausible row of the wrong file.
    """

    def test_a_missing_middle_shard_is_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=3, ep_len=4)
        src = d / "data" / "chunk-000" / "file-000.parquet"
        t = pq.read_table(src)
        pq.write_table(t.slice(0, 4), src)
        pq.write_table(t.slice(8, 4), d / "data" / "chunk-000" / "file-002.parquet")
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        m["dataset_from_index"] = [0, 4, 8]
        m["dataset_to_index"] = [4, 8, 12]
        pq.write_table(pa.Table.from_pydict(m), man)
        with pytest.raises(ValueError, match="shard is missing"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_an_episode_overrunning_its_shard_is_rejected(self, tmp_path, patch_decode):
        """Equal totals are not enough — 3+5 physical rows satisfy a 4+4 manifest."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        src = d / "data" / "chunk-000" / "file-000.parquet"
        t = pq.read_table(src)
        pq.write_table(t.slice(0, 3), src)
        pq.write_table(t.slice(3, 5), d / "data" / "chunk-000" / "file-001.parquet")
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        m["dataset_from_index"] = [0, 4]
        m["dataset_to_index"] = [4, 8]
        pq.write_table(pa.Table.from_pydict(m), man)
        with pytest.raises(ValueError, match="shard holding only"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_complete_shards_are_accepted(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        src = d / "data" / "chunk-000" / "file-000.parquet"
        t = pq.read_table(src)
        pq.write_table(t.slice(0, 4), src)
        pq.write_table(t.slice(4, 4), d / "data" / "chunk-000" / "file-001.parquet")
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        m["dataset_from_index"] = [0, 4]
        m["dataset_to_index"] = [4, 8]
        pq.write_table(pa.Table.from_pydict(m), man)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)
        assert len(ds._eps_df) == 2


class TestVideoOffsetValidation:
    """Camera offsets come from `from_timestamp * fps`, so the inputs to that
    conversion have to be checked — a silent bad cast pairs a view with the
    wrong episode while every other signal agrees with itself."""

    def test_a_missing_timestamp_column_is_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        t = pq.read_table(man)
        pq.write_table(t.drop(["videos/images.rgb.head/from_timestamp"]), man)
        with pytest.raises(ValueError, match="frame offset cannot be resolved"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_non_finite_timestamps_are_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        m["videos/images.rgb.head/from_timestamp"] = [float("nan"), 0.1333]
        pq.write_table(pa.Table.from_pydict(m), man)
        with pytest.raises(ValueError, match="non-finite or negative"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_two_episodes_sharing_a_frame_offset_are_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        for cam in ("images.rgb.head", "images.rgb.hand_left", "images.rgb.hand_right"):
            k = f"videos/{cam}/from_timestamp"
            if k in m:
                m[k] = [0.0, 0.0]  # both episodes start at frame 0 of one shard
        pq.write_table(pa.Table.from_pydict(m), man)
        with pytest.raises(ValueError, match="overlaps episode"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_distinct_but_overlapping_frame_intervals_are_rejected(self, tmp_path, patch_decode):
        """Distinct starts are not enough — the intervals themselves must not overlap.

        Two 4-frame episodes at from_timestamp [0, 2/30] give offsets [0, 2],
        which pass a distinctness test while episode 1 reads two of episode 0's
        frames. The check that only compared starts reported this bucket clean.
        """
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        for cam in ("images.rgb.head", "images.rgb.hand_left", "images.rgb.hand_right"):
            fk, tk = f"videos/{cam}/from_timestamp", f"videos/{cam}/to_timestamp"
            if fk in m:
                m[fk] = [0.0, 2 / 30]
            if tk in m:
                m[tk] = [4 / 30, 6 / 30]
        pq.write_table(pa.Table.from_pydict(m), man)
        with pytest.raises(ValueError, match=r"overlaps episode 1 starting at 2"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_same_offset_in_different_chunks_is_accepted(self, tmp_path, patch_decode):
        """(0,0) and (1,0) are two files; both may legitimately start at frame 0.

        Grouping on file_index alone merged them and rejected valid data — a
        false rejection, which is the worse half of getting this check wrong.
        """
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        for cam in ("images.rgb.head", "images.rgb.hand_left", "images.rgb.hand_right"):
            fk, tk = f"videos/{cam}/from_timestamp", f"videos/{cam}/to_timestamp"
            ck, ik = f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"
            if fk in m:
                m[fk] = [0.0, 0.0]
            if tk in m:
                m[tk] = [4 / 30, 4 / 30]
            if ck in m:
                m[ck] = [0, 1]  # different chunks...
            if ik in m:
                m[ik] = [0, 0]  # ...same file number within each
        pq.write_table(pa.Table.from_pydict(m), man)
        r = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)
        assert len(r) > 0

    def test_a_video_span_disagreeing_with_length_is_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        tk = "videos/images.rgb.head/to_timestamp"
        if tk not in m:
            pytest.skip("fixture carries no to_timestamp column")
        m[tk] = [m["videos/images.rgb.head/from_timestamp"][0] + 9 / 30, m[tk][1]]
        pq.write_table(pa.Table.from_pydict(m), man)
        with pytest.raises(ValueError, match="describes two different episodes"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_a_non_finite_video_span_is_rejected_not_skipped(self, tmp_path, patch_decode):
        """`isfinite AND mismatch` let NaN fall through the witness entirely."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        tk = "videos/images.rgb.head/to_timestamp"
        if tk not in m:
            pytest.skip("fixture carries no to_timestamp column")
        m[tk] = [float("nan"), m[tk][1]]
        pq.write_table(pa.Table.from_pydict(m), man)
        with pytest.raises(ValueError, match="describes two different episodes"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)


class TestManifestRangeValidation:
    """Counts and capacities are both satisfied by ranges that overlap."""

    def test_overlapping_manifest_ranges_are_rejected(self, tmp_path, patch_decode):
        """[0,4) [2,6) [8,12): total is 12, every range fits, episode 1 is wrong.

        `dataset_to_index.max()` equals the physical row count and each episode
        fits inside the single 12-row shard, so both pre-existing checks pass
        while episode 1 reads two of episode 0's rows and two of its own.
        """
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=3, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        m["dataset_from_index"] = [0, 2, 8]
        m["dataset_to_index"] = [4, 6, 12]
        m["length"] = [4, 4, 4]
        pq.write_table(pa.Table.from_pydict(m), man)
        with pytest.raises(ValueError, match="Overlapping manifest ranges"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_a_range_disagreeing_with_length_is_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        m["length"] = [3, 4]  # range says 4 rows, length says 3
        pq.write_table(pa.Table.from_pydict(m), man)
        with pytest.raises(ValueError, match="declares length=3"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_a_gap_is_still_accepted(self, tmp_path, patch_decode):
        """Non-overlap is required; a perfect tiling is NOT.

        A manifest gap means rows belong to no episode, which is how a dropped
        row expresses "skip this one" — handled correctly by the offset rebuild
        and asserted by TestCleanedViewOffsets. Demanding contiguity would
        reject that working configuration to catch nothing.
        """
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=3, ep_len=4)
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        keep = [i for i in range(3) if i != 1]  # drop the middle episode
        pq.write_table(pa.Table.from_pydict({k: [v[i] for i in keep] for k, v in m.items()}), man)
        r = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)
        assert len(r._eps_df) == 2

    def test_a_substituted_equal_length_shard_is_rejected(self, tmp_path, patch_decode):
        """An equal-length copy of another shard passes every count-based check.

        file-001 replaced by a copy of file-000: the totals still balance, every
        episode still fits its resolved shard, and the reader constructs — while
        manifest episodes [0,1] both resolve onto physical episode 0. The
        shard's own episode_index range (from parquet row-group statistics, so
        no column is read) is what distinguishes them.
        """
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        data_dir = d / "data" / "chunk-000"
        src = pq.read_table(data_dir / "file-000.parquet").to_pydict()
        half = {k: v[:4] for k, v in src.items()}
        pq.write_table(pa.Table.from_pydict(half), data_dir / "file-000.parquet")
        pq.write_table(pa.Table.from_pydict(half), data_dir / "file-001.parquet")
        with pytest.raises(ValueError, match="only holds episodes 0..0"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_a_stray_backup_parquet_is_not_a_shard(self, tmp_path, patch_decode):
        """`file-000.backup.parquet` matches the glob but is not a shard.

        The reader's exact parse always dropped it; the stats generator's glob
        did not, so a copy left in place doubled the statistics population while
        the reader read half of it. Both sides now enumerate through
        `iter_data_shards`.
        """
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        data_dir = d / "data" / "chunk-000"
        shutil.copy(data_dir / "file-000.parquet", data_dir / "file-000.backup.parquet")
        assert parse_shard_path(data_dir / "file-000.backup.parquet") is None
        assert [p.name for _, _, p in iter_data_shards(d)] == ["file-000.parquet"]
        r = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)
        assert len(r) > 0

    def test_shards_are_ordered_by_index_not_lexically(self, tmp_path):
        """`file-1000` follows `file-999`; a lexical sort puts it first.

        Both names are canonical (`{:03d}` widens past three digits), so this is
        reachable on a corpus with over a thousand shards.
        """
        b = tmp_path / "b"
        (b / "data" / "chunk-000").mkdir(parents=True)
        for i in (0, 999, 1000):
            pq.write_table(pa.table({"x": [i]}), b / "data" / "chunk-000" / f"file-{i:03d}.parquet")
        assert [f for _, f, _ in iter_data_shards(b)] == [0, 999, 1000]

    def test_a_non_canonical_shard_name_is_not_a_shard(self, tmp_path, patch_decode):
        """`file-0.parquet` is a name the data_path template cannot produce.

        Sampling rebuilds the path from `info.json[data_path]` with `{:03d}`, so
        an unpadded file is enumerable but unreadable: the scan pooled its rows
        and the reader built windows over them, and only the first real data
        load failed — on a different, non-existent path. Rejecting it at
        enumeration turns that into an immediate, accurate error.
        """
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        shard = d / "data" / "chunk-000" / "file-000.parquet"
        shard.rename(shard.with_name("file-0.parquet"))
        assert parse_shard_path(shard.with_name("file-0.parquet")) is None
        assert iter_data_shards(d) == []
        with pytest.raises(FileNotFoundError, match="No data parquet files"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_an_aliased_shard_name_is_not_a_shard(self, tmp_path):
        """`file-00.parquet` would otherwise claim shard 0 alongside `file-000`."""
        b = tmp_path / "b"
        (b / "data" / "chunk-000").mkdir(parents=True)
        for name in ("file-000.parquet", "file-00.parquet"):
            pq.write_table(pa.table({"x": [0]}), b / "data" / "chunk-000" / name)
        assert [f for _, f, _ in iter_data_shards(b)] == [0]

    def test_a_canonical_shard_under_a_non_canonical_chunk_is_rejected(self, tmp_path):
        b = tmp_path / "b"
        (b / "data" / "chunk-0").mkdir(parents=True)
        pq.write_table(pa.table({"x": [0]}), b / "data" / "chunk-0" / "file-000.parquet")
        assert iter_data_shards(b) == []


class TestExclusionParser:
    """One strict parser, because the two casts disagreed."""

    def test_quoted_indices_are_refused_rather_than_reinterpreted(self, tmp_path):
        """`["0"]` excluded episode 0 in the generator and nothing in the reader.

        The base reader keeps the JSON values verbatim and matches them against
        an integer column, so a quoted index excludes nothing there, while an
        `int()` cast in the generator excluded it. Refusing is the one answer
        that is identical on both sides by construction.
        """
        b = tmp_path / "b"
        (b / "meta").mkdir(parents=True)
        (b / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": ["0"]}))
        with pytest.raises(ValueError, match="must be JSON integers"):
            load_excluded_episodes(b)

    def test_booleans_are_refused(self, tmp_path):
        """`True` passes isinstance(x, int) and would silently become episode 1."""
        b = tmp_path / "b"
        (b / "meta").mkdir(parents=True)
        (b / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [True]}))
        with pytest.raises(ValueError, match="must be JSON integers"):
            load_excluded_episodes(b)

    def test_the_actual_reader_refuses_quoted_indices_too(self, tmp_path, patch_decode):
        """The helper being strict is worthless if the reader never calls it.

        The base keeps the JSON values verbatim and matches them against an
        integer column, so `["0"]` excluded nothing there while the generator
        excluded episode 0. Asserting on the helper alone did not catch that —
        this instantiates the reader.
        """
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": ["0"]}))
        with pytest.raises(ValueError, match="must be JSON integers"):
            InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)

    def test_the_actual_reader_still_applies_integer_exclusions(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0]}))
        r = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)
        assert r._eps_df["episode_index"].tolist() == [1]

    def test_plain_integers_are_accepted(self, tmp_path):
        b = tmp_path / "b"
        (b / "meta").mkdir(parents=True)
        (b / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0, 3]}))
        assert load_excluded_episodes(b) == {0, 3}

    def test_a_missing_file_means_nothing_excluded(self, tmp_path):
        assert load_excluded_episodes(tmp_path) == set()


class TestStatsPopulationContract:
    """The four ways the generator and the reader were seen to disagree."""

    def _bucket_and_stats(self, tmp_path, **pop):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=8)
        block = {"split": "train", "trim_active": False, "min_keep": 2, "buckets": ["cat/split_aloha/task"]}
        block.update(pop)
        (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
        eef = {k: [0.0] * EEF_DIM for k in ("mean", "min", "q01")}
        eef.update({k: [1.0] * EEF_DIM for k in ("std", "max", "q99")})
        (tmp_path / "meta" / "stats_split_aloha.json").write_text(json.dumps({"eef": eef, "population": block}))
        return d

    def _read(self, d, tmp_path, **kw):
        return InternDataA1Dataset(
            str(d), a1_stats_root=str(tmp_path), normalize_mode="quantile", num_frames=2, video_stride=1, **kw
        )

    def test_val_derived_stats_are_refused(self, tmp_path, patch_decode):
        d = self._bucket_and_stats(tmp_path, split="val")
        with pytest.raises(DataContractError, match="computed over the 'val' split"):
            self._read(d, tmp_path)

    def test_train_derived_stats_are_accepted_by_a_val_reader(self, tmp_path, patch_decode):
        """A val reader must LOAD train stats, not demand val-derived ones.

        There is one stats file per embodiment and it describes the training
        distribution by definition. Comparing the recorded split against the
        reader's own split refused every val run against a correctly generated
        file — a regression with no failing test until this one.
        """
        d = self._bucket_and_stats(tmp_path, split="train")
        info = json.loads((d / "meta" / "info.json").read_text())
        info["splits"] = {"train": "0:1", "val": "1:2"}
        (d / "meta" / "info.json").write_text(json.dumps(info))
        r = self._read(d, tmp_path, split="val")
        assert r._normalization_stats is not None

    def test_untrimmed_stats_are_refused_by_a_trimming_reader(self, tmp_path, patch_decode):
        d = self._bucket_and_stats(tmp_path, trim_active=False)
        csv = tmp_path / "trim.csv"
        csv.write_text("dataset,episode_index,total_frames,trim_head_to,trim_tail_from\ncat/split_aloha/task,0,8,2,6\n")
        with pytest.raises(DataContractError, match="without a trim list but this reader is trimming"):
            self._read(d, tmp_path, trim_csv=str(csv))

    def test_trimmed_stats_are_refused_by_an_untrimmed_reader(self, tmp_path, patch_decode):
        d = self._bucket_and_stats(tmp_path, trim_active=True)
        with pytest.raises(DataContractError, match="with a trim list but this reader is not"):
            self._read(d, tmp_path)

    def test_a_different_keep_bound_is_refused(self, tmp_path, patch_decode):
        d = self._bucket_and_stats(tmp_path, trim_active=True, min_keep=33)
        csv = tmp_path / "trim.csv"
        csv.write_text("dataset,episode_index,total_frames,trim_head_to,trim_tail_from\ncat/split_aloha/task,0,8,2,6\n")
        with pytest.raises(DataContractError, match="--min_keep=33"):
            self._read(d, tmp_path, trim_csv=str(csv))

    def test_a_bucket_absent_from_the_scan_is_refused(self, tmp_path, patch_decode):
        """The bucket dropped out of the scan but still loads the shared file."""
        d = self._bucket_and_stats(tmp_path, buckets=["cat/split_aloha/other"])
        with pytest.raises(DataContractError, match="none of them this one"):
            self._read(d, tmp_path)

    def test_a_stats_file_without_the_block_is_refused(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=8)
        (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
        eef = {k: [0.0] * EEF_DIM for k in ("mean", "min", "q01")}
        eef.update({k: [1.0] * EEF_DIM for k in ("std", "max", "q99")})
        (tmp_path / "meta" / "stats_split_aloha.json").write_text(json.dumps({"eef": eef}))
        with pytest.raises(DataContractError, match="no 'population' block"):
            self._read(d, tmp_path)

    def test_root_mode_does_not_silently_drop_a_bucket_missing_from_stats(self, tmp_path, patch_decode):
        """DataContractError must escape build_multibucket's tolerant wrapper."""
        _make_bucket(tmp_path, "cat/split_aloha/good", n_eps=2, ep_len=8)
        _make_bucket(tmp_path, "cat/split_aloha/bad", n_eps=2, ep_len=8)
        (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
        eef = {k: [0.0] * EEF_DIM for k in ("mean", "min", "q01")}
        eef.update({k: [1.0] * EEF_DIM for k in ("std", "max", "q99")})
        population = {
            "split": "train",
            "trim_active": False,
            "min_keep": 2,
            "buckets": ["cat/split_aloha/good"],
            "empty_buckets": [],
        }
        (tmp_path / "meta" / "stats_split_aloha.json").write_text(json.dumps({"eef": eef, "population": population}))

        with pytest.raises(DataContractError, match="contributor set no longer matches"):
            InternDataA1Dataset.from_config(
                {
                    "dataset_dir": str(tmp_path),
                    "stats_root": str(tmp_path),
                    "normalize_mode": "quantile",
                    "num_frames": 2,
                    "video_stride": 1,
                },
                split="train",
            )

    def test_a_matching_population_loads(self, tmp_path, patch_decode):
        d = self._bucket_and_stats(tmp_path)
        assert len(self._read(d, tmp_path)) > 0
