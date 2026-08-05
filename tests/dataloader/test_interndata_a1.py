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
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from openwam.dataloader.interndata_a1 import (
    ROBOT_TYPE_TO_EMBODIMENT,
    AmbiguousBucketKey,
    InternDataA1Dataset,
    MultiInternDataA1Dataset,
    detect_arm_layout,
    discover_a1_buckets,
    embodiment_key,
    exclusion_digest,
    resolve_bucket_key,
    resolve_trim_bounds,
    trim_digest,
)
from openwam.dataloader.utils.eef import (
    ARM10_DIM,
    EEF_DIM,
    LEFT_ARM_DIM_MASK,
    quat_wxyz_to_rot6d,
    quat_xyzw_to_rot6d,
)
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
        """extract_interndata_a1_v30.sh stages into <cat>/<emb>/.partial_<name>/.
        SIGKILL/OOM/preemption bypasses its cleanup, so a half-extracted tree
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
        """A normalized Franka bucket already store 0/1 — dividing them by
        the 0.08 stroke would blow them up to 12.5."""
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
        """A synthetic outlier — surfaced, not fatal."""
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
        branch were collapsed into `logger.warning` — which would mean warning
        fatigue on every legitimate alternate-stroke bucket.
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
        """The log-space flip sits at sqrt(0.08*1.0)=0.283, and this dataset's sim
        synthetic outliers can cross the decision boundary (an extreme outlier versus the normal stroke). One glitch row at
        an in-range synthetic maximum therefore reclassifies a panda bucket as Robotiq and squashes
        its real values 12.5x — and that maximum stays under _GRIPPER_SANE_MAX, so the
        out-of-range warning never fires. The mean must escalate it."""
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
    def _write_stats(self, root: Path, embodiment: str, *, pin_rot6d: bool = True):
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
        (root / "meta" / f"stats_{embodiment}.json").write_text(json.dumps({"eef": eef}))

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
            json.dumps({"eef": {k: [0.0] * 10 for k in ("mean", "std", "min", "max", "q01", "q99")}})
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
    vanish from that sum and slides each later episode onto earlier frames —
    In one affected case, many episodes displaced, worst-case large offset
    frames, with no error at runtime. ``meta/excluded_episodes.json`` is applied
    after the offsets are computed, so it does not have this failure mode.
    """

    def test_excluded_first_episode_leaves_the_second_at_its_physical_offset(
        self, tmp_path, patch_decode
    ):
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


class TestTrimStatsProvenance:
    """Trimmed and untrimmed stats are not interchangeable, and the numbers
    alone cannot say which is which — so the pairing is checked, by content
    digest rather than by path (the cleaned view gets relocated)."""

    def _write_stats(self, root: Path, embodiment: str, **extra):
        (root / "meta").mkdir(parents=True, exist_ok=True)
        eef = {
            "mean": [0.0] * EEF_DIM,
            "std": [1.0] * EEF_DIM,
            "min": [-2.0] * EEF_DIM,
            "max": [2.0] * EEF_DIM,
            "q01": [-2.0] * EEF_DIM,
            "q99": [2.0] * EEF_DIM,
        }
        for dim in ROT6D_DIMS_EEF20:
            eef["q01"][dim] = -1.0
            eef["q99"][dim] = 1.0
        (root / "meta" / f"stats_{embodiment}.json").write_text(json.dumps({"eef": eef, **extra}))

    def _write_trim(self, path: Path, rows: str) -> Path:
        path.write_text("dataset,episode_index,total_frames,trim_head_to,trim_tail_from\n" + rows)
        return path

    def test_untrimmed_stats_with_a_trimmed_reader_is_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=40)
        self._write_stats(tmp_path, "split_aloha")  # no provenance recorded
        trim = self._write_trim(tmp_path / "trim.csv", "cat/split_aloha/task,0,40,5,\n")
        with pytest.raises(ValueError, match="not interchangeable"):
            InternDataA1Dataset(
                str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
                normalize_mode="quantile", trim_csv=str(trim), num_frames=9, video_stride=4,
            )

    def test_trimmed_stats_with_an_untrimmed_reader_is_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=40)
        trim = self._write_trim(tmp_path / "trim.csv", "cat/split_aloha/task,0,40,5,\n")
        self._write_stats(tmp_path, "split_aloha", trim_digest=trim_digest(str(trim)))
        with pytest.raises(ValueError, match="not interchangeable"):
            InternDataA1Dataset(
                str(d), a1_stats_root=str(tmp_path), normalize_mode="quantile",
                num_frames=9, video_stride=4,
            )

    def test_matching_digest_is_accepted(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=40)
        trim = self._write_trim(tmp_path / "trim.csv", "cat/split_aloha/task,0,40,5,\n")
        self._write_stats(tmp_path, "split_aloha", trim_digest=trim_digest(str(trim)))
        ds = InternDataA1Dataset(
            str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
            normalize_mode="quantile", trim_csv=str(trim), num_frames=9, video_stride=4,
        )
        assert ds._normalization_stats is not None

    def test_same_content_at_a_different_path_is_accepted(self, tmp_path, patch_decode):
        """Digest, not path — otherwise relocating the cleaned view invalidates
        a stats file that is in fact a perfect match."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=40)
        row = "cat/split_aloha/task,0,40,5,\n"
        a = self._write_trim(tmp_path / "trim.csv", row)
        (tmp_path / "moved").mkdir()
        b = self._write_trim(tmp_path / "moved" / "trim.csv", row)
        self._write_stats(tmp_path, "split_aloha", trim_digest=trim_digest(str(a)))
        ds = InternDataA1Dataset(
            str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
            normalize_mode="quantile", trim_csv=str(b), num_frames=9, video_stride=4,
        )
        assert ds._normalization_stats is not None


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
            str(d), dataset_id="cat/split_aloha/task", normalize_mode=None,
            trim_csv=str(trim), num_frames=2, video_stride=1,
        )
        assert int(ds._eps_df["length"].iloc[0]) == 4
        assert int(ds._ep_data_row_offset[0]) == 0

    def test_resolve_trim_bounds_rejects_the_short_case_and_accepts_a_valid_one(self):
        assert resolve_trim_bounds((3, None, 4), 4, 2) is None      # leaves 1 < 2
        assert resolve_trim_bounds((1, None, 4), 4, 2) == (1, 4)    # leaves 3
        assert resolve_trim_bounds((1, None, 99), 4, 2) is None     # stale total_frames
        assert resolve_trim_bounds((0, None, 4), 4, 2) is None      # no-op


class TestTrimMinKeepProvenance:
    """The same trim CSV under a different ``--min_keep`` yields a different kept
    population — episodes short enough to fall under the bound are left whole
    instead of trimmed — and the CSV digest cannot see that."""

    def _stats(self, root: Path, **extra):
        (root / "meta").mkdir(parents=True, exist_ok=True)
        eef = {
            "mean": [0.0] * EEF_DIM, "std": [1.0] * EEF_DIM,
            "min": [-2.0] * EEF_DIM, "max": [2.0] * EEF_DIM,
            "q01": [-2.0] * EEF_DIM, "q99": [2.0] * EEF_DIM,
        }
        for dim in ROT6D_DIMS_EEF20:
            eef["q01"][dim] = -1.0
            eef["q99"][dim] = 1.0
        (root / "meta" / "stats_split_aloha.json").write_text(json.dumps({"eef": eef, **extra}))

    def _trim(self, path: Path) -> str:
        path.write_text(
            "dataset,episode_index,total_frames,trim_head_to,trim_tail_from\n"
            "cat/split_aloha/task,0,40,20,\n"
        )
        return str(path)

    def test_min_keep_mismatch_is_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=1, ep_len=40)
        trim = self._trim(tmp_path / "trim.csv")
        # Stats built with min_keep=33: the 40-frame episode's trim would leave
        # 20 < 33, so the stats kept it whole. The train reader's bound is 2, so
        # it trims to 20. Same digest, different population.
        self._stats(tmp_path, trim_digest=trim_digest(trim), trim_min_keep=33)
        with pytest.raises(ValueError, match="min_keep"):
            InternDataA1Dataset(
                str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
                normalize_mode="quantile", trim_csv=trim, num_frames=9, video_stride=4,
            )

    def test_val_split_loads_the_train_generated_stats(self, tmp_path, patch_decode):
        """Stats must come from the training distribution, so a val reader is
        expected to load a train-generated file even though its own bound is
        `num_frames`. Enforcing the match on val would make one stats file
        unusable for every val run."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=1, ep_len=40)
        trim = self._trim(tmp_path / "trim.csv")
        self._stats(tmp_path, trim_digest=trim_digest(trim), trim_min_keep=2)
        ds = InternDataA1Dataset(
            str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
            normalize_mode="quantile", trim_csv=trim, num_frames=9, video_stride=4,
            split="val",
        )
        assert ds._normalization_stats is not None
        assert ds._trim_min_len() == 9  # its own bound is still num_frames

    def test_matching_min_keep_is_accepted(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=1, ep_len=40)
        trim = self._trim(tmp_path / "trim.csv")
        self._stats(tmp_path, trim_digest=trim_digest(trim), trim_min_keep=2)
        ds = InternDataA1Dataset(
            str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
            normalize_mode="quantile", trim_csv=trim, num_frames=9, video_stride=4,
        )
        assert ds._normalization_stats is not None


class TestExclusionProvenance:
    """Deletions are a second population filter, independent of the trim CSV:
    the stats scanner honours ``meta/excluded_episodes.json``, so editing it
    changes which rows entered the normalizer while ``trim_digest`` is unchanged."""

    def _stats(self, root: Path, **extra):
        eef = {
            "mean": [0.0] * EEF_DIM, "std": [1.0] * EEF_DIM,
            "min": [-2.0] * EEF_DIM, "max": [2.0] * EEF_DIM,
            "q01": [-2.0] * EEF_DIM, "q99": [2.0] * EEF_DIM,
        }
        for dim in ROT6D_DIMS_EEF20:
            eef["q01"][dim] = -1.0
            eef["q99"][dim] = 1.0
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "meta" / "stats_split_aloha.json").write_text(json.dumps({"eef": eef, **extra}))

    def test_exclusions_added_after_stats_were_built_is_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=20)
        self._stats(tmp_path, exclusions={"cat/split_aloha/task": None})  # built with none
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0]}))
        with pytest.raises(ValueError, match="exclusion digest"):
            InternDataA1Dataset(
                str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
                normalize_mode="quantile", num_frames=9, video_stride=4,
            )

    def test_matching_exclusions_are_accepted(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=20)
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0]}))
        self._stats(tmp_path, exclusions={"cat/split_aloha/task": exclusion_digest(d)})
        ds = InternDataA1Dataset(
            str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
            normalize_mode="quantile", num_frames=9, video_stride=4,
        )
        assert ds._normalization_stats is not None

    def test_legacy_stats_load_only_when_nothing_is_excluded(self, tmp_path, patch_decode):
        """Backward compatibility must not become a hole.

        A file predating the check may well describe the full population. It is
        safe to accept only while this bucket excludes nothing; the moment it
        does, that file may cover rows the reader never emits.
        """
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=20)
        self._stats(tmp_path)  # no "exclusions" key at all
        ds = InternDataA1Dataset(
            str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
            normalize_mode="quantile", num_frames=9, video_stride=4,
        )
        assert ds._normalization_stats is not None

        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0]}))
        with pytest.raises(ValueError, match="predates exclusion provenance"):
            InternDataA1Dataset(
                str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
                normalize_mode="quantile", num_frames=9, video_stride=4,
            )

    def test_a_map_that_cannot_resolve_this_bucket_is_rejected(self, tmp_path, patch_decode):
        """A generated map lists EVERY scanned bucket (null for the ones that
        exclude nothing), so an unresolvable entry means these stats were not
        computed over this bucket — not that it has no exclusions."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=20)
        self._stats(tmp_path, exclusions={"other/split_aloha/different": None})
        with pytest.raises(ValueError, match="none of them resolves to this one"):
            InternDataA1Dataset(
                str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
                normalize_mode="quantile", num_frames=9, video_stride=4,
            )

    def test_digest_ignores_formatting_and_order(self, tmp_path):
        a = _make_bucket(tmp_path, "a/split_aloha/t", n_eps=2, ep_len=4)
        b = _make_bucket(tmp_path, "b/split_aloha/t", n_eps=2, ep_len=4)
        (a / "meta" / "excluded_episodes.json").write_text('{"episode_indices": [1, 0]}')
        (b / "meta" / "excluded_episodes.json").write_text(
            '{\n  "episode_indices": [\n    0,\n    1\n  ]\n}\n'
        )
        assert exclusion_digest(a) == exclusion_digest(b)

    def test_single_bucket_mode_resolves_the_exclusion_entry_by_suffix(self, tmp_path, patch_decode):
        """Without an explicit dataset_id the reader's id is the bare directory
        name, while the stats map is keyed by bucket path. Looking it up
        directly reported the bucket's own exclusions as unrecorded and rejected
        valid data — it must resolve the same way the trim list does."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=20)
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0]}))
        self._stats(tmp_path, exclusions={"cat/split_aloha/task": exclusion_digest(d)})
        ds = InternDataA1Dataset(  # no dataset_id
            str(d), a1_stats_root=str(tmp_path), normalize_mode="quantile",
            num_frames=9, video_stride=4,
        )
        assert ds._normalization_stats is not None

    def test_an_unrecorded_exclusion_is_still_rejected(self, tmp_path, patch_decode):
        """The suffix fallback must not become a way to skip the check: a bucket
        that excludes episodes but appears nowhere in the map is a genuine
        mismatch."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=20)
        (d / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [0]}))
        self._stats(tmp_path, exclusions={"other/split_aloha/elsewhere": "deadbeefdeadbeef"})
        with pytest.raises(ValueError, match="none of them resolves to this one"):
            InternDataA1Dataset(
                str(d), a1_stats_root=str(tmp_path), normalize_mode="quantile",
                num_frames=9, video_stride=4,
            )


class TestStaleShardIndex:
    """`data/file_index` goes stale at shard boundaries — the episode that starts
    a new shard keeps the previous file's index.

    In affected multi-shard buckets, every one of the affected multi-shard buckets is
    affected (~shards-1 episodes each, many episodes total), while all 22
    single-shard buckets are clean. The base `groupby(chunk,file).cumsum()` then
    points those episodes into the PREVIOUS shard — a valid row range that reads
    back real numbers, so it pairs an episode with another one's frames and
    raises nothing.
    """

    def _two_shard_bucket(self, tmp_path):
        """Two shards, with the second episode's `data/file_index` left stale at 0."""
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=4)
        src = d / "data" / "chunk-000" / "file-000.parquet"
        t = pq.read_table(src)
        pq.write_table(t.slice(0, 4), src)                                   # shard 0: ep0
        pq.write_table(t.slice(4, 4), d / "data" / "chunk-000" / "file-001.parquet")  # shard 1: ep1
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        m["data/file_index"] = [0, 0]          # ep1 is really in file-001 — this is the stale bit
        m["dataset_from_index"] = [0, 4]       # global row index, correct
        m["dataset_to_index"] = [4, 8]
        pq.write_table(pa.Table.from_pydict(m), man)
        return d

    def test_the_boundary_episode_resolves_to_its_real_shard(self, tmp_path, patch_decode):
        d = self._two_shard_bucket(tmp_path)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)
        i = ds._eps_df.index[ds._eps_df["episode_index"] == 1][0]
        pos = list(ds._eps_df.index).index(i)
        assert int(ds._eps_df["data/file_index"].loc[i]) == 1, "still pointing at the stale shard"
        assert int(ds._ep_data_row_offset[pos]) == 0, "offset should be file-local to file-001"

    def test_the_unaffected_episode_is_untouched(self, tmp_path, patch_decode):
        d = self._two_shard_bucket(tmp_path)
        ds = InternDataA1Dataset(str(d), normalize_mode=None, num_frames=2, video_stride=1)
        i = ds._eps_df.index[ds._eps_df["episode_index"] == 0][0]
        pos = list(ds._eps_df.index).index(i)
        assert int(ds._eps_df["data/file_index"].loc[i]) == 0
        assert int(ds._ep_data_row_offset[pos]) == 0


class TestShardCompleteness:
    """A missing middle shard must fail loudly, not resolve onto another episode.

    The cumulative boundaries close over whatever files exist, so with file-001
    gone every episode start still lands inside the total found on disk — and
    each later episode maps to a plausible row of the wrong file. The manifest's
    own end index is the independent witness.
    """

    def test_a_missing_middle_shard_is_rejected(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=3, ep_len=4)
        src = d / "data" / "chunk-000" / "file-000.parquet"
        t = pq.read_table(src)
        pq.write_table(t.slice(0, 4), src)                                            # ep0
        pq.write_table(t.slice(8, 4), d / "data" / "chunk-000" / "file-002.parquet")  # ep2; ep1's shard absent
        man = d / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        m = pq.read_table(man).to_pydict()
        m["dataset_from_index"] = [0, 4, 8]
        m["dataset_to_index"] = [4, 8, 12]
        pq.write_table(pa.Table.from_pydict(m), man)

        with pytest.raises(ValueError, match="shard is missing"):
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


class TestStatsSplitProvenance:
    """Normalization must be derived from the training distribution — a val
    reader consumes train-derived stats too — so the check is on how the file
    was GENERATED, not on who is reading it."""

    def _stats(self, root: Path, **extra):
        eef = {
            "mean": [0.0] * EEF_DIM, "std": [1.0] * EEF_DIM,
            "min": [-2.0] * EEF_DIM, "max": [2.0] * EEF_DIM,
            "q01": [-2.0] * EEF_DIM, "q99": [2.0] * EEF_DIM,
        }
        for dim in ROT6D_DIMS_EEF20:
            eef["q01"][dim] = -1.0
            eef["q99"][dim] = 1.0
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "meta" / "stats_split_aloha.json").write_text(json.dumps({"eef": eef, **extra}))

    def test_val_generated_stats_are_refused_by_a_train_reader(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=20)
        self._stats(tmp_path, split="val")
        with pytest.raises(ValueError, match="generated from split"):
            InternDataA1Dataset(
                str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
                normalize_mode="quantile", num_frames=9, video_stride=4,
            )

    def test_train_generated_stats_are_accepted_by_a_val_reader(self, tmp_path, patch_decode):
        d = _make_bucket(tmp_path, "cat/split_aloha/task", n_eps=2, ep_len=20)
        self._stats(tmp_path, split="train")
        ds = InternDataA1Dataset(
            str(d), dataset_id="cat/split_aloha/task", a1_stats_root=str(tmp_path),
            normalize_mode="quantile", num_frames=9, video_stride=4, split="val",
        )
        assert ds._normalization_stats is not None


class TestAmbiguousBucketName:
    """`<cat>/<emb>/<task>/<object>` is a documented depth, so repeating leaf
    names is expected — the error must say that, and prescribe something that
    actually helps (regenerating produces the same names)."""

    def test_the_message_names_the_real_cause_and_a_working_remedy(self):
        with pytest.raises(AmbiguousBucketKey) as e:
            resolve_bucket_key(
                {"a/emb/task/obj": None, "b/emb/task/obj": None},
                dataset_id="obj", dir_name="obj", what="trim_csv", source="test",
            )
        msg = str(e.value)
        assert "repeat across tasks" in msg
        assert "dataset_id" in msg
        assert "regenerat" not in msg.lower(), "must not prescribe a remedy that reproduces it"

    def test_a_unique_suffix_still_resolves(self):
        assert resolve_bucket_key(
            {"a/emb/task": None, "b/emb/other": None},
            dataset_id="task", dir_name="task", what="trim_csv", source="test",
        ) == "a/emb/task"
