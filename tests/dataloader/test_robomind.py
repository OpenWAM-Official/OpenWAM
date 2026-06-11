"""Unit + smoke tests for RoboMINDDataset.

Three layers:
  * pure-function tests for ``resolve_robomind_layout`` / ``robomind_raw_to_20d``
    (no disk, no reader instance);
  * stub-reader tests for the action/proprio hooks + stats↔reader parity
    (instance via ``__new__`` bypassing __init__, à la test_robocoin_normalize);
  * on-disk smoke tests that build a minimal LeRobot v3 bucket per embodiment
    and run __init__ / __getitem__ with video decoding mocked (à la test_oxe_bcz).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from openwam.dataloader.robomind import (
    MultiBucketRoboMINDDataset,
    RoboMINDDataset,
    resolve_robomind_layout,
    robomind_raw_to_20d,
)
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.utils.eef import (
    EEF_DIM,
    LEFT_ARM_DIM_MASK,
    eef14_to_eef20,
    euler_xyz_to_rot6d,
    quat_xyzw_to_rot6d,
)

P = "observation.images."
FPS = 30.0
EP_LENGTH = 40  # > num_frames(33) so a full window exists

# Per-kind embodiment descriptors: (robot_type, raw_dim, [head, left, right]).
EMB = {
    "single_euler": ("franka_panda_3rgb", 7, [P + "camera_top"]),
    "single_quat": ("franka_panda_sim", 8, [P + "camera_front_external", P + "camera_handeye"]),
    "dual_euler": (
        "agilex_cobot_magic_v2",
        14,
        [P + "camera_front", P + "camera_left_wrist", P + "camera_right_wrist"],
    ),
}


# ---------------------------------------------------------------------------
# 1. Layout resolution (pure function)
# ---------------------------------------------------------------------------


class TestResolveLayout:
    def test_single_euler_camera_top(self):
        feats = {P + "camera_top": {}}
        assert resolve_robomind_layout(feats) == (P + "camera_top", None, None, "single_euler")

    def test_single_quat_handeye(self):
        feats = {P + "camera_front_external": {}, P + "camera_handeye": {}, P + "camera_left_external": {}}
        head, left, right, kind = resolve_robomind_layout(feats)
        assert (head, left, right, kind) == (P + "camera_front_external", P + "camera_handeye", None, "single_quat")

    def test_dual_euler_wrists(self):
        feats = {P + "camera_front": {}, P + "camera_left_wrist": {}, P + "camera_right_wrist": {}}
        assert resolve_robomind_layout(feats) == (
            P + "camera_front",
            P + "camera_left_wrist",
            P + "camera_right_wrist",
            "dual_euler",
        )

    def test_franka_3rgb_extra_cams_do_not_become_wrists(self):
        # camera_left / camera_right are fixed third-person, NOT wrist → ignored.
        feats = {P + "camera_top": {}, P + "camera_left": {}, P + "camera_right": {}}
        assert resolve_robomind_layout(feats) == (P + "camera_top", None, None, "single_euler")

    def test_no_head_raises(self):
        with pytest.raises(ValueError, match="no head camera"):
            resolve_robomind_layout({P + "camera_left": {}, P + "camera_right": {}})


# ---------------------------------------------------------------------------
# 2. EEF geometry (pure function)
# ---------------------------------------------------------------------------


class TestRawTo20d:
    def test_single_euler_left_filled_right_zero(self):
        rng = np.random.RandomState(0)
        raw = rng.uniform(-1, 1, size=(5, 7)).astype(np.float32)
        out = robomind_raw_to_20d(raw, "single_euler")
        assert out.shape == (5, 20)
        # left pos passes through, left rot6d matches helper, right half zero
        np.testing.assert_allclose(out[:, 0:3], raw[:, 0:3], atol=1e-6)
        np.testing.assert_allclose(out[:, 3:9], euler_xyz_to_rot6d(raw[:, 3:6]), atol=1e-6)
        np.testing.assert_allclose(out[:, 9:10], raw[:, 6:7], atol=1e-6)
        assert (out[:, 10:20] == 0).all()

    def test_single_quat_uses_quat_rot6d(self):
        rng = np.random.RandomState(1)
        pos = rng.uniform(-1, 1, size=(4, 3)).astype(np.float32)
        quat = rng.uniform(-1, 1, size=(4, 4)).astype(np.float32)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        grip = rng.uniform(0, 1, size=(4, 1)).astype(np.float32)
        raw = np.concatenate([pos, quat, grip], axis=1)
        out = robomind_raw_to_20d(raw, "single_quat")
        assert out.shape == (4, 20)
        np.testing.assert_allclose(out[:, 3:9], quat_xyzw_to_rot6d(quat), atol=1e-6)
        assert (out[:, 10:20] == 0).all()

    def test_dual_euler_matches_eef14_reslice(self):
        rng = np.random.RandomState(2)
        raw = rng.uniform(-1, 1, size=(6, 14)).astype(np.float32)
        out = robomind_raw_to_20d(raw, "dual_euler")
        assert out.shape == (6, 20)
        # Independent reference: reslice → eef14_to_eef20.
        eef12 = np.concatenate([raw[:, 0:3], raw[:, 3:6], raw[:, 7:10], raw[:, 10:13]], axis=1)
        grip2 = np.concatenate([raw[:, 6:7], raw[:, 13:14]], axis=1)
        np.testing.assert_array_equal(out, eef14_to_eef20(eef12, grip2))
        # All 20 dims are populated (right half not all-zero in general).
        assert out[:, 10:20].any()

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown RoboMIND eef kind"):
            robomind_raw_to_20d(np.zeros((1, 7), np.float32), "nope")


# ---------------------------------------------------------------------------
# 3. Stub-reader hooks + stats↔reader parity
# ---------------------------------------------------------------------------


def _stub_reader(kind: str, mode=None, stats=None) -> RoboMINDDataset:
    r = RoboMINDDataset.__new__(RoboMINDDataset)
    r._eef_kind = kind
    r._normalize_mode = mode
    r._normalization_stats = stats
    return r


class TestHooksParity:
    def test_action_20d_equals_raw_converter_when_no_norm(self):
        rng = np.random.RandomState(3)
        raw = rng.uniform(-1, 1, size=(10, 7)).astype(np.float32)
        win = pd.DataFrame({"action": list(raw)})
        r = _stub_reader("single_euler")
        np.testing.assert_array_equal(r._action_20d(win), robomind_raw_to_20d(raw, "single_euler"))

    def test_proprio_20d_takes_first_row(self):
        rng = np.random.RandomState(4)
        raw = rng.uniform(-1, 1, size=(10, 14)).astype(np.float32)
        win = pd.DataFrame({"observation.state": list(raw)})
        r = _stub_reader("dual_euler")
        out = r._proprio_20d(win)
        assert out.shape == (1, 20)
        np.testing.assert_array_equal(out, robomind_raw_to_20d(raw[:1], "dual_euler"))


class TestNormalizationSafety:
    def test_quantile_single_arm_right_half_no_nan(self):
        # Right half is zero-range; quantile must produce finite constants there.
        stats = {
            "min": np.zeros(20, np.float32),
            "max": np.ones(20, np.float32),
            "mean": np.zeros(20, np.float32),
            "std": np.ones(20, np.float32),
            "q01": np.concatenate([-np.ones(10), np.zeros(10)]).astype(np.float32),
            "q99": np.concatenate([np.ones(10), np.zeros(10)]).astype(np.float32),
        }
        r = _stub_reader("single_euler", mode="quantile", stats=stats)
        raw = np.random.RandomState(5).uniform(-1, 1, size=(8, 7)).astype(np.float32)
        win = pd.DataFrame({"action": list(raw)})
        out = r._action_20d(win)
        assert np.isfinite(out).all()
        # zero-range dims map to a single constant (no blow-up)
        assert (out[:, 10:20] == out[0, 10:20]).all()


# ---------------------------------------------------------------------------
# 4. Multiview black-fill (uses the real layout helper, synthetic frames)
# ---------------------------------------------------------------------------


def _solid(color, h, w):
    return Image.new("RGB", (w, h), color)


class TestMultiviewBlackFill:
    def test_head_only_bottom_both_black(self):
        layout = [P + "camera_top", "__missing_left__", "__missing_right__"]
        frames = {P + "camera_top": _solid((200, 100, 50), 256, 320)}
        canvas = assemble_multiview_layout(frames, layout, out_h=384, out_w=320)
        arr = np.asarray(canvas)
        assert arr.shape == (384, 320, 3)
        assert arr[:256].sum() > 0  # main view non-black
        assert (arr[256:] == 0).all()  # both bottom slots black

    def test_head_plus_left_wrist_right_black(self):
        layout = [P + "camera_front", P + "camera_handeye", "__missing_right__"]
        frames = {
            P + "camera_front": _solid((200, 100, 50), 256, 320),
            P + "camera_handeye": _solid((10, 220, 10), 128, 160),
        }
        arr = np.asarray(assemble_multiview_layout(frames, layout, out_h=384, out_w=320))
        assert arr[256:, :160].sum() > 0  # bottom-left non-black
        assert (arr[256:, 160:] == 0).all()  # bottom-right black


# ---------------------------------------------------------------------------
# 5/6/7. On-disk bucket builder + smoke integration
# ---------------------------------------------------------------------------


def _make_state_action(kind: str, raw_dim: int, n_rows: int) -> tuple:
    rng = np.random.RandomState(11)
    state = rng.uniform(-1, 1, size=(n_rows, raw_dim)).astype(np.float32)
    if kind == "single_quat":
        q = rng.uniform(-1, 1, size=(n_rows, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        state[:, 3:7] = q  # unit quaternion so _post_init's check passes
    action = rng.uniform(-1, 1, size=(n_rows, raw_dim)).astype(np.float32)
    return state, action


def make_bucket(root: Path, kind: str, n_episodes: int = 2) -> Path:
    robot_type, raw_dim, cams = EMB[kind]
    bucket = root / f"{robot_type}_b10"
    (bucket / "meta").mkdir(parents=True, exist_ok=True)

    n_rows = n_episodes * EP_LENGTH
    state, action = _make_state_action(kind, raw_dim, n_rows)
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "task_index": np.zeros(n_rows, dtype=np.int64),
            "observation.state": list(state),
            "action": list(action),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), data_dir / "file-000.parquet")

    # episodes parquet with per-camera offsets for every resolved camera
    eps_dir = bucket / "meta" / "episodes"
    eps_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cum = 0
    for ep in range(n_episodes):
        row = {
            "episode_index": ep,
            "length": EP_LENGTH,
            "dataset_from_index": cum,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for cam in cams:
            row[f"videos/{cam}/chunk_index"] = 0
            row[f"videos/{cam}/file_index"] = 0
        rows.append(row)
        cum += EP_LENGTH
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), eps_dir / "chunk-000.parquet")

    # tasks.parquet: index = prompt text, column task_index = int
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["do the robomind task"], name="task")).to_parquet(
        bucket / "meta" / "tasks.parquet"
    )

    # video placeholders (mocked decoder ignores content)
    for cam in cams:
        vd = bucket / "videos" / cam / "chunk-000"
        vd.mkdir(parents=True, exist_ok=True)
        (vd / "file-000.mp4").write_bytes(b"")

    features = {
        "action": {"dtype": "float32", "shape": [raw_dim]},
        "observation.state": {"dtype": "float32", "shape": [raw_dim]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for cam in cams:
        features[cam] = {"dtype": "video", "shape": [3, 256, 320]}
    info = {
        "robot_type": robot_type,
        "fps": FPS,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info))
    return bucket


@contextmanager
def _mock_decode():
    def _fake(path, frame_indices, h, w):
        return [Image.new("RGB", (w, h), (40, 40, 40)) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_fake):
        yield


class TestInit:
    @pytest.mark.parametrize("kind", list(EMB))
    def test_resolve_sets_kind_and_mask(self, tmp_path, kind):
        b = make_bucket(tmp_path, kind)
        with _mock_decode():
            ds = RoboMINDDataset(dataset_dir=str(b), multiview=True)
        assert ds.eef_kind == kind
        assert ds.robot_type == EMB[kind][0]
        if kind == "dual_euler":
            assert ds.ACTION_DIM_MASK is None
        else:
            np.testing.assert_array_equal(ds.ACTION_DIM_MASK, LEFT_ARM_DIM_MASK)

    def test_missing_stats_with_quantile_raises(self, tmp_path):
        b = make_bucket(tmp_path, "single_euler")
        with pytest.raises(FileNotFoundError, match="stats file is missing"):
            RoboMINDDataset(dataset_dir=str(b), normalize_mode="quantile")


class TestGetItem:
    @pytest.mark.parametrize("kind", list(EMB))
    def test_sample_shapes_and_supervision(self, tmp_path, kind):
        b = make_bucket(tmp_path, kind)
        with _mock_decode():
            ds = RoboMINDDataset(dataset_dir=str(b), multiview=True, height=384, width=320)
            s = ds[0]
        assert len(s["video"]) == 9
        assert s["video"][0].size == (320, 384)  # PIL (W, H)
        assert s["action"].shape == (32, EEF_DIM)
        assert s["action_mask"].shape == (32, EEF_DIM)
        assert s["proprio"].shape == (1, EEF_DIM)
        assert s["proprio_mask"].shape == (1, EEF_DIM)
        assert s["action"].dtype == s["proprio"].dtype  # float32
        assert s["prompt"] == "do the robomind task"
        # supervision present; per-dim mask matches embodiment arm count
        assert bool(s["action_mask"].any())
        if kind == "dual_euler":
            assert bool(s["action_mask"][0, 10:].all())  # right arm supervised
        else:
            assert not bool(s["action_mask"][0, 10:].any())  # right arm masked off

    def test_supervision_disabled_all_false(self, tmp_path):
        b = make_bucket(tmp_path, "single_euler")
        with _mock_decode():
            ds = RoboMINDDataset(dataset_dir=str(b), multiview=True, enable_action_supervision=False)
            s = ds[0]
        assert not bool(s["action_mask"].any())
        assert not bool(s["proprio_mask"].any())


class TestRootMode:
    def test_from_config_root_discovers_buckets(self, tmp_path):
        for kind in EMB:
            make_bucket(tmp_path, kind)
        with _mock_decode():
            ds = RoboMINDDataset.from_config({"dataset_dir": str(tmp_path), "multiview": True}, split="train")
        assert isinstance(ds, MultiBucketRoboMINDDataset)
        assert len(ds.buckets) == 3
        assert ds.action_dim == EEF_DIM
        with _mock_decode():
            s = ds[0]
        assert s["action"].shape == (32, EEF_DIM)
