"""Unit tests for Bridge / RT-1 / DROID OXE readers.

All three readers share the LeRobotV3Reader scaffolding; their differences
are: head camera key, optional wrist camera, state schema (BC-Z-style
Euler / RT-1 quat / DROID two-column cartesian), and action source column.
This file exercises each through synthetic LeRobot v3 buckets with
mocked video decode.
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
from scipy.spatial.transform import Rotation as _R

from openwam.dataloader.oxe_bridge import OxeBridgeDataset
from openwam.dataloader.oxe_droid import OxeDroidDataset
from openwam.dataloader.oxe_rt1 import OxeRt1Dataset
from openwam.dataloader.utils.eef import EEF_DIM

EP_LENGTH = 60


def _write_episodes(bucket: Path, n_episodes: int, head_cam: str, wrist_cam: str | None = None) -> None:
    eps_dir = bucket / "meta" / "episodes"
    eps_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cum = 0
    for ep in range(n_episodes):
        r = {
            "episode_index": ep,
            "length": EP_LENGTH,
            "dataset_from_index": cum,
            "data/chunk_index": 0,
            "data/file_index": 0,
            f"videos/{head_cam}/chunk_index": 0,
            f"videos/{head_cam}/file_index": 0,
        }
        if wrist_cam is not None:
            r[f"videos/{wrist_cam}/chunk_index"] = 0
            r[f"videos/{wrist_cam}/file_index"] = 0
        rows.append(r)
        cum += EP_LENGTH
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), eps_dir / "chunk-000.parquet")


def _write_tasks(bucket: Path) -> None:
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick the block"], name="task")).to_parquet(
        bucket / "meta" / "tasks.parquet"
    )


def _write_tasks_annotated(bucket: Path, n_episodes: int) -> None:
    """Per-episode LLM-rewritten prompts (what the reader actually consumes)."""
    df = pd.DataFrame(
        {"task": [f"pick the block — episode {i}" for i in range(n_episodes)]},
        index=pd.Index(range(n_episodes), name="episode_index"),
    )
    df.to_parquet(bucket / "meta" / "tasks_annotated.parquet")


def _write_video_placeholder(bucket: Path, cam: str) -> None:
    vid_dir = bucket / "videos" / cam / "chunk-000"
    vid_dir.mkdir(parents=True, exist_ok=True)
    (vid_dir / "file-000.mp4").write_bytes(b"")


def _write_eef_stats(bucket: Path) -> None:
    stats = {
        "n_samples": EP_LENGTH * 2,
        "min": [-1.0] * 10,
        "max": [1.0] * 10,
        "mean": [0.0] * 10,
        "std": [0.5] * 10,
        "q01": [-0.9] * 10,
        "q99": [0.9] * 10,
    }
    (bucket / "meta" / "eef_stats.json").write_text(json.dumps(stats))


@contextmanager
def _mock_decoder():
    def _fake(path, frame_indices, h, w):
        return [Image.new("RGB", (w, h), (0, 0, 0)) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_fake):
        yield


def _write_bcz_style_data(bucket: Path, n_rows: int) -> None:
    """Write data parquet matching BC-Z / Bridge schema (8-D state + 7-D action)."""
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    state = rng.uniform(-0.5, 0.5, size=(n_rows, 8)).astype(np.float32)
    state[:, 6] = 0.0
    state[:, 7] = rng.uniform(0, 1, size=n_rows)
    action = rng.uniform(-0.5, 0.5, size=(n_rows, 7)).astype(np.float32)
    action[:, 6] = rng.uniform(0, 1, size=n_rows)
    df = pd.DataFrame(
        {
            "task_index": np.zeros(n_rows, dtype=np.int64),
            "observation.state": list(state),
            "action": list(action),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), data_dir / "file-000.parquet")


def _write_rt1_data(bucket: Path, n_rows: int) -> None:
    """Write RT-1 data: state[8] with quat xyzw at [3:7]."""
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(11)
    state = np.zeros((n_rows, 8), dtype=np.float32)
    state[:, :3] = rng.uniform(-0.5, 0.5, size=(n_rows, 3))
    state[:, 3:7] = _R.random(n_rows, random_state=rng).as_quat().astype(np.float32)
    state[:, 7] = rng.uniform(0, 1, size=n_rows)
    action = rng.uniform(-0.5, 0.5, size=(n_rows, 7)).astype(np.float32)
    action[:, 6] = rng.uniform(0, 1, size=n_rows)
    df = pd.DataFrame(
        {
            "task_index": np.zeros(n_rows, dtype=np.int64),
            "observation.state": list(state),
            "action": list(action),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), data_dir / "file-000.parquet")


def _write_droid_data(bucket: Path, n_rows: int) -> None:
    """Write DROID data: cartesian[6] + gripper[1] + action.original[7]."""
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(13)
    cart = rng.uniform(-0.5, 0.5, size=(n_rows, 6)).astype(np.float32)
    grip = rng.uniform(0, 1, size=(n_rows, 1)).astype(np.float32)
    action_orig = rng.uniform(-0.5, 0.5, size=(n_rows, 7)).astype(np.float32)
    action_orig[:, 6] = rng.uniform(0, 1, size=n_rows)
    df = pd.DataFrame(
        {
            "task_index": np.zeros(n_rows, dtype=np.int64),
            "observation.state.cartesian_position": list(cart),
            "observation.state.gripper_position": grip[:, 0],  # scalar column
            "action.original": list(action_orig),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), data_dir / "file-000.parquet")


def _make_info(bucket: Path) -> None:
    info = {
        "fps": 10.0,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info))


# ---------------------------------------------------------------------------
# Bridge tests
# ---------------------------------------------------------------------------


def _make_bridge_bucket(tmp_path: Path, n_episodes: int = 2) -> Path:
    b = tmp_path / "Bridge-Dataset"
    b.mkdir(parents=True, exist_ok=True)
    (b / "meta").mkdir(exist_ok=True)
    _make_info(b)
    _write_episodes(b, n_episodes, head_cam="observation.images.image_0")
    _write_tasks(b)
    _write_tasks_annotated(b, n_episodes)
    _write_bcz_style_data(b, n_episodes * EP_LENGTH)
    _write_video_placeholder(b, "observation.images.image_0")
    _write_eef_stats(b)
    return b


class TestBridge:
    def test_loads(self, tmp_path):
        b = _make_bridge_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeBridgeDataset(dataset_dir=str(b))
            sample = ds[0]
        assert sample["action"].shape == (32, EEF_DIM)
        assert sample["action_mask"].shape == (32, EEF_DIM)
        assert sample["proprio_mask"].shape == (1, EEF_DIM)

    def test_left_arm_filled(self, tmp_path):
        b = _make_bridge_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeBridgeDataset(dataset_dir=str(b))
            sample = ds[0]
        assert sample["action"][:, :10].abs().sum() > 0
        assert (sample["action"][:, 10:] == 0).all()
        assert sample["proprio_mask"][0, :10].all()
        assert not sample["proprio_mask"][0, 10:].any()

    def test_uses_image_0_camera(self, tmp_path):
        b = _make_bridge_bucket(tmp_path)
        ds = OxeBridgeDataset(dataset_dir=str(b))
        assert ds.HEAD_CAMERA == "observation.images.image_0"


# ---------------------------------------------------------------------------
# RT-1 tests
# ---------------------------------------------------------------------------


def _make_rt1_bucket(tmp_path: Path, n_episodes: int = 2) -> Path:
    b = tmp_path / "RT-1-Dataset"
    b.mkdir(parents=True, exist_ok=True)
    (b / "meta").mkdir(exist_ok=True)
    _make_info(b)
    _write_episodes(b, n_episodes, head_cam="observation.images.image")
    _write_tasks(b)
    _write_tasks_annotated(b, n_episodes)
    _write_rt1_data(b, n_episodes * EP_LENGTH)
    _write_video_placeholder(b, "observation.images.image")
    _write_eef_stats(b)
    return b


class TestRt1:
    def test_loads(self, tmp_path):
        b = _make_rt1_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeRt1Dataset(dataset_dir=str(b))
            sample = ds[0]
        assert sample["action"].shape == (32, EEF_DIM)
        # quat-based proprio still produces valid 20-D
        assert sample["proprio"].shape == (1, EEF_DIM)

    def test_quat_sanity_check_passes_on_unit_quats(self, tmp_path):
        # _post_init reads sample state and asserts ‖q‖≈1; our fake data
        # uses scipy random_state which produces unit quats — should not raise.
        b = _make_rt1_bucket(tmp_path)
        OxeRt1Dataset(dataset_dir=str(b))  # no raise

    def test_quat_sanity_check_fails_on_non_unit_quats(self, tmp_path):
        b = _make_rt1_bucket(tmp_path)
        # Overwrite data parquet with non-unit quats (0.5 norm)
        data_path = b / "data" / "chunk-000" / "file-000.parquet"
        df = pd.read_parquet(data_path)
        states = np.stack(df["observation.state"].values).copy()
        states[:, 3:7] = 0.5  # all 0.5 → norm sqrt(0.25*4)=1.0 actually — make it
        states[:, 3] = 0.1
        states[:, 4] = 0.1
        states[:, 5] = 0.1
        states[:, 6] = 0.1  # norm sqrt(0.04) ≈ 0.2 — way off
        df["observation.state"] = list(states)
        pq.write_table(pa.Table.from_pandas(df), data_path)
        import pytest

        with pytest.raises(ValueError, match="Quaternion norm check failed"):
            OxeRt1Dataset(dataset_dir=str(b))


# ---------------------------------------------------------------------------
# DROID tests
# ---------------------------------------------------------------------------


def _make_droid_bucket(tmp_path: Path, n_episodes: int = 2) -> Path:
    b = tmp_path / "DROID-Dataset"
    b.mkdir(parents=True, exist_ok=True)
    (b / "meta").mkdir(exist_ok=True)
    _make_info(b)
    _write_episodes(
        b,
        n_episodes,
        head_cam="observation.images.exterior_1_left",
        wrist_cam="observation.images.wrist_left",
    )
    _write_tasks(b)
    _write_tasks_annotated(b, n_episodes)
    _write_droid_data(b, n_episodes * EP_LENGTH)
    _write_video_placeholder(b, "observation.images.exterior_1_left")
    _write_video_placeholder(b, "observation.images.wrist_left")
    _write_eef_stats(b)
    return b


class TestDroid:
    def test_loads(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeDroidDataset(dataset_dir=str(b))
            sample = ds[0]
        assert sample["action"].shape == (32, EEF_DIM)
        assert sample["proprio"].shape == (1, EEF_DIM)

    def test_uses_action_original_not_joint(self, tmp_path):
        # DROID's reader requests action.original column; if a bucket only
        # supplies the joint-space "action" column it should fail.
        b = _make_droid_bucket(tmp_path)
        ds = OxeDroidDataset(dataset_dir=str(b))
        assert "action.original" in ds.NEEDED_COLS
        assert "observation.state.cartesian_position" in ds.NEEDED_COLS
        # Default joint-space action column is NOT requested
        assert "action" not in ds.NEEDED_COLS
        assert "observation.state" not in ds.NEEDED_COLS

    def test_uses_cartesian_state(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        ds = OxeDroidDataset(dataset_dir=str(b))
        assert "observation.state.cartesian_position" in ds.NEEDED_COLS
        assert "observation.state.gripper_position" in ds.NEEDED_COLS

    def test_camera_layout_has_left_wrist(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        ds = OxeDroidDataset(dataset_dir=str(b), multiview=True)
        # multiview=True → [head, left_wrist, right_wrist_or_missing]
        assert ds._camera_layout[0] == "observation.images.exterior_1_left"
        assert ds._camera_layout[1] == "observation.images.wrist_left"
        # right_wrist is None → "__missing_right__"
        assert ds._camera_layout[2] == "__missing_right__"

    def test_wrist_decode_failure_is_tolerated(self, tmp_path):
        """When the wrist mp4 raises (e.g. file missing on disk in the real
        DROID bucket: 254/74604 wrist_left clips are absent), the reader must
        return a usable sample with an empty wrist slot rather than propagating
        the error. Propagating it triggers a DataLoader-worker death → rank-0
        early iterator EOF → DDP deadlock (rank 0 in destroy_process_group,
        peer ranks still in backward NCCL allreduce)."""
        from unittest.mock import patch

        b = _make_droid_bucket(tmp_path)

        def _decoder(path, frame_indices, h, w):
            # head slot (256x320) succeeds; wrist slot (128x160) raises.
            if (h, w) == (128, 160):
                raise FileNotFoundError(path)
            return [Image.new("RGB", (w, h), (0, 0, 0)) for _ in frame_indices]

        with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_decoder):
            ds = OxeDroidDataset(dataset_dir=str(b), multiview=True)
            sample = ds[0]
        # Sample is usable: head slot present, wrist slot fell back to black.
        assert sample["action"].shape == (32, EEF_DIM)
        assert len(sample["video"]) > 0
        # The fail counter recorded the wrist failures (one per video frame).
        assert ds._wrist_fail_count >= 1


# ---------------------------------------------------------------------------
# Normalization modes — every reader must support min-max / z-score / quantile
# (+ null passthrough). All three share LeRobotV3Reader stats loading +
# apply_normalization, so this matrix locks the contract across the readers.
# ---------------------------------------------------------------------------

_READERS = [
    (_make_bridge_bucket, OxeBridgeDataset),
    (_make_rt1_bucket, OxeRt1Dataset),
    (_make_droid_bucket, OxeDroidDataset),
]


class TestNormalizeModes:
    @pytest.mark.parametrize("make_bucket, cls", _READERS, ids=["bridge", "rt1", "droid"])
    @pytest.mark.parametrize("mode", ["min-max", "z-score", "quantile"])
    def test_mode_loads_stats_and_normalizes(self, tmp_path, make_bucket, cls, mode):
        b = make_bucket(tmp_path)
        with _mock_decoder():
            ds = cls(dataset_dir=str(b), normalize_mode=mode)
            sample = ds[0]
        # Every non-null mode loads the per-dataset stats dict.
        assert ds._normalization_stats is not None
        assert sample["action"].shape == (32, EEF_DIM)
        assert sample["action"][:, :10].abs().sum() > 0
        # All modes must yield finite values.
        assert sample["action"].isfinite().all()
        assert sample["proprio"].isfinite().all()
        # The bounded modes clip the active left-arm dims into [-1, 1];
        # z-score is unbounded by design, so only finiteness is asserted there.
        if mode in ("min-max", "quantile"):
            assert (sample["action"][:, :10].abs() <= 1.0 + 1e-5).all()
            assert (sample["proprio"][:, :10].abs() <= 1.0 + 1e-5).all()

    @pytest.mark.parametrize("make_bucket, cls", _READERS, ids=["bridge", "rt1", "droid"])
    def test_null_mode_skips_stats(self, tmp_path, make_bucket, cls):
        b = make_bucket(tmp_path)
        with _mock_decoder():
            ds = cls(dataset_dir=str(b), normalize_mode=None)
            sample = ds[0]
        assert ds._normalization_stats is None
        assert sample["action"].shape == (32, EEF_DIM)


# ---------------------------------------------------------------------------
# enable_action_supervision=False — Bridge / RT-1 / DROID.
# Mirrors TestEnableActionSupervisionFalse in test_oxe_bcz.py. Locks the
# video-only-auxiliary contract across the three remaining OXE readers:
# both masks integer-zero everywhere, while action/proprio VALUES and the
# video/prompt payload stay intact so the source still contributes frames.
# ---------------------------------------------------------------------------


class TestEnableActionSupervisionFalse:
    @pytest.mark.parametrize("make_bucket, cls", _READERS, ids=["bridge", "rt1", "droid"])
    def test_supervision_off_masks_zero(self, tmp_path, make_bucket, cls):
        b = make_bucket(tmp_path)
        with _mock_decoder():
            ds = cls(dataset_dir=str(b), enable_action_supervision=False)
            s = ds[0]
        # Masks entirely False — no action/proprio supervision signal.
        assert not s["action_mask"].any()
        assert not s["proprio_mask"].any()
        # Shapes unchanged.
        assert s["action_mask"].shape == (32, EEF_DIM)
        assert s["proprio_mask"].shape == (1, EEF_DIM)
        # Values still loaded (just masked out), video/prompt still present.
        assert s["action"][:, :10].abs().sum() > 0
        assert s["proprio"][:, :10].abs().sum() > 0
        assert len(s["video"]) > 0
        assert isinstance(s["prompt"], str) and len(s["prompt"]) > 0

    def test_supervision_off_droid_multiview(self, tmp_path):
        # DROID carries a left wrist camera → exercises the L-shape multiview
        # assembly path with supervision off (head + wrist slots both decoded).
        b = _make_droid_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeDroidDataset(
                dataset_dir=str(b),
                multiview=True,
                height=384,
                width=320,
                enable_action_supervision=False,
            )
            s = ds[0]
        assert not s["action_mask"].any()
        assert not s["proprio_mask"].any()
        assert s["action"].shape == (32, EEF_DIM)
        assert len(s["video"]) > 0

    @pytest.mark.parametrize("make_bucket, cls", _READERS, ids=["bridge", "rt1", "droid"])
    def test_from_config_supervision_off(self, tmp_path, make_bucket, cls):
        # The mixture/training path reads the flag from yaml via from_config;
        # `enable_action_supervision: false` must reach the reader as False
        # (False is not None → forwarded), not get dropped as a falsy value.
        b = make_bucket(tmp_path)
        cfg = {
            "type": cls.DATASET_NAME,
            "dataset_dir": str(b),
            "normalize_mode": "quantile",
            "enable_action_supervision": False,
        }
        with _mock_decoder():
            ds = cls.from_config(cfg, split="train")
            s = ds[0]
        assert ds._enable_action_supervision is False
        assert not s["action_mask"].any()
        assert not s["proprio_mask"].any()
