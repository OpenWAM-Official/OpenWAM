"""Unit tests for BehaviorDataset (BEHAVIOR-1K, LeRobot v2.1, R1Pro).

Builds a minimal **v2.1** bucket on disk — one parquet per episode under
``data/task-{chunk:04d}/episode_{idx:08d}.parquet`` plus ``meta/episodes.jsonl``
(NOT the v3 ``meta/episodes/*.parquet`` layout) — and exercises the reader's
v2.1 IO overrides, the 256-D-state→EEF extraction, the unified-80D scatter
(base velocity at [68:71)), normalization against ``stats_R1Pro.json``, and the
companion stats-computation script. Video decoding is mocked so no mp4 bytes are
needed; the mask / value / prompt contracts are verified end-to-end.
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

from openwam.dataloader.behavior import (
    _ACT_BASE,
    _ACT_LGRIP,
    _ACT_RGRIP,
    _L_EEF_POS,
    _L_EEF_QUAT,
    _R_EEF_POS,
    _R_EEF_QUAT,
    BehaviorDataset,
)
from openwam.dataloader.utils.unify_action import UNIFY_DIM

EP_LENGTH = 40
FPS = 30.0
CHUNKS_SIZE = 10000
STATE_DIM = 256
ACTION_DIM = 23
HEAD = "observation.images.rgb.head"
LWRIST = "observation.images.rgb.left_wrist"
RWRIST = "observation.images.rgb.right_wrist"
CAMS = (HEAD, LWRIST, RWRIST)
# Unified slots that carry real data: L eef+grip [0:10), R eef+grip [34:44),
# base velocity [68:71). Everything else (dex hands, reserved tail) is masked.
EXPECTED_VALID = list(range(0, 10)) + list(range(34, 44)) + list(range(68, 71))
UNIFY_MAP = ["0-9", "34-43", "68-70"]


def _unit_quats(rng: np.random.RandomState, n: int) -> np.ndarray:
    """``(n, 4)`` xyzw unit quaternions (norm == 1, as the reader asserts)."""
    q = rng.uniform(-1, 1, size=(n, 4)).astype(np.float64)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.astype(np.float32)


def _make_state(rng: np.random.RandomState, n: int, *, unit_quats: bool = True) -> np.ndarray:
    """``(n, 256)`` state with valid EEF pos + (unit) quats at the reader's offsets."""
    state = rng.uniform(-1, 1, size=(n, STATE_DIM)).astype(np.float32)
    state[:, _L_EEF_POS] = rng.uniform(0.1, 0.6, size=(n, 3))
    state[:, _R_EEF_POS] = rng.uniform(0.1, 0.6, size=(n, 3))
    lq = _unit_quats(rng, n)
    rq = _unit_quats(rng, n)
    if not unit_quats:  # break the invariant to exercise the _post_init guard
        lq *= 3.0
        rq *= 3.0
    state[:, _L_EEF_QUAT] = lq
    state[:, _R_EEF_QUAT] = rq
    return state


def _make_action(rng: np.random.RandomState, n: int) -> np.ndarray:
    """``(n, 23)`` action: base velocity at [0:3], binary {-1,+1} grippers."""
    action = rng.uniform(-1, 1, size=(n, ACTION_DIM)).astype(np.float32)
    action[:, _ACT_BASE] = rng.uniform(-0.3, 0.3, size=(n, 3))  # base vel
    action[:, _ACT_LGRIP] = rng.choice([-1.0, 1.0], size=n)
    action[:, _ACT_RGRIP] = rng.choice([-1.0, 1.0], size=n)
    return action


def _write_episode_parquet(bucket: Path, ep: int, *, unit_quats: bool = True) -> None:
    chunk = ep // CHUNKS_SIZE
    data_dir = bucket / "data" / f"task-{chunk:04d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(100 + ep)
    df = pd.DataFrame(
        {
            "observation.state": list(_make_state(rng, EP_LENGTH, unit_quats=unit_quats)),
            "action": list(_make_action(rng, EP_LENGTH)),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), data_dir / f"episode_{ep:08d}.parquet")


def _write_episode_videos(bucket: Path, ep: int) -> None:
    chunk = ep // CHUNKS_SIZE
    for cam in CAMS:
        vdir = bucket / "videos" / f"task-{chunk:04d}" / cam
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / f"episode_{ep:08d}.mp4").write_bytes(b"")  # mocked decoder ignores content


def _write_episodes_jsonl(bucket: Path, episodes: list[int]) -> None:
    (bucket / "meta").mkdir(parents=True, exist_ok=True)
    with open(bucket / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps({"episode_index": ep, "length": EP_LENGTH, "tasks": [f"do task {ep}"]}) + "\n")


def _write_info(bucket: Path) -> None:
    (bucket / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "fps": FPS,
        "chunks_size": CHUNKS_SIZE,
        "robot_type": "R1Pro",
        # v2.1 templates (reader's _post_init rewrites these to {chunk_index}/{file_index}).
        "data_path": "data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet",
        "video_path": "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4",
        "features": {
            HEAD: {"dtype": "video", "shape": [720, 720, 3]},
            LWRIST: {"dtype": "video", "shape": [480, 480, 3]},
            RWRIST: {"dtype": "video", "shape": [480, 480, 3]},
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
        },
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info))


def make_behavior_bucket(
    tmp_path: Path,
    n_episodes: int = 2,
    *,
    on_disk: list[int] | None = None,
    unit_quats: bool = True,
    with_stats: bool = False,
) -> Path:
    """Build a synthetic BEHAVIOR-1K v2.1 bucket.

    ``on_disk`` (default: all) selects which jsonl episodes actually get a parquet +
    videos written — episodes listed in the jsonl but absent on disk must be dropped
    by the reader (partial-download robustness).
    """
    bucket = tmp_path / "behaviour-1k"
    bucket.mkdir(parents=True, exist_ok=True)
    episodes = list(range(n_episodes))
    on_disk = episodes if on_disk is None else on_disk
    _write_info(bucket)
    _write_episodes_jsonl(bucket, episodes)
    for ep in on_disk:
        _write_episode_parquet(bucket, ep, unit_quats=unit_quats)
        _write_episode_videos(bucket, ep)
    if with_stats:
        from openwam.dataloader.utils.stats_computation.behavior_stats_computation import compute_behavior_stats

        result = compute_behavior_stats(bucket)
        (bucket / "meta" / "stats_R1Pro.json").write_text(json.dumps(result))
    return bucket


@contextmanager
def _mock_video_decoder():
    def _fake(path, frame_indices, h, w):
        return [Image.new("RGB", (w, h), (0, 0, 0)) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_fake):
        yield


def _make_ds(bucket: Path, **kw):
    kw.setdefault("multiview", True)
    kw.setdefault("unify_action", True)
    kw.setdefault("unify_action_map", UNIFY_MAP)
    kw.setdefault("normalize_mode", None)
    return BehaviorDataset(dataset_dir=str(bucket), height=384, width=320, **kw)


# ── init / IO -----------------------------------------------------------------


class TestInit:
    def test_loads_v21_bucket(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=3)
        with _mock_video_decoder():
            ds = _make_ds(b)
        assert len(ds._eps_df) == 3
        assert ds.action_dim == UNIFY_DIM == 80
        assert ds._dataset_id == "behaviour-1k"

    def test_partial_download_drops_absent_episodes(self, tmp_path):
        # 4 episodes in jsonl, only 0 and 2 on disk → reader keeps exactly those.
        b = make_behavior_bucket(tmp_path, n_episodes=4, on_disk=[0, 2])
        with _mock_video_decoder():
            ds = _make_ds(b)
        assert sorted(ds._eps_df["episode_index"].tolist()) == [0, 2]

    def test_non_unit_quat_raises(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=1, unit_quats=False)
        with _mock_video_decoder(), pytest.raises(ValueError, match="unit-norm"):
            _make_ds(b)


# ── shapes / masks / unified layout -------------------------------------------


class TestGetItem:
    def test_action_proprio_shapes(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            s = _make_ds(b)[0]
        assert s["action"].shape == (32, 80)
        assert s["proprio"].shape == (1, 80)
        assert s["action_mask"].shape == (32, 80)
        assert s["proprio_mask"].shape == (1, 80)

    def test_unified_valid_dims(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            s = _make_ds(b)[0]
        am = s["action_mask"].numpy()
        assert sorted(np.where(am[0])[0].tolist()) == EXPECTED_VALID
        # all 32 (< EP_LENGTH) timesteps valid on the mapped dims
        assert am[:, EXPECTED_VALID].all()
        pm = s["proprio_mask"].numpy()
        assert sorted(np.where(pm[0])[0].tolist()) == EXPECTED_VALID

    def test_dex_and_reserved_slots_zero(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            s = _make_ds(b)[0]
        a = s["action"].numpy()
        assert (a[:, 10:34] == 0).all()  # L hand
        assert (a[:, 44:68] == 0).all()  # R hand
        assert (a[:, 71:80] == 0).all()  # reserved tail
        assert np.isfinite(a).all()

    def test_base_velocity_present(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            s = _make_ds(b)[0]
        base = s["action"].numpy()[:, 68:71]
        assert np.isfinite(base).all()
        assert np.abs(base).sum() > 0  # base velocity actually written

    def test_multiview_canvas_size(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            s = _make_ds(b)[0]
        assert len(s["video"]) == 9  # (33-1)//4 + 1
        assert s["video"][0].size == (320, 384)  # PIL (W, H)


# ── prompts -------------------------------------------------------------------


class TestPrompt:
    def test_prompt_from_episodes_jsonl(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            ds = _make_ds(b)
            assert ds[0]["prompt"] == "do task 0"
            # ep 0 yields EP_LENGTH windows (train min len 1) → idx EP_LENGTH starts ep 1
            assert ds[EP_LENGTH]["prompt"] == "do task 1"


# ── normalization -------------------------------------------------------------


class TestNormalize:
    def test_quantile_loads_stats_and_clips(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        with _mock_video_decoder():
            ds = _make_ds(b, normalize_mode="quantile")
            s = ds[0]
        assert ds._normalization_stats is not None
        a = s["action"].numpy()
        # mapped dims clip to [-1, 1] under quantile
        assert (np.abs(a[:, EXPECTED_VALID]) <= 1.0 + 1e-5).all()

    def test_quantile_rot6d_passthrough(self, tmp_path):
        # rot6d stats are pinned to identity → normalized rot6d == raw rot6d.
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        with _mock_video_decoder():
            raw = _make_ds(b, normalize_mode=None)[0]["action"].numpy()
            norm = _make_ds(b, normalize_mode="quantile")[0]["action"].numpy()
        # L rot6d = unified [3:9], R rot6d = [37:43]
        for sl in (slice(3, 9), slice(37, 43)):
            np.testing.assert_allclose(norm[:, sl], raw[:, sl], atol=1e-5)

    def test_null_skips_stats(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            ds = _make_ds(b, normalize_mode=None)
        assert ds._normalization_stats is None

    def test_missing_stats_with_quantile_raises(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)  # no stats file written
        with _mock_video_decoder(), pytest.raises(FileNotFoundError, match="stats_R1Pro.json"):
            _make_ds(b, normalize_mode="quantile")


# ── stats-computation script --------------------------------------------------


class TestStatsScript:
    def test_schema_and_rot6d_pin(self, tmp_path):
        from openwam.dataloader.utils.stats_computation.behavior_stats_computation import compute_behavior_stats

        b = make_behavior_bucket(tmp_path, n_episodes=3)
        result = compute_behavior_stats(b)
        assert set(result) == {"eef", "base_vel"}
        eef, base = result["eef"], result["base_vel"]
        for k in ("mean", "std", "min", "max", "q01", "q99"):
            assert len(eef[k]) == 20
            assert len(base[k]) == 3
        # rot6d dims (3:9 / 13:19) pinned to identity
        assert eef["rot6d_identity"] is True
        for i in (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18):
            assert eef["min"][i] == -1.0 and eef["max"][i] == 1.0
            assert eef["q01"][i] == -1.0 and eef["q99"][i] == 1.0
            assert eef["mean"][i] == 0.0 and eef["std"][i] == 1.0
        # base velocity is NOT pinned (real stats from data)
        assert "rot6d_identity" not in base
        assert base["layout"] == "vx,vy,vyaw"

    def test_no_rot6d_identity_flag(self, tmp_path):
        from openwam.dataloader.utils.stats_computation.behavior_stats_computation import compute_behavior_stats

        b = make_behavior_bucket(tmp_path, n_episodes=2)
        result = compute_behavior_stats(b, rot6d_identity=False)
        assert result["eef"]["rot6d_identity"] is False


# ── pickle / from_config ------------------------------------------------------


class TestPickle:
    def test_round_trip(self, tmp_path):
        import pickle

        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            ds = _make_ds(b)
            ds2 = pickle.loads(pickle.dumps(ds))
            s = ds2[0]
        assert s["action"].shape == (32, 80)


class TestFromConfig:
    def test_from_config_single_bucket(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        cfg = {
            "type": "behavior",
            "dataset_dir": str(b),
            "multiview": True,
            "height": 384,
            "width": 320,
            "normalize_mode": "quantile",
            "unify_action": True,
            "unify_action_map": UNIFY_MAP,
        }
        with _mock_video_decoder():
            ds = BehaviorDataset.from_config(cfg, split="train")
            s = ds[0]
        assert s["action"].shape == (32, 80)
        assert s["proprio"].shape == (1, 80)
