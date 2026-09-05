"""Tests for the AgiBotWorld dataloader (openwam/dataloader/agibotworld.py).

Focus: the ``segment_flag`` / ``segment_delta`` boundary-cleanup path and the
``segment_max_trim_ratio`` drop rule, both of which reach into the shared
``LeRobotV3Reader`` window index via the ``_valid_start`` / ``_valid_end``
columns. The end-to-end cases assert that the trimmed offset feeds BOTH the
parquet slice and the video decode — a prefix trim applied to only one of the
two would silently pair frame t with action t+delta.

Synthetic buckets are written to tmp_path; video decode is monkeypatched to
return frame-index-encoded images so alignment is checkable without mp4s.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from openwam.dataloader.agibotworld import (
    _GRIPPER_CONTRACT,
    _STATS_SCHEMA_VERSION,
    AgiBotWorldDataset,
    _validate_trim_ratio,
)
from openwam.dataloader.utils.lerobotv3 import DataContractError
from openwam.dataloader.utils.stats_computation.agibotworld_stats_computation import _partial_bucket

FPS = 15.0
HEAD = "observation.images.head"
LEFT = "observation.images.hand_left"
RIGHT = "observation.images.hand_right"
CAMS = (HEAD, LEFT, RIGHT)
NUM_FRAMES = 33
VIDEO_STRIDE = 4


# ---------------------------------------------------------------------------
# _validate_trim_ratio
# ---------------------------------------------------------------------------


class TestValidateTrimRatio:
    def test_none_passes_through(self):
        assert _validate_trim_ratio(None) is None

    @pytest.mark.parametrize("v", [0.7, 1.0, "0.5", 0.001])
    def test_valid(self, v):
        assert _validate_trim_ratio(v) == float(v)

    @pytest.mark.parametrize("v", [0.0, -0.1, 1.0001, 70, float("nan"), float("inf"), "abc", object()])
    def test_rejected(self, v):
        # 70 (percent) and 0 (would drop everything) must raise, not clamp.
        with pytest.raises(ValueError):
            _validate_trim_ratio(v)


# ---------------------------------------------------------------------------
# Synthetic bucket
# ---------------------------------------------------------------------------


def _make_bucket(
    bucket: Path,
    episodes: list[dict],
    *,
    with_segment_cols: bool = True,
    action_velocity=None,
    state_velocity=None,
    action_gripper=None,
    state_gripper=None,
) -> Path:
    """Write a minimal grippered AgiBotWorld bucket.

    ``episodes`` items: ``{"length": int, "flag": int, "delta": int}``. All
    episodes live in one data shard and one video file per camera, so the
    per-episode offsets are plain cumulative sums — which is exactly the layout
    that makes an offset bug read into the NEXT episode instead of erroring.
    """
    meta = bucket / "meta"
    (meta / "episodes").mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": FPS,
                "robot_type": "g2a",
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {"segment_flag": {"dtype": "int64"}, "segment_delta": {"dtype": "int64"}},
            }
        )
    )
    rows, cum = [], 0
    for ep, spec in enumerate(episodes):
        row = {
            "episode_index": ep,
            "length": int(spec["length"]),
            "tasks": ["do the thing"],
            "dataset_from_index": cum,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for cam in CAMS:
            row[f"videos/{cam}/chunk_index"] = 0
            row[f"videos/{cam}/file_index"] = 0
        if with_segment_cols:
            row["segment_flag"] = int(spec["flag"])
            row["segment_delta"] = int(spec["delta"])
        rows.append(row)
        cum += int(spec["length"])
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), meta / "episodes" / "chunk-000.parquet")

    pd.DataFrame({"task_index": [0]}, index=pd.Index(["do the thing"], name="task")).to_parquet(meta / "tasks.parquet")

    # data shard: ee_base[0] encodes the GLOBAL row index so a misaligned slice
    # is detectable from the action payload alone.
    total = cum

    def _values(value, dim):
        if value is None:
            return np.zeros((total, dim), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (dim,):
            arr = np.broadcast_to(arr, (total, dim)).copy()
        if arr.shape != (total, dim):
            raise ValueError(f"expected {(total, dim)}, got {arr.shape}")
        return arr

    action_vel = _values(action_velocity, 3)
    state_vel = _values(state_velocity, 3)
    action_grip = _values(action_gripper, 2)
    state_grip = _values(state_gripper, 2)
    (meta / "stats.json").write_text(
        json.dumps(
            {
                "action.robot_velocity": {
                    "min": action_vel.min(axis=0).tolist(),
                    "max": action_vel.max(axis=0).tolist(),
                },
                "observation.state.robot_velocity": {
                    "min": state_vel.min(axis=0).tolist(),
                    "max": state_vel.max(axis=0).tolist(),
                },
            }
        )
    )

    gidx = np.arange(total, dtype=np.float32)
    ee = np.zeros((total, 18), dtype=np.float32)
    ee[:, 0] = gidx
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_index = np.concatenate([np.arange(int(s["length"]), dtype=np.int64) for s in episodes])
    ep_index = np.concatenate([np.full(int(s["length"]), e, dtype=np.int64) for e, s in enumerate(episodes)])
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "task_index": np.zeros(total, dtype=np.int64),
                    "episode_index": ep_index,
                    "frame_index": frame_index,
                    "action.ee_base": list(ee),
                    "observation.state.ee_base": list(ee),
                    "action.gripper": list(action_grip),
                    "observation.state.gripper": list(state_grip),
                    "action.robot_velocity": list(action_vel),
                    "observation.state.robot_velocity": list(state_vel),
                }
            )
        ),
        data_dir / "file-000.parquet",
    )
    return bucket


@pytest.fixture
def patch_decode(monkeypatch):
    """Encode each requested frame index into pixel 0 so the decoded frames can be
    mapped back to the file-local indices the reader asked for."""

    def fake_decode(path, frame_indices, height, width):
        out = []
        for fi in frame_indices:
            im = Image.new("RGB", (width, height), (0, 0, 0))
            im.putpixel((0, 0), (int(fi) % 256, (int(fi) // 256) % 256, 7))
            out.append(im)
        return out

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", fake_decode)


def _decoded_index(canvas: Image.Image) -> int:
    r, g, _ = canvas.getpixel((0, 0))
    return r + 256 * g


def _reader(bucket: Path, **kw) -> AgiBotWorldDataset:
    return AgiBotWorldDataset(
        dataset_dir=str(bucket),
        num_frames=NUM_FRAMES,
        video_stride=VIDEO_STRIDE,
        height=384,
        width=320,
        multiview=True,
        unify_action=False,
        **kw,
    )


# ---------------------------------------------------------------------------
# Flag semantics
# ---------------------------------------------------------------------------


class TestSegmentFlags:
    def test_flag0_untouched(self, tmp_path):
        ds = _reader(_make_bucket(tmp_path / "b", [{"length": 100, "flag": 0, "delta": 0}]))
        assert ds._ep_valid_start.tolist() == [0]
        assert ds._ep_valid_end.tolist() == [100]
        assert len(ds) == 99  # final one-row start has no real next-state target

    def test_flag1_trims_prefix(self, tmp_path):
        ds = _reader(_make_bucket(tmp_path / "b", [{"length": 100, "flag": 1, "delta": 30}]))
        assert ds._ep_valid_start.tolist() == [30]
        assert ds._ep_valid_end.tolist() == [100]
        assert len(ds) == 69

    def test_flag2_trims_suffix(self, tmp_path):
        ds = _reader(_make_bucket(tmp_path / "b", [{"length": 100, "flag": 2, "delta": 30}]))
        assert ds._ep_valid_start.tolist() == [0]
        assert ds._ep_valid_end.tolist() == [70]
        assert len(ds) == 69

    def test_flag3_dropped(self, tmp_path):
        ds = _reader(
            _make_bucket(
                tmp_path / "b",
                [
                    {"length": 100, "flag": 0, "delta": 0},
                    {"length": 100, "flag": 3, "delta": 0},
                    {"length": 50, "flag": 0, "delta": 0},
                ],
            )
        )
        assert ds._eps_df["episode_index"].tolist() == [0, 2]
        assert len(ds) == 148

    def test_delta_at_or_over_length_drops_episode(self, tmp_path):
        # delta >= length leaves nothing: must drop, never produce a negative span.
        ds = _reader(
            _make_bucket(
                tmp_path / "b",
                [
                    {"length": 40, "flag": 1, "delta": 40},
                    {"length": 40, "flag": 2, "delta": 90},
                    {"length": 40, "flag": 0, "delta": 0},
                ],
            )
        )
        assert ds._eps_df["episode_index"].tolist() == [2]
        assert (ds._ep_valid_end >= ds._ep_valid_start).all()

    def test_disabled_keeps_full_segments(self, tmp_path):
        eps = [{"length": 100, "flag": 1, "delta": 30}, {"length": 100, "flag": 3, "delta": 0}]
        ds = _reader(_make_bucket(tmp_path / "b", eps), use_segment_annotations=False)
        assert len(ds._eps_df) == 2
        assert ds._ep_valid_start.tolist() == [0, 0]
        assert len(ds) == 198

    def test_missing_columns_falls_back(self, tmp_path, caplog):
        eps = [{"length": 100, "flag": 0, "delta": 0}]
        ds = _reader(_make_bucket(tmp_path / "b", eps, with_segment_cols=False))
        assert len(ds) == 99
        assert "segment annotation columns" in caplog.text


# ---------------------------------------------------------------------------
# segment_max_trim_ratio
# ---------------------------------------------------------------------------


class TestMaxTrimRatio:
    EPS = [
        {"length": 100, "flag": 1, "delta": 10},  # 10% trimmed -> keep
        {"length": 100, "flag": 2, "delta": 70},  # 70% trimmed -> boundary
        {"length": 100, "flag": 1, "delta": 90},  # 90% trimmed -> drop at 0.7
        {"length": 100, "flag": 0, "delta": 0},  # untrimmed -> always keep
    ]

    def test_none_keeps_everything(self, tmp_path):
        ds = _reader(_make_bucket(tmp_path / "b", self.EPS), segment_max_trim_ratio=None)
        assert ds._eps_df["episode_index"].tolist() == [0, 1, 2, 3]

    def test_threshold_is_inclusive(self, tmp_path):
        # "70% or more trimmed" — an episode trimmed to exactly the threshold goes.
        ds = _reader(_make_bucket(tmp_path / "b", self.EPS), segment_max_trim_ratio=0.7)
        assert ds._eps_df["episode_index"].tolist() == [0, 3]
        assert len(ds) == 89 + 99

    def test_higher_threshold_keeps_more(self, tmp_path):
        # 0.8 spares the 70%-trimmed episode and still drops the 90% one.
        ds = _reader(_make_bucket(tmp_path / "b", self.EPS), segment_max_trim_ratio=0.8)
        assert ds._eps_df["episode_index"].tolist() == [0, 1, 3]
        # above every trim ratio present, nothing is dropped.
        ds = _reader(_make_bucket(tmp_path / "b2", self.EPS), segment_max_trim_ratio=0.95)
        assert ds._eps_df["episode_index"].tolist() == [0, 1, 2, 3]

    def test_untrimmed_survives_ratio_one(self, tmp_path):
        # ratio=1.0 may only remove episodes trimmed to nothing (already dropped),
        # never a flag-0 episode.
        ds = _reader(_make_bucket(tmp_path / "b", self.EPS), segment_max_trim_ratio=1.0)
        assert 3 in ds._eps_df["episode_index"].tolist()

    def test_ignored_when_annotations_off(self, tmp_path, caplog):
        ds = _reader(
            _make_bucket(tmp_path / "b", self.EPS),
            use_segment_annotations=False,
            segment_max_trim_ratio=0.7,
        )
        assert len(ds._eps_df) == 4
        assert "ignored because" in caplog.text

    def test_config_keys_expose_both_knobs(self):
        assert "use_segment_annotations" in AgiBotWorldDataset.CONFIG_KEYS
        assert "segment_max_trim_ratio" in AgiBotWorldDataset.CONFIG_KEYS


# ---------------------------------------------------------------------------
# End-to-end: the trimmed offset must drive parquet AND video identically
# ---------------------------------------------------------------------------


class TestTrimmedOffsetAlignment:
    EPS = [
        {"length": 120, "flag": 1, "delta": 25},  # prefix trim
        {"length": 120, "flag": 2, "delta": 25},  # suffix trim
        {"length": 120, "flag": 0, "delta": 0},  # control
    ]

    def _ds(self, tmp_path):
        return _reader(_make_bucket(tmp_path / "b", self.EPS))

    def test_first_window_starts_after_the_trimmed_prefix(self, tmp_path, patch_decode):
        ds = self._ds(tmp_path)
        s = ds[0]  # first window of episode 0, whose valid range is [25, 120)
        # action payload encodes the global parquet row index; episode 0 starts at 0
        assert s["action"][0, 0].item() == pytest.approx(25.0)
        # video frames must come from the SAME offset, not from frame 0
        assert _decoded_index(s["video"][0]) == 25
        assert [_decoded_index(f) for f in s["video"]] == list(range(25, 25 + NUM_FRAMES, VIDEO_STRIDE))

    def test_window_count_matches_trimmed_span(self, tmp_path):
        ds = self._ds(tmp_path)
        assert len(ds) == 94 + 94 + 119

    def test_last_window_of_suffix_trimmed_episode_stops_at_valid_end(self, tmp_path, patch_decode):
        ds = self._ds(tmp_path)
        ep1_first = 94  # episode 0 contributed 94 supervised-window starts
        last = ep1_first + 93  # last window of episode 1
        s = ds[last]
        # episode 1 valid range is [0, 95); its last window starts at 93 and has
        # two real rows (one real next-state action), so the video mask samples
        # one frame and the payload
        # must stay inside episode 1 (global rows 120..239).
        assert int(s["video_mask"].sum()) == 1
        assert int(s["action_mask"].any(dim=1).sum()) == 1
        assert s["action"][0, 0].item() == pytest.approx(120.0 + 93.0)
        assert _decoded_index(s["video"][0]) == 120 + 93

    def test_no_window_reads_past_its_own_episode(self, tmp_path, patch_decode):
        ds = self._ds(tmp_path)
        bounds = [(0, 120), (120, 240), (240, 360)]
        starts = ds._cum_n_starts
        for i in range(len(ds)):
            ep = int(np.searchsorted(starts, i, side="right") - 1)
            lo, hi = bounds[ep]
            s = ds[i]
            rows = s["action"][s["action_mask"].any(dim=1)][:, 0].numpy()
            assert rows.min() >= lo and rows.max() < hi, f"window {i} (episode {ep}) leaked outside {lo}..{hi}"
            frames = [_decoded_index(f) for f, m in zip(s["video"], s["video_mask"].tolist()) if m]
            assert min(frames) >= lo and max(frames) < hi

    def test_trimmed_frames_are_never_served(self, tmp_path, patch_decode):
        ds = self._ds(tmp_path)
        seen = set()
        for i in range(len(ds)):
            s = ds[i]
            seen.update(int(r) for r in s["action"][s["action_mask"].any(dim=1)][:, 0].numpy())
        # episode 0 rows 0..24 (trimmed prefix) and episode 1 rows 215..239
        # (trimmed suffix, global 120+95 .. 120+119) must be absent.
        assert not (seen & set(range(0, 25)))
        assert not (seen & set(range(120 + 95, 240)))
        # Every kept row with a real successor must appear. The last kept row of
        # each episode (119/214/359) is a clamped action and stays unsupervised.
        expected = set(range(25, 119)) | set(range(120, 214)) | set(range(240, 359))
        assert seen == expected


# ---------------------------------------------------------------------------
# Independent action/state movement masks + canonical gripper direction
# ---------------------------------------------------------------------------


class TestPhysicalSemantics:
    @staticmethod
    def _unified_reader(bucket: Path) -> AgiBotWorldDataset:
        return AgiBotWorldDataset(
            dataset_dir=str(bucket),
            num_frames=NUM_FRAMES,
            video_stride=VIDEO_STRIDE,
            height=384,
            width=320,
            multiview=True,
            unify_action=True,
            normalize_mode=None,
        )

    def test_command_only_base_motion_keeps_action_and_masks_proprio(self, tmp_path, patch_decode):
        bucket = _make_bucket(
            tmp_path / "b",
            [{"length": 40, "flag": 0, "delta": 0}],
            action_velocity=[1.25, 0.0, -0.5],
            state_velocity=[0.0, 0.0, 0.0],
            action_gripper=[0.0, 1.0],  # source: left open, right closed
            state_gripper=[0.035, 0.125],  # source actuator: left open, right closed
        )
        sample = self._unified_reader(bucket)[0]

        # Source action 0=open / 1=closed is inverted into 0=closed / 1=open.
        assert sample["action"][0, 9].item() == pytest.approx(1.0)
        assert sample["action"][0, 43].item() == pytest.approx(0.0)
        # State closing-actuator endpoints become the same canonical aperture fraction.
        assert sample["proprio"][0, 9].item() == pytest.approx(1.0)
        assert sample["proprio"][0, 43].item() == pytest.approx(0.0)

        assert sample["action_mask"][:, 68].all()
        assert sample["action_mask"][:, 70].all()
        assert not sample["proprio_mask"][:, 68].any()
        assert not sample["proprio_mask"][:, 70].any()
        # Masked values are kept numerically neutral as well.
        assert sample["proprio"][0, 68].item() == 0.0
        assert sample["proprio"][0, 70].item() == 0.0
        assert sample["action"][0, 68].item() == pytest.approx(1.25)
        assert sample["action"][0, 70].item() == pytest.approx(-0.5)

    def test_state_only_base_motion_masks_action_independently(self, tmp_path, patch_decode):
        bucket = _make_bucket(
            tmp_path / "b",
            [{"length": 40, "flag": 0, "delta": 0}],
            action_velocity=[0.0, 0.0, 0.0],
            state_velocity=[0.25, 0.0, 0.4],
        )
        sample = self._unified_reader(bucket)[0]

        assert not sample["action_mask"][:, 68].any()
        assert not sample["action_mask"][:, 70].any()
        assert sample["proprio_mask"][:, 68].all()
        assert sample["proprio_mask"][:, 70].all()
        assert sample["action"][0, 68].item() == 0.0
        assert sample["action"][0, 70].item() == 0.0
        assert sample["proprio"][0, 68].item() == pytest.approx(0.25)
        assert sample["proprio"][0, 70].item() == pytest.approx(0.4)

    def test_non_unified_path_does_not_require_base_motion_stats(self, tmp_path, patch_decode):
        bucket = _make_bucket(
            tmp_path / "b",
            [{"length": 40, "flag": 0, "delta": 0}],
        )
        (bucket / "meta" / "stats.json").unlink()

        ds = AgiBotWorldDataset(
            dataset_dir=str(bucket),
            num_frames=NUM_FRAMES,
            video_stride=VIDEO_STRIDE,
            normalize_mode=None,
            unify_action=False,
        )
        sample = ds[0]
        assert sample["action"].shape[-1] == 20
        assert sample["proprio"].shape[-1] == 20

    def test_stats_use_independent_motion_population_and_invert_action_gripper(self, tmp_path):
        bucket = _make_bucket(
            tmp_path / "b",
            [{"length": 10, "flag": 0, "delta": 0}],
            action_velocity=[1.0, 0.0, -1.0],
            state_velocity=[0.0, 0.0, 0.0],
            action_gripper=[0.2, 0.8],
            state_gripper=[0.035, 0.125],
        )
        _, partial, _ = _partial_bucket(str(bucket), "train", True, 0.7)

        assert "action.robot_velocity" in partial
        assert "observation.state.robot_velocity" not in partial
        np.testing.assert_allclose(partial["action.gripper"]["mean"], [0.8, 0.2])
        np.testing.assert_allclose(partial["observation.state.gripper"]["mean"], [1.0, 0.0])


class TestTemporalAlignment:
    def test_truncated_tail_drops_clamped_final_target(self, tmp_path, patch_decode):
        bucket = _make_bucket(
            tmp_path / "b",
            [{"length": 12, "flag": 0, "delta": 0}],
        )
        ds = AgiBotWorldDataset(
            dataset_dir=str(bucket),
            num_frames=9,
            video_stride=4,
            normalize_mode=None,
        )

        assert ds._n_supervised_action_steps(9) == 9
        assert ds._n_supervised_action_steps(4) == 3
        assert ds._n_supervised_action_steps(1) == 0
        # Offset 3 is a full 9-row window: T_action=8 already omits the final
        # source row. Offset 10 has two rows and must supervise only the first.
        assert int(ds[3]["action_mask"].any(dim=1).sum()) == 8
        assert int(ds[len(ds) - 1]["action_mask"].any(dim=1).sum()) == 1

    def test_one_row_episode_is_not_sampleable(self, tmp_path, patch_decode):
        bucket = _make_bucket(
            tmp_path / "b",
            [
                {"length": 1, "flag": 0, "delta": 0},
                {"length": 4, "flag": 0, "delta": 0},
            ],
        )
        ds = AgiBotWorldDataset(
            dataset_dir=str(bucket),
            num_frames=9,
            video_stride=4,
            normalize_mode=None,
        )

        assert ds._train_min_window_len() == 2
        assert len(ds) == 3
        assert all(ds[i]["action_mask"].any() for i in range(len(ds)))


# ---------------------------------------------------------------------------
# Normalization stats must use and attest to the same trimmed population
# ---------------------------------------------------------------------------


class TestTrimAwareStats:
    @staticmethod
    def _stats_block(dim: int) -> dict:
        return {
            "mean": [0.0] * dim,
            "std": [1.0] * dim,
            "min": [-1.0] * dim,
            "max": [1.0] * dim,
            "q01": [-1.0] * dim,
            "q99": [1.0] * dim,
        }

    def test_partial_bucket_excludes_trimmed_and_blacklisted_rows(self, tmp_path):
        bucket = _make_bucket(
            tmp_path / "root" / "b",
            [
                {"length": 10, "flag": 1, "delta": 2},  # keep global rows 2..9
                {"length": 10, "flag": 2, "delta": 3},  # keep global rows 10..16
                {"length": 10, "flag": 3, "delta": 0},  # static: drop
                {"length": 10, "flag": 0, "delta": 0},  # generic exclusion: drop
                {"length": 10, "flag": 1, "delta": 8},  # >=70% trimmed: drop
            ],
        )
        (bucket / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [3]}))

        name, partial, population = _partial_bucket(str(bucket), "train", True, 0.7)

        assert name == "b"
        assert population["num_episodes"] == 2
        assert population["num_rows"] == 15
        assert population["num_action_rows"] == 13
        assert population["num_state_rows"] == 15
        assert population["excluded_episode_indices"] == [3]
        assert partial["action.ee_base"]["count"] == 13
        assert partial["observation.state.ee_base"]["count"] == 15
        # ee_base[0] encodes the global physical row in the fixture.  These
        # bounds prove that neither prefix/suffix outside the kept spans leaked.
        assert partial["action.ee_base"]["min"][0] == pytest.approx(2.0)
        assert partial["action.ee_base"]["max"][0] == pytest.approx(15.0)
        assert partial["observation.state.ee_base"]["max"][0] == pytest.approx(16.0)

    def test_reader_accepts_matching_population_and_rejects_trim_policy_drift(self, tmp_path):
        root = tmp_path / "root"
        bucket = _make_bucket(
            root / "b",
            [{"length": 40, "flag": 1, "delta": 4}, {"length": 40, "flag": 0, "delta": 0}],
        )
        _, _partial, bucket_population = _partial_bucket(str(bucket), "train", True, 0.7)
        stats = {
            "robot_type": "g2a",
            "gripper_contract": _GRIPPER_CONTRACT,
            "population": {
                "schema_version": _STATS_SCHEMA_VERSION,
                "split": "train",
                "use_segment_annotations": True,
                "segment_max_trim_ratio": 0.7,
                "buckets": {"b": bucket_population},
            },
            "action.ee_base": self._stats_block(18),
            "observation.state.ee_base": self._stats_block(18),
            "action.gripper": self._stats_block(2),
            "observation.state.gripper": self._stats_block(2),
        }
        (root / "meta").mkdir(exist_ok=True)
        stats_path = root / "meta" / "stats_g2a.json"
        stats_path.write_text(json.dumps(stats))

        ds = _reader(bucket, normalize_mode="quantile", segment_max_trim_ratio=0.7)
        # Reader-side materialization pins rot6d fields to identity regardless
        # of the affine stats stored on disk.
        np.testing.assert_array_equal(ds._action_norm_stats["mean"][3:9], np.zeros(6))
        np.testing.assert_array_equal(ds._action_norm_stats["std"][3:9], np.ones(6))

        stats["gripper_contract"] = {**_GRIPPER_CONTRACT, "open_endpoint_m": 0.036}
        stats_path.write_text(json.dumps(stats))
        with pytest.raises(DataContractError, match="gripper_contract"):
            _reader(bucket, normalize_mode="quantile", segment_max_trim_ratio=0.7)

        stats["gripper_contract"] = _GRIPPER_CONTRACT
        stats["population"]["segment_max_trim_ratio"] = 0.8
        stats_path.write_text(json.dumps(stats))
        with pytest.raises(DataContractError, match="generated with"):
            _reader(bucket, normalize_mode="quantile", segment_max_trim_ratio=0.7)

    def test_reader_rejects_non_train_stats(self, tmp_path):
        root = tmp_path / "root"
        bucket = _make_bucket(root / "b", [{"length": 40, "flag": 0, "delta": 0}])
        _, _partial, bucket_population = _partial_bucket(str(bucket), "train", True, 0.7)
        stats = {
            "robot_type": "g2a",
            "gripper_contract": _GRIPPER_CONTRACT,
            "population": {
                "schema_version": _STATS_SCHEMA_VERSION,
                "split": "val",
                "use_segment_annotations": True,
                "segment_max_trim_ratio": 0.7,
                "buckets": {"b": bucket_population},
            },
            "action.ee_base": self._stats_block(18),
            "observation.state.ee_base": self._stats_block(18),
            "action.gripper": self._stats_block(2),
            "observation.state.gripper": self._stats_block(2),
        }
        (root / "meta").mkdir(exist_ok=True)
        (root / "meta" / "stats_g2a.json").write_text(json.dumps(stats))

        with pytest.raises(DataContractError, match="train-derived"):
            _reader(bucket, normalize_mode="quantile", segment_max_trim_ratio=0.7)

    def test_reader_rejects_removed_pooled_contributor(self, tmp_path):
        root = tmp_path / "root"
        bucket = _make_bucket(root / "b", [{"length": 40, "flag": 0, "delta": 0}])
        _, _partial, bucket_population = _partial_bucket(str(bucket), "train", True, 0.7)
        stats = {
            "robot_type": "g2a",
            "gripper_contract": _GRIPPER_CONTRACT,
            "population": {
                "schema_version": _STATS_SCHEMA_VERSION,
                "split": "train",
                "use_segment_annotations": True,
                "segment_max_trim_ratio": 0.7,
                # ``removed`` no longer exists on disk but still polluted the
                # pooled numbers in this stale file.
                "buckets": {"b": bucket_population, "removed": bucket_population},
            },
            "action.ee_base": self._stats_block(18),
            "observation.state.ee_base": self._stats_block(18),
            "action.gripper": self._stats_block(2),
            "observation.state.gripper": self._stats_block(2),
        }
        (root / "meta").mkdir(exist_ok=True)
        (root / "meta" / "stats_g2a.json").write_text(json.dumps(stats))

        with pytest.raises(DataContractError, match="pooled contributor set"):
            _reader(bucket, normalize_mode="quantile", segment_max_trim_ratio=0.7)
