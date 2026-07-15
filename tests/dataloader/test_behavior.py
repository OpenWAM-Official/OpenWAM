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
    _ACT_LARM,
    _ACT_LGRIP,
    _ACT_RARM,
    _ACT_RGRIP,
    _ACT_TRUNK,
    _ARM_JOINT_DIM,
    _BASE_QVEL,
    _BASE_YAW,
    _JOINT_DIM,
    _L_ARM_QPOS,
    _L_ARM_QPOS_SIN,
    _L_EEF_POS,
    _L_EEF_QUAT,
    _L_GRIP_QPOS,
    _R_ARM_QPOS,
    _R_ARM_QPOS_SIN,
    _R_EEF_POS,
    _R_EEF_QUAT,
    _R_GRIP_QPOS,
    _TRUNK_QPOS,
    BehaviorDataset,
    _state_to_raw_proprio_eef,
    _state_to_raw_proprio_joint,
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
# base velocity [68:71), trunk [71:75). Everything else (dex hands, reserved
# tail [75:80)) is masked.
EXPECTED_VALID = list(range(0, 10)) + list(range(34, 44)) + list(range(68, 71)) + list(range(71, 75))
UNIFY_MAP = ["0-9", "34-43", "68-70", "71-74"]


def _unit_quats(rng: np.random.RandomState, n: int) -> np.ndarray:
    """``(n, 4)`` xyzw unit quaternions (norm == 1, as the reader asserts)."""
    q = rng.uniform(-1, 1, size=(n, 4)).astype(np.float64)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.astype(np.float32)


def _make_state(rng: np.random.RandomState, n: int, *, unit_quats: bool = True) -> np.ndarray:
    """``(n, 256)`` state with valid ACHIEVED proprio channels at the reader's offsets.

    Populates every field the proprio path + the ``_post_init`` layout guard read:
    EEF pos/quat, arm qpos (with its ``sin(qpos)`` block — the proprio_obs invariant
    the guard checks), 2-finger gripper qpos ∈ [0, 0.05], trunk qpos, world-frame
    base velocity, and the base yaw used for the world→base rotation. Other dims stay
    random (the reader never reads them).
    """
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
    # achieved arm qpos + its sin-block (guard checks sin(qpos)==sin-block)
    for qsl, ssl in ((_L_ARM_QPOS, _L_ARM_QPOS_SIN), (_R_ARM_QPOS, _R_ARM_QPOS_SIN)):
        q = rng.uniform(-1.5, 1.5, size=(n, 7)).astype(np.float32)
        state[:, qsl] = q
        state[:, ssl] = np.sin(q)
    state[:, _L_GRIP_QPOS] = rng.uniform(0.0, 0.05, size=(n, 2))  # finger travel [0, 0.05]
    state[:, _R_GRIP_QPOS] = rng.uniform(0.0, 0.05, size=(n, 2))
    state[:, _TRUNK_QPOS] = rng.uniform(-0.4, 0.4, size=(n, 4))
    state[:, _BASE_QVEL] = rng.uniform(-0.2, 0.2, size=(n, 3))  # world-frame base vel
    state[:, _BASE_YAW] = rng.uniform(-np.pi, np.pi, size=n)
    return state


def _make_action(rng: np.random.RandomState, n: int) -> np.ndarray:
    """``(n, 23)`` action: base velocity at [0:3], binary {-1,+1} grippers."""
    action = rng.uniform(-1, 1, size=(n, ACTION_DIM)).astype(np.float32)
    action[:, _ACT_BASE] = rng.uniform(-0.3, 0.3, size=(n, 3))  # base vel
    action[:, _ACT_TRUNK] = rng.uniform(-0.4, 0.4, size=(n, 4))  # torso joints
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


def _write_info(bucket: Path, *, splits: dict | None = None) -> None:
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
    if splits is not None:
        info["splits"] = splits
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
        assert (a[:, 75:80] == 0).all()  # reserved tail (trunk now fills 71:75)
        assert np.isfinite(a).all()

    def test_base_velocity_present(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            s = _make_ds(b)[0]
        base = s["action"].numpy()[:, 68:71]
        assert np.isfinite(base).all()
        assert np.abs(base).sum() > 0  # base velocity actually written

    def test_trunk_present(self, tmp_path):
        # Native action[3:7] (4 torso joints) is scattered into reserved [71:75).
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            s = _make_ds(b)[0]
        trunk = s["action"].numpy()[:, 71:75]
        assert np.isfinite(trunk).all()
        assert np.abs(trunk).sum() > 0  # trunk joints actually written

    def test_multiview_canvas_size(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            s = _make_ds(b)[0]
        assert len(s["video"]) == 9  # (33-1)//4 + 1
        assert s["video"][0].size == (320, 384)  # PIL (W, H)

    def test_end_of_episode_single_frame_window(self, tmp_path):
        # Last start of ep 0 → a 1-frame window (actual_raw_len==1): the +1 EEF shift
        # has no next frame, so there are ZERO supervised action steps (the clamped
        # target must NOT be marked valid). Proprio (current frame) is still valid.
        b = make_behavior_bucket(tmp_path, n_episodes=1)  # ep 0 yields EP_LENGTH starts
        with _mock_video_decoder():
            s = _make_ds(b)[EP_LENGTH - 1]
        a = s["action"].numpy()
        assert a.shape == (32, 80)
        assert np.isfinite(a).all()
        assert not s["action_mask"].numpy().any()  # no real next-frame target exists
        assert s["proprio_mask"].numpy()[0, EXPECTED_VALID].all()

    def test_end_of_episode_two_frame_window_masks_clamped_step(self, tmp_path):
        # 2-frame boundary window (actual_raw_len==2): step 0 is a real transition
        # (eef target = next frame), step 1's target is clamped → must be masked.
        b = make_behavior_bucket(tmp_path, n_episodes=1)
        with _mock_video_decoder():
            s = _make_ds(b)[EP_LENGTH - 2]
        am = s["action_mask"].numpy()
        assert am[0, EXPECTED_VALID].all()  # first transition is a real target
        assert not am[1:].any()  # clamped final step is masked out


# ── single-view (ego-only) vs multiview canvas --------------------------------


class TestSingleView:
    """multiview=false → a single ego (head) frame at exactly (height, width); the
    spec's single-view mode is 256x320. multiview=true is locked to the 384x320
    L-shape canvas (the reader rejects any other multiview size)."""

    def test_single_view_256x320_ego(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            ds = BehaviorDataset(
                dataset_dir=str(b),
                height=256,
                width=320,
                multiview=False,
                unify_action=True,
                unify_action_map=UNIFY_MAP,
                normalize_mode=None,
            )
            s = ds[0]
        assert ds._multiview is False
        # plain ego frames, no L-shape canvas; PIL size is (W, H) = (320, 256)
        assert len(s["video"]) == 9  # (33-1)//4 + 1, same window as multiview
        assert all(f.size == (320, 256) for f in s["video"])

    def test_single_view_from_config_256x320(self, tmp_path):
        # The yaml single-view recipe (multiview:false + 256x320) threads through.
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        cfg = {
            "type": "behavior",
            "dataset_dir": str(b),
            "multiview": False,
            "height": 256,
            "width": 320,
            "normalize_mode": None,
            "unify_action": True,
            "unify_action_map": UNIFY_MAP,
        }
        with _mock_video_decoder():
            ds = BehaviorDataset.from_config(cfg, split="train")
            s = ds[0]
        assert ds._multiview is False
        assert s["video"][0].size == (320, 256)

    def test_multiview_wrong_size_rejected(self, tmp_path):
        # multiview is the fixed 384x320 L-shape canvas — any other size fails fast
        # (a non-standard canvas would silently break mixture collation).
        b = make_behavior_bucket(tmp_path, n_episodes=1)
        with _mock_video_decoder(), pytest.raises(ValueError, match="multiview mode requires height"):
            BehaviorDataset(
                dataset_dir=str(b),
                height=256,
                width=320,
                multiview=True,
                unify_action=True,
                unify_action_map=UNIFY_MAP,
                normalize_mode=None,
            )

    def test_multiview_default_size_ok(self, tmp_path):
        # The 384x320 multiview default is unaffected by the size guard.
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            s = _make_ds(b)[0]
        assert s["video"][0].size == (320, 384)


# ── prompts -------------------------------------------------------------------


class TestPrompt:
    def test_prompt_from_episodes_jsonl(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            ds = _make_ds(b)
            assert ds[0]["prompt"] == "do task 0"
            # ep 0 yields EP_LENGTH windows (train min len 1) → idx EP_LENGTH starts ep 1
            assert ds[EP_LENGTH]["prompt"] == "do task 1"

    def test_empty_prompt_raises(self, tmp_path):
        # A served episode with blank `tasks` must fail fast (episode_annotated
        # resolver does not guard emptiness on its own).
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with open(b / "meta" / "episodes.jsonl", "w") as f:
            f.write(json.dumps({"episode_index": 0, "length": EP_LENGTH, "tasks": []}) + "\n")
            f.write(json.dumps({"episode_index": 1, "length": EP_LENGTH, "tasks": ["do task 1"]}) + "\n")
        with _mock_video_decoder(), pytest.raises(ValueError, match="empty 'tasks' prompt"):
            _make_ds(b)


# ── split semantics -----------------------------------------------------------


class TestSplit:
    def test_val_split_is_empty_no_leak(self, tmp_path):
        # info.json declares no val split → train serves all episodes, val is empty
        # (must NOT silently leak the training set into a val loader).
        b = make_behavior_bucket(tmp_path, n_episodes=3)
        with _mock_video_decoder():
            train = _make_ds(b, split="train")
            val = _make_ds(b, split="val")
        assert len(train) > 0
        assert len(val) == 0

    def test_declared_train_split_keeps_all_noncontiguous_episodes(self, tmp_path):
        # Mirror the REAL dataset: info.json declares splits.train="0:N" (a positional
        # count) while episode_index is non-contiguous (task*chunks_size + local). The
        # train split must keep ALL on-disk episodes — NOT range-filter by
        # episode_index value (which would drop ~every task but task-0000).
        bucket = tmp_path / "behaviour-1k"
        bucket.mkdir(parents=True, exist_ok=True)
        idxs = [10, CHUNKS_SIZE + 10, 2 * CHUNKS_SIZE + 10]  # tasks 0, 1, 2 (local 10)
        _write_info(bucket, splits={"train": f"0:{len(idxs)}"})
        _write_episodes_jsonl(bucket, idxs)
        for ep in idxs:
            _write_episode_parquet(bucket, ep)
            _write_episode_videos(bucket, ep)
        with _mock_video_decoder():
            ds = _make_ds(bucket, split="train")
        assert sorted(ds._eps_df["episode_index"].tolist()) == idxs  # all 3 tasks kept
        assert ds._eps_df["data/chunk_index"].tolist() == [0, 1, 2]  # chunk = idx // chunks_size


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


# ── deploy denormalizer artifact ---------------------------------------------


class TestDeployNormalizer:
    """The reader emits meta/normalization_stats.npy in RAW-27 action space. At
    deploy the policy server wraps the RAW Normalizer in _UnifyAwareNormalizer
    (PR #17): it gathers the model's 80-D unified output back to the 27 raw dims,
    THEN unnormalizes — so the artifact is authored in RAW space, not scattered."""

    def test_quantile_registered_in_deploy_mode_map(self):
        # Deploy must recognize the reader-family default 'quantile' (→ q99), else
        # _build_normalizer silently disables the normalizer on a real checkpoint.
        from openwam.dataloader.transforms.normalize import YAML_TO_NORM_MODE

        assert YAML_TO_NORM_MODE.get("quantile") == "q99"

    def test_stats_npy_written_and_schema(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        with _mock_video_decoder():
            ds = _make_ds(b, normalize_mode="quantile")
        # path exposed for the trainer's save_normalization_stats() copy
        assert ds.normalization_stats_path is not None
        assert Path(ds.normalization_stats_path).exists()
        raw = np.load(ds.normalization_stats_path, allow_pickle=True).item()
        assert set(raw) == {"unified"}  # action_mode key from behavior.yaml
        stats = raw["unified"]
        # RAW-27 stats (eef20 + base3 + trunk4); the deploy _UnifyAwareNormalizer
        # gathers the model's 80-D output back to these 27 raw dims first.
        for k in ("mean", "std", "min", "max", "q01", "q99"):
            assert stats[k].shape == (27,)
        # rot6d raw dims (3:9 / 13:19) pinned to identity in the stats file.
        for i in (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18):
            assert stats["q01"][i] == -1.0 and stats["q99"][i] == 1.0
            assert stats["mean"][i] == 0.0 and stats["std"][i] == 1.0

    def test_deploy_unify_normalizer_matches_reader(self, tmp_path):
        # The deploy _UnifyAwareNormalizer built from the RAW .npy + the unify map
        # reproduces the reader's own normalization (train ↔ deploy consistency):
        #   proprio IN : normalize raw → scatter raw→80 == the reader's normalized 80-D
        #   action OUT : gather 80→raw → unnormalize == the reader's raw physical action
        from openwam.dataloader.transforms.normalize import Normalizer, load_mode_stats
        from openwam.dataloader.utils.unify_action import parse_unify_spec, unmap_from_unify
        from openwam.deploy.model_loader import _UnifyAwareNormalizer

        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        with _mock_video_decoder():
            raw80 = _make_ds(b, normalize_mode=None)[0]["action"].numpy()  # un-normalized 80-D
            ds = _make_ds(b, normalize_mode="quantile")
            norm80 = ds[0]["action"].numpy()  # reader-normalized 80-D
        mode_stats = load_mode_stats(ds.normalization_stats_path, "unified")  # RAW-27
        assert mode_stats["mean"].shape == (27,)
        dst_index = parse_unify_spec(UNIFY_MAP, UNIFY_DIM)
        uan = _UnifyAwareNormalizer(Normalizer(mode="q99", stats=mode_stats), dst_index, UNIFY_DIM)

        raw27 = unmap_from_unify(raw80, dst_index)  # (T, 27) un-normalized raw
        # proprio IN: scatter-normalize(raw27) reproduces the reader's normalized 80-D action.
        np.testing.assert_allclose(uan.normalize(raw27), norm80, atol=1e-5)
        # action OUT: gather-unnormalize(norm80) inverts back to raw on non-clipped dims.
        recovered27 = uan.unnormalize(norm80)
        norm27 = unmap_from_unify(norm80, dst_index)
        inside = np.abs(norm27) < 1.0 - 1e-3  # exclude quantile-clipped entries
        np.testing.assert_allclose(recovered27[inside], raw27[inside], atol=1e-4)

    def test_stats_merge_preserves_other_mode(self, tmp_path):
        # Two action_modes sharing ONE bucket must COEXIST in normalization_stats.npy.
        # Constructing joint after unified must not destroy the 'unified' key: the write
        # is a read-modify-write merge, not a blind overwrite. (A blind overwrite would
        # drop 'unified'; deploy then hard-fails on the missing key → refused deploy.)
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        with _mock_video_decoder():
            uds = _make_ds(b, normalize_mode="quantile")  # writes {'unified'}
            after_unified = set(np.load(uds.normalization_stats_path, allow_pickle=True).item())
            jds = _make_joint_ds(b, normalize_mode="quantile")  # merges in {'joint'}
        assert after_unified == {"unified"}
        raw = np.load(jds.normalization_stats_path, allow_pickle=True).item()
        assert set(raw) == {"unified", "joint"}  # both modes survive
        assert raw["unified"]["mean"].shape == (27,)  # eef20 + base3 + trunk4
        assert raw["joint"]["mean"].shape == (23,)  # arm16 + base3 + trunk4

    def test_readonly_mount_skips_deploy_stats_write(self, tmp_path):
        # A read-only dataset mount (np.save → OSError) must degrade to "no deploy
        # artifact" + warning, NOT crash __init__. In-process normalization still loads,
        # so training on a RO mount works; only the deploy artifact is skipped.
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        with (
            _mock_video_decoder(),
            patch(
                "openwam.dataloader.bases.lerobot_v3_reader.np.save",
                side_effect=OSError("read-only file system"),
            ),
        ):
            ds = _make_ds(b, normalize_mode="quantile")  # must NOT raise
            assert ds.normalization_stats_path is None  # artifact skipped
            assert ds._normalization_stats is not None  # in-process stats still built
            s = ds[0]
        assert s["action"].shape == (32, 80)

    def test_transient_read_error_preserves_existing_modes(self, tmp_path):
        # A transient OSError on the merge-READ (NFS/fuseblk blip) must NOT be mistaken
        # for corruption: the write is skipped and the existing file is left intact,
        # rather than clobbering a valid other-mode key with a single-key rewrite.
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        stats_npy = b / "meta" / "normalization_stats.npy"
        with _mock_video_decoder():
            uds = _make_ds(b, normalize_mode="quantile")  # writes {'unified'}
            assert set(np.load(uds.normalization_stats_path, allow_pickle=True).item()) == {"unified"}
            real_load = np.load

            def _load_blip(path, *a, **k):
                if "normalization_stats.npy" in str(path):
                    raise OSError("transient nfs read")
                return real_load(path, *a, **k)

            with patch("openwam.dataloader.bases.lerobot_v3_reader.np.load", side_effect=_load_blip):
                jds = _make_joint_ds(b, normalize_mode="quantile")  # read blip → skip write
            assert jds.normalization_stats_path is None  # write skipped, not clobbered
        # existing file untouched: 'unified' survives (NOT rewritten to just {'joint'}).
        raw = np.load(str(stats_npy), allow_pickle=True).item()
        assert set(raw) == {"unified"}


# ── color jitter (train-split video augmentation) -----------------------------


class TestColorJitter:
    """Load-time color jitter is built by the shared base reader from the yaml
    `color_jitter` switch — enabled on train only, never on val/eval."""

    _CJ = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.0}

    def test_enabled_on_train_split(self, tmp_path):
        from openwam.dataloader.transforms.video import VideoColorJitter

        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            ds = _make_ds(b, color_jitter=self._CJ, split="train")
            # jitter does not change the served clip's shape (it runs in __getitem__)
            s = ds[0]
        assert isinstance(ds._color_jitter, VideoColorJitter)
        assert (
            ds._color_jitter.brightness,
            ds._color_jitter.contrast,
            ds._color_jitter.saturation,
            ds._color_jitter.hue,
        ) == (0.2, 0.2, 0.2, 0.0)
        assert len(s["video"]) == 9
        assert s["video"][0].size == (320, 384)

    def test_disabled_on_val_split(self, tmp_path):
        # Augmentation must never touch val/eval video (deterministic eval).
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            ds = _make_ds(b, color_jitter=self._CJ, split="val")
        assert ds._color_jitter is None

    def test_disabled_when_absent(self, tmp_path):
        # Omitted / null switch → no jitter (video byte-identical to before).
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            ds_absent = _make_ds(b)
            ds_null = _make_ds(b, color_jitter=None)
            ds_false = _make_ds(b, color_jitter=False)
        assert ds_absent._color_jitter is None
        assert ds_null._color_jitter is None
        assert ds_false._color_jitter is None

    def test_from_config_threads_color_jitter(self, tmp_path):
        # The yaml `color_jitter:` block reaches the reader through from_config.
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        cfg = {
            "type": "behavior",
            "dataset_dir": str(b),
            "multiview": True,
            "height": 384,
            "width": 320,
            "normalize_mode": None,
            "unify_action": True,
            "unify_action_map": UNIFY_MAP,
            "color_jitter": self._CJ,
        }
        with _mock_video_decoder():
            ds = BehaviorDataset.from_config(cfg, split="train")
        assert ds._color_jitter is not None
        assert ds._color_jitter.brightness == 0.2


# ── stats-computation script --------------------------------------------------


class TestStatsScript:
    def test_schema_and_rot6d_pin(self, tmp_path):
        from openwam.dataloader.utils.stats_computation.behavior_stats_computation import compute_behavior_stats

        b = make_behavior_bucket(tmp_path, n_episodes=3)
        result = compute_behavior_stats(b)
        assert set(result) == {"eef", "base_vel", "trunk", "arm_joint"}
        eef, base, trunk, arm = result["eef"], result["base_vel"], result["trunk"], result["arm_joint"]
        for k in ("mean", "std", "min", "max", "q01", "q99"):
            assert len(eef[k]) == 20
            assert len(base[k]) == 3
            assert len(trunk[k]) == 4
            assert len(arm[k]) == _ARM_JOINT_DIM == 16
        # rot6d dims (3:9 / 13:19) pinned to identity
        assert eef["rot6d_identity"] is True
        for i in (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18):
            assert eef["min"][i] == -1.0 and eef["max"][i] == 1.0
            assert eef["q01"][i] == -1.0 and eef["q99"][i] == 1.0
            assert eef["mean"][i] == 0.0 and eef["std"][i] == 1.0
        # base velocity is NOT pinned (real stats from data)
        assert "rot6d_identity" not in base
        assert base["layout"] == "vx,vy,vyaw"
        # trunk (torso joints) is NOT pinned either
        assert "rot6d_identity" not in trunk
        assert trunk["layout"] == "torso_joint_abs"
        # arm_joint block: real stats, NOT pinned (no rot6d in joint mode)
        assert "rot6d_identity" not in arm
        assert arm["layout"] == "L_arm7,L_grip1,R_arm7,R_grip1"
        # every block pools the ACTION + PROPRIO streams (== RoboCOIN), so the
        # stats cover the proprio marginals (gripper open-scale, base-frame vel),
        # not just the action command's.
        for blk in (eef, base, trunk, arm):
            assert blk["pool"] == "action+proprio"

    def test_stats_pool_both_action_and_proprio_streams(self, tmp_path):
        # The gripper stats must reflect the pooled action+proprio marginal: pooling
        # the achieved open-scale changes mean/std vs an action-only computation
        # (identical only if we forgot to pool). Guards the fix from silently
        # regressing back to action-only stats.
        import numpy as _np

        from openwam.dataloader.utils.stats_computation.behavior_stats_computation import (
            _rows_to_blocks,
            compute_behavior_stats,
        )
        from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator

        b = make_behavior_bucket(tmp_path, n_episodes=3)
        pooled = compute_behavior_stats(b)["eef"]
        # Recompute an ACTION-ONLY eef accumulator over the same episodes.
        import pyarrow.parquet as _pq

        action_only = Accumulator(dim=20)
        for f in sorted((b / "data").glob("task-*/episode_*.parquet")):
            df = _pq.read_table(f, columns=["observation.state", "action"]).to_pandas()
            st = _np.stack(df["observation.state"].values).astype(_np.float32)
            ac = _np.stack(df["action"].values).astype(_np.float32)
            action_only.update_batch(_rows_to_blocks(st, ac)[0])
        ao = action_only.finalize()
        # pooled count is exactly 2x the action-only count (action + proprio rows).
        assert pooled["num_timesteps"] == 2 * action_only.count
        # gripper mean (slots 9/19) differs once the proprio open-scale is pooled in.
        assert abs(pooled["mean"][9] - ao["mean"][9]) > 1e-6 or abs(pooled["mean"][19] - ao["mean"][19]) > 1e-6

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


# ── action modes: joint / eef / unified --------------------------------------


def _make_joint_ds(bucket: Path, **kw):
    """Joint-mode reader: action_mode=joint forces unify off (incompatible)."""
    kw.setdefault("multiview", True)
    kw.setdefault("normalize_mode", None)
    kw["action_mode"] = "joint"
    kw["unify_action"] = False
    return BehaviorDataset(dataset_dir=str(bucket), height=384, width=320, **kw)


class TestActionModes:
    """BEHAVIOR's three action modes: unified (default, EEF→80-D), eef (raw 27),
    joint (raw 23 from the native action[23] JointController setpoints)."""

    def test_joint_shapes_and_all_visible_mask(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            ds = _make_joint_ds(b)
            s = ds[0]
        assert ds.action_dim == _JOINT_DIM == 23
        assert s["action"].shape == (32, 23)
        assert s["proprio"].shape == (1, 23)
        # joint/eef rule: NO mask — every dim visible on valid steps.
        am = s["action_mask"].numpy()
        assert am.shape == (32, 23)
        assert am.all()  # all 23 dims valid on all 32 (< EP_LENGTH) steps
        assert s["proprio_mask"].numpy().all()

    def test_joint_proprio_is_rendered_achieved_state(self, tmp_path):
        # raw-23 proprio (t=0) = achieved state rendered by _state_to_raw_proprio_joint
        # [L_arm_qpos7, L_grip_openscale1, R_arm_qpos7, R_grip_openscale1, base3, trunk4]
        # — NOT the action command. Replay the writer rng, render, compare exactly.
        b = make_behavior_bucket(tmp_path, n_episodes=1)
        with _mock_video_decoder():
            p = _make_joint_ds(b)[0]["proprio"].numpy()[0]  # (23,)
        rng = np.random.RandomState(100)  # ep 0 writer seed
        state0 = _make_state(rng, EP_LENGTH)[:1]  # (1, 256)
        expected = _state_to_raw_proprio_joint(state0)[0]
        np.testing.assert_allclose(p, expected, rtol=0, atol=1e-6)
        # gripper open-scale lands at slots 7 / 15, continuous in [-1, +1] (NOT the
        # binary ±1 command — it is the achieved finger opening).
        assert -1.0 <= p[7] <= 1.0 and -1.0 <= p[15] <= 1.0

    def test_joint_action_is_row_aligned_and_differs_from_proprio(self, tmp_path):
        # Joint ACTION reads the native command at t (no +1 shift): action[0] == the
        # reordered native row 0. PROPRIO is the achieved state, a DIFFERENT field —
        # so the two must NOT be equal (that mismatch is the whole point of the fix).
        b = make_behavior_bucket(tmp_path, n_episodes=1)
        with _mock_video_decoder():
            s = _make_joint_ds(b)[0]
        a0 = s["action"].numpy()[0]
        rng = np.random.RandomState(100)
        _make_state(rng, EP_LENGTH)
        act0 = _make_action(rng, EP_LENGTH)[0]  # native (23,)
        expected_a0 = np.concatenate(
            [
                act0[_ACT_LARM],
                act0[_ACT_LGRIP : _ACT_LGRIP + 1],
                act0[_ACT_RARM],
                act0[_ACT_RGRIP : _ACT_RGRIP + 1],
                act0[_ACT_BASE],
                act0[_ACT_TRUNK],
            ]
        ).astype(np.float32)
        np.testing.assert_allclose(a0, expected_a0, rtol=0, atol=1e-6)  # action row-aligned
        assert not np.allclose(a0, s["proprio"].numpy()[0])  # proprio != command

    def test_joint_proprio_arm_from_state_qpos_not_action(self, tmp_path):
        # The joint arm proprio must come from the ACHIEVED arm qpos in
        # observation.state ([158:165]/[197:204]), NOT the action[7:14]/[15:22] columns.
        b = make_behavior_bucket(tmp_path, n_episodes=1)
        with _mock_video_decoder():
            p = _make_joint_ds(b)[0]["proprio"].numpy()[0]
        rng = np.random.RandomState(100)
        state0 = _make_state(rng, EP_LENGTH)[0]  # (256,)
        act0 = _make_action(rng, EP_LENGTH)[0]
        np.testing.assert_allclose(p[0:7], state0[_L_ARM_QPOS], rtol=0, atol=1e-6)  # from state
        np.testing.assert_allclose(p[8:15], state0[_R_ARM_QPOS], rtol=0, atol=1e-6)
        assert not np.allclose(p[0:7], act0[_ACT_LARM])  # and NOT the action columns

    def test_joint_plus_unify_rejected(self, tmp_path):
        # joint requires unify_action=false (the 80-D space is EEF-semantic).
        b = make_behavior_bucket(tmp_path, n_episodes=1)
        with _mock_video_decoder(), pytest.raises(ValueError, match="requires unify_action=false"):
            BehaviorDataset(
                dataset_dir=str(b),
                height=384,
                width=320,
                action_mode="joint",
                unify_action=True,
                unify_action_map=UNIFY_MAP,
            )

    def test_eef_plus_unify_rejected(self, tmp_path):
        # eef + unify_action=true would silently use the unified 80-D path while the
        # deploy stats key says 'eef' → the mode contract rejects it.
        b = make_behavior_bucket(tmp_path, n_episodes=1)
        with _mock_video_decoder(), pytest.raises(ValueError, match="requires unify_action=false"):
            BehaviorDataset(
                dataset_dir=str(b),
                height=384,
                width=320,
                action_mode="eef",
                unify_action=True,
                unify_action_map=UNIFY_MAP,
            )

    def test_unified_without_unify_rejected(self, tmp_path):
        # unified must pair with unify_action=true (else it is just raw eef).
        b = make_behavior_bucket(tmp_path, n_episodes=1)
        with _mock_video_decoder(), pytest.raises(ValueError, match="requires unify_action=true"):
            BehaviorDataset(dataset_dir=str(b), height=384, width=320, action_mode="unified", unify_action=False)

    def test_invalid_action_mode_rejected(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=1)
        with _mock_video_decoder(), pytest.raises(ValueError, match="action_mode must be one of"):
            BehaviorDataset(dataset_dir=str(b), height=384, width=320, action_mode="bogus")

    def test_offcase_action_mode_rejected(self, tmp_path):
        # Case-sensitive: 'Joint' must NOT silently become 'joint' (deploy looks up
        # the exact saved string → off-case would disable normalization).
        b = make_behavior_bucket(tmp_path, n_episodes=1)
        with _mock_video_decoder(), pytest.raises(ValueError, match="action_mode must be one of"):
            BehaviorDataset(dataset_dir=str(b), height=384, width=320, action_mode="Joint", unify_action=False)

    def test_eef_mode_raw27_no_mask(self, tmp_path):
        # action_mode=eef, unify off → raw 27-D EEF, all dims visible.
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            ds = BehaviorDataset(
                dataset_dir=str(b),
                height=384,
                width=320,
                action_mode="eef",
                unify_action=False,
                normalize_mode=None,
                multiview=True,
            )
            s = ds[0]
        assert ds.action_dim == 27
        assert s["action"].shape == (32, 27)
        assert s["proprio"].shape == (1, 27)
        assert s["action_mask"].numpy().all()  # no mask

    def test_eef_proprio_is_rendered_achieved_state(self, tmp_path):
        # eef/unified proprio (t=0) = _state_to_raw_proprio_eef(state[0]): achieved
        # eef pose + gripper open-scale + base-frame velocity + trunk qpos — NOT the
        # action command. Compare value-for-value (normalize off, raw 27).
        b = make_behavior_bucket(tmp_path, n_episodes=1)
        with _mock_video_decoder():
            ds = BehaviorDataset(
                dataset_dir=str(b),
                height=384,
                width=320,
                action_mode="eef",
                unify_action=False,
                normalize_mode=None,
                multiview=True,
            )
            s = ds[0]
        p = s["proprio"].numpy()[0]  # (27,)
        rng = np.random.RandomState(100)  # ep 0 writer seed
        state0 = _make_state(rng, EP_LENGTH)[:1]
        expected = _state_to_raw_proprio_eef(state0)[0]
        np.testing.assert_allclose(p, expected, rtol=0, atol=1e-6)
        # gripper open-scale (slots 9/19) continuous in [-1,1]; base (20:23) finite.
        assert -1.0 <= p[9] <= 1.0 and -1.0 <= p[19] <= 1.0
        assert np.isfinite(p[20:23]).all()
        # proprio EEF pose comes from state[0] (achieved), NOT the +1-shifted action.
        act = s["action"].numpy()  # eef action arm target = eef(state[1])
        assert not np.allclose(p[0:3], act[0, 0:3])  # proprio L_pos != action L_pos target

    def test_unified_default_unchanged(self, tmp_path):
        # Regression: action_mode=unified (the default) is byte-identical to before
        # — EEF scattered into 80-D with the mapped-only mask.
        b = make_behavior_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder():
            ds = _make_ds(b)  # unify_action=True, action_mode defaults to "unified"
            s = ds[0]
        assert ds.action_dim == UNIFY_DIM == 80
        assert ds._action_mode == "eef"  # representation under unified IS eef
        assert ds.DEPLOY_ACTION_MODE == "unified"
        assert sorted(np.where(s["action_mask"].numpy()[0])[0].tolist()) == EXPECTED_VALID

    def test_joint_normalization_loads_arm_joint_block(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        with _mock_video_decoder():
            ds = _make_joint_ds(b, normalize_mode="quantile")
            s = ds[0]
        assert ds._normalization_stats is not None
        # combined joint stats are 23-D (arm_joint16 + base3 + trunk4)
        assert ds._normalization_stats["mean"].shape[0] == 23
        a = s["action"].numpy()
        assert np.isfinite(a).all()
        assert (np.abs(a) <= 1.0 + 1e-5).all()  # quantile clips all (no mask) dims

    def test_joint_deploy_stats_key_is_joint(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        with _mock_video_decoder():
            ds = _make_joint_ds(b, normalize_mode="quantile")
        raw = np.load(ds.normalization_stats_path, allow_pickle=True).item()
        assert set(raw) == {"joint"}  # DEPLOY_ACTION_MODE == action_mode
        assert len(raw["joint"]["mean"]) == 23

    def test_joint_missing_arm_joint_block_raises(self, tmp_path):
        # An eef-era stats file (no arm_joint block) must fail fast in joint mode.
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        stats_path = b / "meta" / "stats_R1Pro.json"
        stats = json.loads(stats_path.read_text())
        del stats["arm_joint"]
        stats_path.write_text(json.dumps(stats))
        with _mock_video_decoder(), pytest.raises(KeyError, match="arm_joint"):
            _make_joint_ds(b, normalize_mode="quantile")


# ── min-max default (benchmark convention) ------------------------------------


class TestMinMaxDefault:
    """BEHAVIOR is a closed-loop scored benchmark: its default flipped from the
    robocoin-family quantile to min-max so training targets are not saturated
    at [q01, q99] (base velocity dims lose a substantial fraction of the demos' top speed:
    q99 is substantially below max). These tests pin the new default and the
    deploy-side min_max normalize being training-identical (clipped,
    constant-dim safe)."""

    def test_default_normalize_mode_is_min_max(self, tmp_path):
        from openwam.dataloader.behavior import BehaviorDataset

        assert BehaviorDataset.DEFAULT_NORMALIZE_MODE == "min-max"
        # _make_ds force-passes normalize_mode=None (explicit null = disable),
        # so construct directly to exercise the UNSET → class-default path.
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        with _mock_video_decoder():
            ds = BehaviorDataset(
                dataset_dir=str(b),
                height=384,
                width=320,
                multiview=True,
                unify_action=True,
                unify_action_map=UNIFY_MAP,
            )
        assert ds._normalize_mode == "min-max"

    def test_min_max_registered_in_deploy_mode_map(self):
        from openwam.dataloader.transforms.normalize import YAML_TO_NORM_MODE

        assert YAML_TO_NORM_MODE.get("min-max") == "min_max"

    def test_min_max_bounded_and_rot6d_passthrough(self, tmp_path):
        b = make_behavior_bucket(tmp_path, n_episodes=2, with_stats=True)
        with _mock_video_decoder():
            raw = _make_ds(b, normalize_mode=None)[0]["action"].numpy()
            norm_ds = _make_ds(b, normalize_mode="min-max")
            norm = norm_ds[0]["action"].numpy()
        assert (np.abs(norm[:, EXPECTED_VALID]) <= 1.0 + 1e-5).all()
        # rot6d stats stay identity-pinned → passthrough under min-max too
        for sl in (slice(3, 9), slice(37, 43)):
            np.testing.assert_allclose(norm[:, sl], raw[:, sl], atol=1e-5)

    def test_deploy_min_max_normalize_matches_training_and_clips(self):
        """Deploy Normalizer(min_max).normalize must be bit-parallel to the
        training-side apply_normalization — CLIPPED (training clips min-max;
        an unclipped deploy would hand the model proprio thousands of sigma
        out-of-distribution on degenerate near-constant dims where
        scale=2/eps) and exact on large constant dims (float32 offset
        absorption previously yielded -1.907/0.0 instead of -1)."""
        from openwam.dataloader.transforms.normalize import Normalizer
        from openwam.dataloader.utils.normalization import apply_normalization

        lo = np.array([0.0, 0.044, 12.0, 90.0, 0.1], np.float32)
        hi = np.array([0.0, 0.044, 12.0, 90.0, 0.1002], np.float32)
        deploy = Normalizer(mode="min_max", stats={"min": lo, "max": hi})
        stats = {"min": lo, "max": hi, "mean": lo, "std": np.ones(5, np.float32)}
        probes = [lo, hi, lo + 0.015, np.array([1e-4, 0.02, 11.5, 90.006, 0.2], np.float32)]
        for x in probes:
            train = apply_normalization(x[None, :].astype(np.float32), stats, "min-max")[0]
            got = deploy.normalize(x.astype(np.float32))
            np.testing.assert_allclose(got, train, atol=1e-5)
            assert np.abs(got).max() <= 1.0 + 1e-6  # bounded like training
        # unnormalize still recovers the constants exactly
        rec = deploy.unnormalize(np.array([-1.0, -1.0, -1.0, -1.0, 0.3], np.float32))
        np.testing.assert_allclose(rec[:4], lo[:4], atol=1e-4)

    def test_deploy_q99_normalize_matches_training_on_constant_dims(self):
        """Same parity for the q99 path (BEHAVIOR/robocoin quantile ckpts):
        degenerate constant dims |c|>=16 previously normalized to 0.0 at
        deploy while training saw exactly -1 (8<=|c|<16 happened to be
        rescued by the old branch's clip)."""
        from openwam.dataloader.transforms.normalize import Normalizer
        from openwam.dataloader.utils.normalization import apply_normalization

        q = np.array([0.0, 12.0, 90.0, 0.1], np.float32)  # q01 == q99 == c
        deploy = Normalizer(mode="q99", stats={"q01": q, "q99": q + np.array([0, 0, 0, 0.2], np.float32)})
        stats = {"q01": q, "q99": q + np.array([0, 0, 0, 0.2], np.float32)}
        for x in (q, q + 0.05):
            train = apply_normalization(x[None, :].astype(np.float32), stats, "quantile")[0]
            np.testing.assert_allclose(deploy.normalize(x.astype(np.float32)), train, atol=1e-5)
