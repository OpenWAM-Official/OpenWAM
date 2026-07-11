"""Unit tests for the bespoke RoboCasa365 dataloader (raw LeRobot v3.0).

Builds a minimal RoboCasa365-shaped **v3.0 aggregated** repo on disk
(``meta/episodes/chunk-000/file-000.parquet`` with per-episode offset metadata +
``source_prefix`` task tag + ``tasks`` strings, one aggregated
``data/chunk-000/file-000.parquet`` with all episodes' rows concatenated, and
aggregated ``videos/<cam>/chunk-000/file-000.mp4``) and exercises the readers.
Video decoding is mocked, so no real mp4s are needed.

RoboCasa365 is single-arm: action & proprio are the absolute EEF pose derived from
``observation.state`` and slotted into the LEFT 10 dims of the canonical 20-D EEF
(right 10 zero-padded + masked), like the OXE single-arm readers.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import openwam.dataloader.robocasa365 as rc
from openwam.dataloader.robocasa365 import MultiTaskRoboCasa365Dataset, RoboCasa365Dataset
from openwam.dataloader.utils.eef import EEF_DIM

EP_LENGTH = 60
N_EPISODES = 3
HEAD_CAM = "observation.images.robot0_agentview_left"
WRIST_CAM = "observation.images.robot0_eye_in_hand"
PROMPT = "Open the right drawer."


def _make_state(n_rows: int, seed: int) -> np.ndarray:
    """RoboCasa365-shaped observation.state[16]: base_pos(0:3)+base_rot(3:7)+
    eef_pos_rel(7:10)+eef_rot_rel(10:14, unit quat xyzw)+gripper_qpos(14:16)."""
    rng = np.random.RandomState(seed)
    state = np.zeros((n_rows, 16), dtype=np.float64)
    # base: a drifting planar (x, y) world position + a valid yaw quaternion (base_rotation xyzw =
    # [0, 0, sin(yaw/2), cos(yaw/2)]), so the base actually moves (exercises base-velocity proprio).
    state[:, 0:2] = np.cumsum(rng.uniform(-0.05, 0.05, size=(n_rows, 2)), axis=0)
    yaw = np.cumsum(rng.uniform(-0.1, 0.1, size=n_rows))
    state[:, 5] = np.sin(yaw / 2.0)
    state[:, 6] = np.cos(yaw / 2.0)
    state[:, 7:10] = rng.uniform(-1, 1, size=(n_rows, 3))  # eef_pos_rel
    q = rng.uniform(-1, 1, size=(n_rows, 4))
    state[:, 10:14] = q / np.linalg.norm(q, axis=1, keepdims=True)  # unit quat xyzw
    state[:, 14:16] = rng.uniform(0, 0.04, size=(n_rows, 2))  # gripper_qpos
    return state


def _make_action(n_rows: int, seed: int) -> np.ndarray:
    """RoboCasa365-shaped LeRobot action[12] (layout B): base_motion(0:4 = x/y/yaw vel + torso),
    control_mode(4), eef Δpos(5:8), eef Δrot(8:11), gripper(11). Only base_motion(0:4) + mode(4)
    (the mobile channel) are exercised; the arm dims are filled but the reader ignores them (arm
    comes from observation.state)."""
    rng = np.random.RandomState(seed + 1000)
    action = np.zeros((n_rows, 12), dtype=np.float64)
    action[:, 0:3] = rng.uniform(-1, 1, size=(n_rows, 3))  # base x/y/yaw velocity
    action[:, 3] = rng.uniform(0, 0.34, size=n_rows)  # torso lift (position)
    action[:, 4] = rng.choice([-1.0, 1.0], size=n_rows)  # control_mode
    action[:, 5:11] = rng.uniform(-1, 1, size=(n_rows, 6))  # arm OSC delta (ignored by reader)
    action[:, 11] = rng.choice([-1.0, 1.0], size=n_rows)  # gripper_close command (used as ACTION gripper)
    return action


def _write_v3_repo(root: Path, task_specs: list) -> Path:
    """Write a minimal LeRobot **v3.0** aggregated RoboCasa365 repo; return its root dir.

    ``task_specs`` = ``[(task_name, n_episodes), ...]`` — v3 packs several tasks into ONE repo,
    tagged per episode by ``source_prefix``. Layout: ``meta/episodes/chunk-000/file-000.parquet``
    (per-episode offset metadata), one aggregated ``data/chunk-000/file-000.parquet`` (all episodes'
    rows concatenated), aggregated ``videos/<cam>/chunk-000/file-000.mp4`` (empty; decoder mocked).
    All episodes share (chunk 0, file 0), so each episode's file-local row offset is the cumulative
    length of the episodes before it.
    """
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    total = sum(n for _, n in task_specs)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "PandaOmron",
        "chunks_size": 1000,
        "fps": 20,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "splits": {"train": f"0:{total}"},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))

    states, actions, task_idx, ep_rows, cum, epi = [], [], [], [], 0, 0
    for ti, (task_name, n_episodes) in enumerate(task_specs):
        for _ in range(n_episodes):
            states.extend(list(_make_state(EP_LENGTH, seed=epi)))
            actions.extend(list(_make_action(EP_LENGTH, seed=epi)))
            task_idx.extend([ti] * EP_LENGTH)
            row = {
                "episode_index": epi,
                "dataset_from_index": cum,
                "dataset_to_index": cum + EP_LENGTH,
                "length": EP_LENGTH,
                "tasks": [PROMPT],
                "data/chunk_index": 0,
                "data/file_index": 0,
                "source_prefix": f"pretrain/atomic/{task_name}/20250819",
                "source_episode_index": epi,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for cam in (HEAD_CAM, WRIST_CAM):
                row[f"videos/{cam}/chunk_index"] = 0
                row[f"videos/{cam}/file_index"] = 0
            ep_rows.append(row)
            cum += EP_LENGTH
            epi += 1

    pd.DataFrame({"observation.state": states, "action": actions, "task_index": task_idx}).to_parquet(
        root / "data" / "chunk-000" / "file-000.parquet"
    )
    pd.DataFrame(ep_rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    for cam in (HEAD_CAM, WRIST_CAM):
        vd = root / "videos" / cam / "chunk-000"
        vd.mkdir(parents=True, exist_ok=True)
        (vd / "file-000.mp4").write_bytes(b"")  # mocked decoder ignores content
    return root


def make_robocasa_bucket(tmp_path: Path, n_episodes: int = N_EPISODES, task_name: str = "OpenDrawer") -> Path:
    """A single-task v3 aggregated repo (one ``source_prefix``)."""
    return _write_v3_repo(tmp_path / "pretrain-atomic", [(task_name, n_episodes)])


def make_multitask_bucket(tmp_path: Path, tasks=("taskA", "taskB"), n_episodes: int = N_EPISODES) -> Path:
    """A v3 aggregated repo holding several tasks (distinct ``source_prefix`` tags)."""
    return _write_v3_repo(tmp_path / "pretrain-atomic", [(t, n_episodes) for t in tasks])


@contextmanager
def _mock_video_decoder():
    def _fake(video_path, frame_indices, height, width):
        return [Image.new("RGB", (width, height), (0, 0, 0)) for _ in frame_indices]

    with patch.object(rc, "decode_video_frames", side_effect=_fake):
        yield


class TestInit:
    def test_loads_bucket(self, tmp_path):
        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(data_root=str(b), task_name="OpenDrawer", normalize_mode=None)
        assert ds.action_dim == EEF_DIM
        assert ds.task_name == "OpenDrawer"
        assert len(ds._ep_pos) == N_EPISODES
        assert ds.camera_layout[0] == HEAD_CAM and ds.camera_layout[1] == WRIST_CAM
        assert len(ds) == N_EPISODES * (EP_LENGTH - 1)  # train max_start = ep_len-2, +1

    def test_bad_video_stride_raises(self, tmp_path):
        b = make_robocasa_bucket(tmp_path)
        with pytest.raises(ValueError, match="divisible by video_stride"):
            RoboCasa365Dataset(data_root=str(b), normalize_mode=None, num_frames=33, video_stride=5)

    def test_bad_resolution_raises(self, tmp_path):
        b = make_robocasa_bucket(tmp_path)
        with pytest.raises(ValueError, match="divisible by 32"):
            RoboCasa365Dataset(data_root=str(b), normalize_mode=None, multiview=False, height=100, width=96)

    def test_temporal_contract_not_enforced(self, tmp_path):
        # main/robotwin no longer enforce the encoder temporal contract at runtime (documented only);
        # a "mismatched" num_frames/video_stride must build without raising, not error out.
        # num_frames=9, video_stride=4 → 3 video frames (would fail the old causal (3-1)%4==0 guard).
        b = make_robocasa_bucket(tmp_path)
        ds = RoboCasa365Dataset(data_root=str(b), normalize_mode=None, multiview=False, height=64, width=96,
                                num_frames=9, video_stride=4)
        assert ds.num_video_frames == 3


class TestGetItem:
    def _sample(self, tmp_path, **kw):
        b = make_robocasa_bucket(tmp_path)
        kw.setdefault("normalize_mode", None)
        with _mock_video_decoder():
            return RoboCasa365Dataset(data_root=str(b), task_name="OpenDrawer", **kw)[0]

    def test_multiview_yields_l_canvas(self, tmp_path):
        s = self._sample(tmp_path, multiview=True, height=384, width=320)
        assert len(s["video"]) == 9  # (33-1)/4 + 1
        assert s["video"][0].size == (320, 384)  # PIL .size = (W, H)
        assert len(s["video_mask"]) == 9 and s["video_mask"].all()

    def test_action_20d_left_filled_right_zero(self, tmp_path):
        s = self._sample(tmp_path, multiview=False, height=64, width=96)
        assert s["action"].shape == (32, EEF_DIM)
        assert s["action"][:, :10].abs().sum() > 0  # left arm real
        assert (s["action"][:, 10:] == 0).all()  # right arm zero-padded

    def test_arm10_pins_known_pose(self):
        # Value-pin the absolute-pose contract (not just shape): a known 16-D state -> exact arm10,
        # so a silent change to the slicing / quat-convention / gripper formula is caught.
        from openwam.dataloader.robocasa365 import state_to_arm10

        st = np.zeros((1, 16), np.float32)
        st[0, 7:10] = [0.1, -0.2, 0.3]  # eef_pos_rel
        st[0, 10:14] = [0.0, 0.0, 0.0, 1.0]  # identity quaternion (xyzw)
        st[0, 14:16] = [0.05, 0.01]  # gripper qpos -> separation 0.04
        a = state_to_arm10(st)[0]
        assert a[:3] == pytest.approx([0.1, -0.2, 0.3])
        assert a[3:9] == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # identity rotation -> rot6d
        # gripper = width 0.04 rendered to command space: 1 - 20*0.04 = 0.2
        assert a[9] == pytest.approx(0.2)

    def test_proprio_20d_left_filled_right_zero(self, tmp_path):
        s = self._sample(tmp_path, multiview=False, height=64, width=96)
        assert s["proprio"].shape == (1, EEF_DIM)
        assert s["proprio"][0, :10].abs().sum() > 0
        assert (s["proprio"][0, 10:] == 0).all()

    def test_masks_left_arm_pattern(self, tmp_path):
        s = self._sample(tmp_path, multiview=False, height=64, width=96)
        assert s["action_mask"].shape == (32, EEF_DIM)
        assert s["action_mask"][:, :10].all()  # left arm valid
        assert not s["action_mask"][:, 10:].any()  # right arm masked
        assert s["proprio_mask"][0, :10].all()
        assert not s["proprio_mask"][0, 10:].any()

    def test_prompt_from_episodes_parquet(self, tmp_path):
        s = self._sample(tmp_path, multiview=False, height=64, width=96)
        assert s["prompt"].endswith(PROMPT)
        assert s["prompt"].startswith("A video recorded from a robot")


class TestNormalize:
    def test_min_max_autocompute_bounded(self, tmp_path):
        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(
                data_root=str(b), task_name="OpenDrawer", multiview=False, height=64, width=96, normalize_mode="min-max"
            )
            s = ds[0]
        assert (s["action"][:, :10].abs() <= 1.0 + 1e-5).all()
        # stats auto-written next to the bucket, 20-D for the trainer's buffers
        assert Path(ds.normalization_stats_path).exists()
        ns = ds.normalization_stats
        assert ns is not None and len(ns["mean"]) == EEF_DIM
        assert (ns["mean"][10:] == 0).all() and (ns["std"][10:] == 1).all()  # neutral right half

    def test_denormalize_roundtrip_left_arm(self, tmp_path):
        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(
                data_root=str(b), task_name="OpenDrawer", multiview=False, height=64, width=96, normalize_mode="min-max"
            )
            s = ds[0]
        # Real inverse check: s["action"] is NORMALIZED. denormalize -> raw, then re-normalize with
        # the same stats must recover the normalized action (a tautological shape-only check could
        # not catch a broken inverse transform).
        from openwam.dataloader.utils.normalization import apply_normalization

        deno = ds.denormalize_action(s["action"].numpy())  # normalized -> raw physical
        reno = apply_normalization(deno, ds.normalization_stats, "min-max")  # raw -> normalized
        assert reno[:, :10] == pytest.approx(s["action"].numpy()[:, :10], abs=1e-4)
        assert (deno[:, 10:] == 0).all()  # right arm stays zero

    def test_null_passthrough(self, tmp_path):
        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(data_root=str(b), multiview=False, height=64, width=96, normalize_mode=None)
            sample = ds[0]
        assert ds.normalization_stats is None
        assert sample["action"].shape == (32, EEF_DIM)

    def test_static_window_flagged(self, tmp_path, monkeypatch):
        # _build_sample must flag a no-motion window as static (the input the train-time resampler
        # uses to skip "hasn't-started-moving" windows; mirrors robotwin's filter_static_segments).
        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(
                data_root=str(b), task_name="OpenDrawer", multiview=False, height=64, width=96,
                normalize_mode=None, static_segment_threshold=1e-4,
            )
            base = np.zeros((1, 16), np.float32)
            base[0, 7:10] = [0.1, 0.2, 0.3]
            base[0, 10:14] = [0.0, 0.0, 0.0, 1.0]  # valid identity quaternion (xyzw)
            base[0, 14:16] = [0.04, 0.0]
            static = np.repeat(base, ds.num_frames, axis=0)
            monkeypatch.setattr(ds, "_read_state", lambda ep, s, e: static[: e - s].copy())
            assert ds._build_sample(0, 0)["_is_static"] is True
            moving = static.copy()
            moving[1, 7] += 0.05  # 5 cm EEF jump at step 1 -> not static
            monkeypatch.setattr(ds, "_read_state", lambda ep, s, e: moving[: e - s].copy())
            assert ds._build_sample(0, 0)["_is_static"] is False

    def test_unify_scatters_to_80d_left_valid_right_masked(self, tmp_path):
        # unify_action=true scatters the 20-D EEF into the shared 80-D space; single-arm robocasa365
        # lands the real arm in the LEFT eef slots (0-9, valid) and the zero-padded right half in
        # 34-43, masked out of the loss (LEFT_ARM_DIM_MASK honored through the scatter).
        from openwam.dataloader.utils.unify_action import UNIFY_DIM

        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(data_root=str(b), task_name="OpenDrawer", multiview=False, height=64,
                                    width=96, normalize_mode="min-max", unify_action=True,
                                    unify_action_map=["0-9", "34-43"])
            s = ds[0]
        assert ds.action_dim == UNIFY_DIM == 80
        assert s["action"].shape == (32, UNIFY_DIM)
        assert s["proprio"].shape == (1, UNIFY_DIM)
        am = s["action_mask"].numpy()
        assert am.shape == (32, UNIFY_DIM)
        assert am[0, :10].all()          # left eef slots valid
        assert not am[0, 10:].any()      # right eef (34-43) + all unmapped slots masked
        pm = s["proprio_mask"].numpy()
        assert pm[0, :10].all() and not pm[0, 10:].any()
        # denormalize un-unifies (80 -> 20) then unnormalizes -> physical 20-D
        assert ds.denormalize_action(s["action"].numpy()).shape == (32, EEF_DIM)

    def test_unify_off_is_20d(self, tmp_path):
        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(data_root=str(b), task_name="OpenDrawer", multiview=False, height=64,
                                    width=96, normalize_mode="min-max", unify_action=False)
            s = ds[0]
        assert ds.action_dim == EEF_DIM == 20
        assert s["action"].shape == (32, EEF_DIM)

    def test_mobile_base_symmetric_25d(self, tmp_path):
        # mobile_base=True folds base5 INTO the raw vector [arm20, base5], mapped via ONE 25-D map to
        # 80-D. ACTION base command valid in [68:73); PROPRIO base VELOCITY valid in [68:71), torso +
        # control_mode masked. denormalize returns 25-D [arm20, base5].
        from openwam.dataloader.utils.unify_action import UNIFY_DIM

        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(data_root=str(b), task_name="OpenDrawer", multiview=False, height=64,
                                    width=96, normalize_mode="min-max", unify_action=True,
                                    unify_action_map=["0-9", "34-43", "68-72"], mobile_base=True)
            s = ds._build_sample(0, 1)  # start>0 so the frame-0 base velocity is a real finite-diff
        assert ds.action_dim == UNIFY_DIM == 80
        am = s["action_mask"].numpy()
        assert am[0, 68:71].all() and not am[0, 71] and am[0, 72]  # vel + control_mode valid, torso masked (default)
        assert am[0, :10].all()          # left eef valid
        assert not am[0, 34:44].any()    # right eef masked
        assert np.abs(s["action"].numpy()[:, 68:73]).sum() > 0  # base carries a (normalized) command
        pm = s["proprio_mask"].numpy()
        assert pm[0, 68:71].all()                        # proprio base VELOCITY valid
        assert not pm[0, 71:73].any()                    # torso + control_mode masked in proprio
        assert np.abs(s["proprio"].numpy()[0, 68:71]).sum() > 0  # carries a (nonzero) base velocity
        deno = ds.denormalize_action(s["action"].numpy())
        assert deno.shape == (32, 25)  # [arm20, base5]

    def test_mobile_base_without_unify_is_25d(self, tmp_path):
        # mobile_base and unify_action are DECOUPLED: non-unify emits the raw 25-D [arm20, base5] head.
        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(data_root=str(b), task_name="OpenDrawer", multiview=False, height=64,
                                    width=96, normalize_mode="min-max", mobile_base=True)
            s = ds._build_sample(0, 1)
        assert ds.action_dim == 25
        assert s["action"].shape == (32, 25) and s["proprio"].shape == (1, 25)
        am = s["action_mask"].numpy()
        assert am[0, :10].all() and not am[0, 10:20].any()          # arm-left valid, right masked
        assert am[0, 20:23].all() and not am[0, 23] and am[0, 24]   # base vel + control_mode valid, torso masked
        pm = s["proprio_mask"].numpy()
        assert pm[0, 20:23].all() and not pm[0, 23:25].any()  # base velocity valid, torso+mode masked

    def test_mobile_proprio_velocity_command_space(self, tmp_path):
        # A′: the proprio base velocity is the finite-diff rescaled into the action's command space
        # (× fps / _BASE_VEL_PHYS_MAX), so it shares the action base stats. Value-pin against the
        # reader's own base_velocity_cmd on the same two base poses (no train/eval divergence).
        from openwam.dataloader.robocasa365 import base_velocity_cmd

        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(data_root=str(b), task_name="OpenDrawer", multiview=False, height=64,
                                    width=96, normalize_mode=None, mobile_base=True)  # null-norm → raw values
            s = ds._build_sample(0, 3)
        prev_cur = ds._read_state(0, 2, 4)[:, 0:7]  # base pose at start-1=2 .. start+1=4
        expected = base_velocity_cmd(prev_cur, ds._fps)
        assert s["proprio"].numpy()[0, 20:23] == pytest.approx(expected, abs=1e-5)

    def test_deploy_stats_roundtrip_20d(self, tmp_path):
        """The DEPLOY round-trip (the N1/S1 bug): the persisted stats are 20-D and keyed
        'eef', so the deploy normalizer (load_mode_stats + Normalizer) inverts the model's
        20-D action without a 10-vs-20 broadcast error and round-trips."""
        from openwam.dataloader.transforms.normalize import YAML_TO_NORM_MODE, Normalizer, load_mode_stats

        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(
                data_root=str(b), task_name="OpenDrawer", multiview=False, height=64, width=96, normalize_mode="min-max"
            )
            s = ds[0]
        # persisted file: 20-D stats under 'eef' (what save_normalization_stats copies + deploy reads)
        eef = load_mode_stats(ds.normalization_stats_path, "eef")
        assert eef is not None and len(eef["mean"]) == EEF_DIM, "persisted deploy stats must be 20-D"
        # deploy: build the SAME normalizer and un-normalize a 20-D model action (no broadcast error)
        norm = Normalizer(mode=YAML_TO_NORM_MODE["min-max"], stats=eef)
        act20 = s["action"][0].numpy()  # (20,)
        phys = norm.unnormalize(act20)
        assert phys.shape == (EEF_DIM,)
        assert norm.normalize(phys) == pytest.approx(act20, abs=1e-4)  # round-trips

    def test_quantile_mode_rejected(self, tmp_path):
        # quantile is not deploy-resolvable (YAML_TO_NORM_MODE lacks it) -> reject at init,
        # else the checkpoint silently loses normalization at serve time.
        b = make_robocasa_bucket(tmp_path)
        with pytest.raises(ValueError, match="deploy-resolvable"):
            RoboCasa365Dataset(data_root=str(b), normalize_mode="quantile", multiview=False, height=64, width=96)


class TestMultiAndRegistry:
    def test_multi_from_config_single_bucket(self, tmp_path):
        b = make_robocasa_bucket(tmp_path)
        cfg = {
            "type": "robocasa365",
            "dataset_dir": str(b),  # dataset_dir IS a bucket -> single-bucket mode
            "multiview": False,
            "height": 64,
            "width": 96,
            "normalize_mode": None,
        }
        with _mock_video_decoder():
            ds = MultiTaskRoboCasa365Dataset.from_config(cfg, split="train")
            s = ds[0]
        assert ds.action_dim == EEF_DIM
        assert len(ds._datasets) == 1
        assert s["action"].shape == (32, EEF_DIM)
        assert s["proprio"].shape == (1, EEF_DIM)

    def test_multi_surfaces_normalization_stats_path(self, tmp_path):
        # Regression: the trainer's save_normalization_stats() reads
        # dataset.normalization_stats_path off the REGISTERED (multi-task) wrapper to copy
        # normalization_stats.npy into the checkpoint dir, which deploy's _build_normalizer
        # REQUIRES (raises FileNotFoundError if absent). The wrapper must surface the
        # sub-dataset's resolved path — not just expose it on the inner RoboCasa365Dataset.
        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = MultiTaskRoboCasa365Dataset(
                dataset_dir=str(b), task_name="OpenDrawer",
                multiview=False, height=64, width=96, normalize_mode="min-max",
            )
            _ = ds[0]
        assert ds.normalization_stats_path is not None
        assert Path(ds.normalization_stats_path).exists()
        # exactly what deploy reads back: 20-D stats under the 'eef' key
        from openwam.dataloader.transforms.normalize import load_mode_stats

        eef = load_mode_stats(ds.normalization_stats_path, "eef")
        assert eef is not None and len(eef["min"]) == EEF_DIM

    def test_multi_discovers_tasks_by_source_prefix(self, tmp_path):
        # v3: ONE aggregated repo with two tasks (distinct source_prefix) -> two sub-datasets.
        root = make_multitask_bucket(tmp_path, tasks=["taskA", "taskB"])
        with _mock_video_decoder():
            ds = MultiTaskRoboCasa365Dataset(dataset_dir=str(root), multiview=False, height=64, width=96,
                                             normalize_mode=None)
        assert len(ds._datasets) == 2
        assert len(ds) == 2 * N_EPISODES * (EP_LENGTH - 1)

    def test_single_task_offset_in_shared_shard(self, tmp_path):
        # A task that is NOT first in its aggregated shard must read at its TRUE file-local offset:
        # offsets are computed over the FULL episode table before the single-task filter, else taskB
        # (physically after taskA in the same file) would read taskA's rows. Real repos pack many
        # tasks per shard, so this is the read-correctness contract the single-file case can't catch.
        root = make_multitask_bucket(tmp_path, tasks=["taskA", "taskB"], n_episodes=2)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(data_root=str(root), task_name="taskB", multiview=False, height=64,
                                    width=96, normalize_mode=None)
            s = ds._build_sample(0, 0)
        # taskB's first episode is GLOBAL episode 2 (seed=2), physically at row offset 2*EP_LENGTH.
        expected_arm0 = rc.state_to_arm10(_make_state(EP_LENGTH, seed=2))[0]
        assert s["proprio"][0, :10].numpy() == pytest.approx(expected_arm0, abs=1e-4)

    def test_multi_uses_one_shared_stats(self, tmp_path):
        # Multi-task + normalization → ONE shared stats file pooled over all tasks in the repo,
        # forwarded to every sub-dataset (robotwin's shared-stats contract).
        root = make_multitask_bucket(tmp_path, tasks=["taskA", "taskB"])
        with _mock_video_decoder():
            ds = MultiTaskRoboCasa365Dataset(dataset_dir=str(root), multiview=False, height=64, width=96,
                                             normalize_mode="min-max")
        shared = Path(root) / "robocasa365_multitask_eef_stats.npy"
        assert shared.exists(), "multi-task must pool ONE shared stats file at the repo root"
        # every sub-dataset points at the SAME shared file (not per-task stats)
        paths = {d.normalization_stats_path for d in ds._datasets}
        assert paths == {str(shared)}, paths
        # no per-task stats files were written
        assert not list(Path(root).glob("**/taskA_eef_stats.npy"))

    def test_multi_mobile_shared_eefbase_stats(self, tmp_path):
        # Multi-task + mobile_base: ONE shared _eefbase_ stats file (a combined 25-D 'eef_base' block)
        # pooled over all tasks and forwarded to every sub-dataset; the whole [arm20, base5] vector
        # maps to 80-D via the single map. Covers the multitask mobile path e2e (in-process).
        root = make_multitask_bucket(tmp_path, tasks=["taskA", "taskB"])
        with _mock_video_decoder():
            ds = MultiTaskRoboCasa365Dataset(
                dataset_dir=str(root), multiview=False, height=64, width=96, normalize_mode="min-max",
                unify_action=True, unify_action_map=["0-9", "34-43", "68-72"], mobile_base=True,
            )
            s = ds[0]
        shared = Path(root) / "robocasa365_multitask_eefbase_stats.npy"
        assert shared.exists(), "multi-task mobile must pool ONE shared _eefbase_ stats file"
        blob = np.load(shared, allow_pickle=True).item()
        assert "eef_base" in blob and len(blob["eef_base"]["mean"]) == 25, "shared stats need a 25-D eef_base block"
        assert "base" not in blob and "base_vel" not in blob, "no separate base/base_vel blocks (converged)"
        assert {d.normalization_stats_path for d in ds._datasets} == {str(shared)}  # all share it
        assert not list(Path(root).glob("**/*_eef_stats.npy"))  # not the arm-only file
        assert ds.action_dim == 80
        assert s["action"].shape == (32, 80)
        am = s["action_mask"].numpy()
        assert am[0, 68:71].all() and not am[0, 71] and am[0, 72]  # vel + control_mode valid, torso masked
        assert am[0, :10].all() and not am[0, 34:44].any()
        assert np.abs(s["action"].numpy()[:, 68:73]).sum() > 0  # base carries a command

    def test_root_mode_keeps_all_including_mobile(self, tmp_path):
        # Full RoboCasa365 (fixed-base filter removed): task discovery from the v3 aggregated repo
        # keeps EVERY task's source_prefix, including mobile (formerly moma_required=Yes) tasks — the
        # base command is trained, not dropped.
        root = make_multitask_bucket(tmp_path, tasks=["OpenDrawer", "SomeMobileTask"])
        roots = MultiTaskRoboCasa365Dataset._resolve_task_roots(str(root), None, None)
        assert {tn for tn, _ in roots} == {"OpenDrawer", "SomeMobileTask"}  # mobile task NOT dropped

    def test_registered_to_multi(self):
        from openwam.dataloader.registry import DATASET_REGISTRY, list_registered_datasets

        assert "robocasa365" in list_registered_datasets()
        assert DATASET_REGISTRY["robocasa365"] is MultiTaskRoboCasa365Dataset


class TestTinyArchTrainingStep:
    def test_sample_has_training_batch_keys(self, tmp_path):
        """The sample must carry every key + shape the trainer consumes (checkable
        without the heavy model stack)."""
        import torch

        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            s = RoboCasa365Dataset(data_root=str(b), multiview=False, height=64, width=96, normalize_mode=None)[0]
        for key in ("video", "action", "action_mask", "video_mask", "proprio", "proprio_mask", "prompt"):
            assert key in s, f"sample missing training key: {key}"
        assert isinstance(s["action"], torch.Tensor) and s["action"].shape == (32, EEF_DIM)
        assert isinstance(s["proprio"], torch.Tensor) and s["proprio"].shape == (1, EEF_DIM)
        assert s["action_mask"].dtype == torch.bool
        assert isinstance(s["prompt"], str) and s["prompt"]

    def test_sample_flows_through_prepare_inputs(self, tmp_path):
        """微型训练步: a sample flows through the trainer's ``prepare_inputs``
        (mirrors test_robotwin_dataloader). Needs the full model stack (Wan
        backbone → modelscope); skipped otherwise."""
        pytest.importorskip("modelscope")
        from tests.test_openwam_trainer import _make_tiny_arch

        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(data_root=str(b), multiview=False, height=64, width=96, normalize_mode=None)
            sample = ds[0]
        arch = _make_tiny_arch()
        inputs = arch.prepare_inputs(sample)
        assert inputs["action_is_pad"].shape == (1, 32, ds.action_dim)


def test_base_velocity_body_frame():
    """_base_velocity_body: 2 consecutive base poses (world base_pos(3)+quat(4)) -> body-frame
    [vx, vy, vyaw] (per-step displacement; SE(2), z + roll/pitch ignored)."""
    from openwam.dataloader.robocasa365 import _base_velocity_body

    def pose(x, y, yaw):  # base_position(3) + base_rotation quat xyzw (yaw about z)
        return [x, y, 0.7, 0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))]

    # 1) pure forward in world +x, no rotation -> body vx=+dx
    v = _base_velocity_body(np.array([pose(0, 0, 0.0), pose(0.1, 0, 0.0)], np.float32))
    assert v == pytest.approx([0.1, 0.0, 0.0], abs=1e-5)
    # 2) moved world +y while facing +y (yaw=pi/2) -> forward in body: vx=+0.1, vy=0, vyaw=pi/2
    v = _base_velocity_body(np.array([pose(0, 0, 0.0), pose(0, 0.1, np.pi / 2)], np.float32))
    assert v == pytest.approx([0.1, 0.0, np.pi / 2], abs=1e-4)
    # 3) pure rotation, no translation
    v = _base_velocity_body(np.array([pose(0, 0, 0.0), pose(0, 0, 0.5)], np.float32))
    assert v == pytest.approx([0.0, 0.0, 0.5], abs=1e-5)
    # 4) yaw wraparound: pi-0.1 -> -(pi-0.1) is a +0.2 step across ±pi, not -(2pi-0.2)
    v = _base_velocity_body(np.array([pose(0, 0, np.pi - 0.1), pose(0, 0, -(np.pi - 0.1))], np.float32))
    assert v[2] == pytest.approx(0.2, abs=1e-4)


def test_mobile_proprio_velocity_first_frame_zero(tmp_path):
    """At window start=0 there is no previous frame → proprio base velocity is 0 (slots still valid)."""
    b = make_robocasa_bucket(tmp_path)
    with _mock_video_decoder():
        ds = RoboCasa365Dataset(
            data_root=str(b), task_name="OpenDrawer", multiview=False, height=64, width=96,
            normalize_mode=None, unify_action=True, unify_action_map=["0-9", "34-43", "68-72"],
            mobile_base=True,
        )
        s = ds._build_sample(0, 0)
    assert s["proprio_mask"].numpy()[0, 68:71].all()
    assert (s["proprio"].numpy()[0, 68:71] == 0).all()


def test_mobile_proprio_velocity_normalized(tmp_path):
    """min-max mobile: the combined 25-D eef_base stats normalize the proprio base velocity into
    [-1, 1]; the stats file carries ONE 25-D 'eef_base' block (no separate base/base_vel blocks)."""
    b = make_robocasa_bucket(tmp_path)
    with _mock_video_decoder():
        ds = RoboCasa365Dataset(
            data_root=str(b), task_name="OpenDrawer", multiview=False, height=64, width=96,
            normalize_mode="min-max", unify_action=True, unify_action_map=["0-9", "34-43", "68-72"],
            mobile_base=True,
        )
        s = ds._build_sample(0, 1)
    blob = np.load(ds.normalization_stats_path, allow_pickle=True).item()
    assert "eef_base" in blob and len(blob["eef_base"]["mean"]) == 25
    assert "base_vel" not in blob and "base" not in blob
    assert ds.normalization_stats_path.endswith("_eefbase_stats.npy")
    bv = s["proprio"].numpy()[0, 68:71]
    assert (np.abs(bv) <= 1.0 + 1e-5).all()  # normalized into [-1, 1]
    assert s["proprio_mask"].numpy()[0, 68:71].all()


def test_action_gripper_is_command_proprio_is_rendered_width(tmp_path):
    # ACTION gripper (dim 9) = the recorded action.gripper_close command (exact timing, {-1,+1}), NOT
    # the achieved width; PROPRIO gripper = the achieved finger-separation width rendered to [-1,+1].
    from openwam.dataloader.robocasa365 import _gripper_width_to_cmd

    b = make_robocasa_bucket(tmp_path)
    with _mock_video_decoder():
        ds = RoboCasa365Dataset(data_root=str(b), task_name="OpenDrawer", multiview=False, height=64,
                                width=96, normalize_mode=None)  # raw, non-unify 20-D
        s = ds._build_sample(0, 0)
    st = _make_state(EP_LENGTH, seed=0)   # episode 0 = seed 0 (see _write_v3_repo)
    ac = _make_action(EP_LENGTH, seed=0)
    # proprio gripper (frame 0) = rendered achieved width
    assert s["proprio"].numpy()[0, 9] == pytest.approx(_gripper_width_to_cmd(st[0, 14] - st[0, 15]), abs=1e-5)
    # action gripper (step i) = the recorded command at frame i (idx 11), NOT the next-frame width
    assert s["action"].numpy()[:5, 9] == pytest.approx(ac[:5, 11], abs=1e-5)


def test_mask_torso_action(tmp_path):
    # mask_torso_action=True (default) masks torso (base idx 3 → 80-D slot 71) out of the ACTION loss;
    # control_mode (slot 72) stays supervised. False supervises all 5 base command dims.
    b = make_robocasa_bucket(tmp_path)
    with _mock_video_decoder():
        masked = RoboCasa365Dataset(data_root=str(b), task_name="OpenDrawer", multiview=False, height=64,
                                    width=96, normalize_mode="min-max", unify_action=True,
                                    unify_action_map=["0-9", "34-43", "68-72"], mobile_base=True)  # default mask=True
        unmasked = RoboCasa365Dataset(data_root=str(b), task_name="OpenDrawer", multiview=False, height=64,
                                      width=96, normalize_mode="min-max", unify_action=True,
                                      unify_action_map=["0-9", "34-43", "68-72"], mobile_base=True,
                                      mask_torso_action=False)
        s_masked = masked._build_sample(0, 1)
        am_masked = s_masked["action_mask"].numpy()[0]
        am_unmasked = unmasked._build_sample(0, 1)["action_mask"].numpy()[0]
    assert am_masked[68:71].all() and not am_masked[71] and am_masked[72]  # vel + mode valid, torso masked
    assert am_unmasked[68:73].all()                                        # all 5 base dims supervised
    # proprio is unaffected (torso already masked there regardless of mask_torso_action)
    pm = s_masked["proprio_mask"].numpy()[0]
    assert pm[68:71].all() and not pm[71:73].any()


def test_from_config_threads_mobile_base(tmp_path):
    # mobile_base MUST be reachable through from_config (all config-driven runs, even single-task, go
    # through MultiTaskRoboCasa365Dataset.from_config): the sub-dataset gets the flag, fills the base5
    # command + proprio velocity, and writes an _eefbase_ stats file with a 25-D eef_base block.
    b = make_robocasa_bucket(tmp_path)
    cfg = {
        "type": "robocasa365", "dataset_dir": str(b), "multiview": False, "height": 64, "width": 96,
        "normalize_mode": "min-max", "unify_action": True, "unify_action_map": ["0-9", "34-43", "68-72"],
        "mobile_base": True,
    }
    with _mock_video_decoder():
        ds = MultiTaskRoboCasa365Dataset.from_config(cfg, split="train")
        assert ds._datasets[0]._mobile_base is True
        s = ds._datasets[0]._build_sample(0, 1)  # start>0 → a finite-diff velocity
    pm = s["proprio_mask"].numpy()
    assert pm[0, 68:71].all() and not pm[0, 71:73].any()
    assert ds._datasets[0].normalization_stats_path.endswith("_eefbase_stats.npy")
    assert "eef_base" in np.load(ds._datasets[0].normalization_stats_path, allow_pickle=True).item()


def _two_repos(tmp_path):
    """Two SEPARATE v3 aggregated repos (the atomic + composite case): disjoint task sets."""
    a = _write_v3_repo(tmp_path / "repo_atomic", [("taskA", N_EPISODES), ("taskB", N_EPISODES)])
    b = _write_v3_repo(tmp_path / "repo_composite", [("taskC", N_EPISODES), ("taskD", N_EPISODES)])
    return str(a), str(b)


def test_multi_repo_discovers_across_repos(tmp_path):
    # dataset_dir = [repoA, repoB] -> discover tasks across BOTH repos (the 300-task atomic+composite
    # case: two separate HF repos). One sub-dataset per (task, its repo).
    a, b = _two_repos(tmp_path)
    with _mock_video_decoder():
        ds = MultiTaskRoboCasa365Dataset(dataset_dir=[a, b], multiview=False, height=64, width=96, normalize_mode=None)
    assert len(ds._datasets) == 4
    assert {d.task_name for d in ds._datasets} == {"taskA", "taskB", "taskC", "taskD"}
    assert len(ds) == 4 * N_EPISODES * (EP_LENGTH - 1)


def test_multi_repo_task_roots_subset(tmp_path):
    # task_roots selects across repos: one task from each repo.
    a, b = _two_repos(tmp_path)
    with _mock_video_decoder():
        ds = MultiTaskRoboCasa365Dataset(dataset_dir=[a, b], task_roots=["taskA", "taskC"],
                                         multiview=False, height=64, width=96, normalize_mode=None)
    assert {d.task_name for d in ds._datasets} == {"taskA", "taskC"}


def test_multi_repo_shared_stats_requires_explicit(tmp_path):
    # Multi-repo has no single root dir to auto-place the shared stats -> require an explicit
    # normalization_stats_path (no silent fallback).
    a, b = _two_repos(tmp_path)
    with _mock_video_decoder():
        with pytest.raises(ValueError, match="normalization_stats_path"):
            MultiTaskRoboCasa365Dataset(dataset_dir=[a, b], multiview=False, height=64, width=96,
                                        normalize_mode="min-max")


def test_multi_repo_shared_stats_explicit_pooled(tmp_path):
    # With an explicit stats path, stats are pooled over ALL tasks across BOTH repos and every
    # sub-dataset shares that one file.
    a, b = _two_repos(tmp_path)
    stats_path = str(tmp_path / "shared_multitask_stats.npy")
    with _mock_video_decoder():
        ds = MultiTaskRoboCasa365Dataset(dataset_dir=[a, b], normalization_stats_path=stats_path,
                                         multiview=False, height=64, width=96, normalize_mode="min-max")
        s = ds[0]
    assert Path(stats_path).exists()
    assert {d.normalization_stats_path for d in ds._datasets} == {stats_path}  # all share the pooled file
    assert s["action"].shape == (32, EEF_DIM)


def test_from_config_multi_repo(tmp_path):
    a, b = _two_repos(tmp_path)
    cfg = {"type": "robocasa365", "dataset_dir": [a, b], "multiview": False, "height": 64, "width": 96,
           "normalize_mode": None}
    with _mock_video_decoder():
        ds = MultiTaskRoboCasa365Dataset.from_config(cfg, split="train")
    assert len(ds._datasets) == 4


def test_single_repo_explicit_stats_path_honored(tmp_path):
    # An explicit normalization_stats_path is honored even if it doesn't exist yet: stats are computed
    # AT that path, not silently dropped / computed at the default location (no silent fallback).
    root = make_multitask_bucket(tmp_path, tasks=["taskA", "taskB"])
    explicit = str(tmp_path / "my_explicit_stats.npy")
    with _mock_video_decoder():
        ds = MultiTaskRoboCasa365Dataset(dataset_dir=str(root), normalization_stats_path=explicit,
                                         multiview=False, height=64, width=96, normalize_mode="min-max")
    assert Path(explicit).exists(), "explicit stats path must be honored as the compute target"
    assert {d.normalization_stats_path for d in ds._datasets} == {explicit}
    assert not (Path(root) / "robocasa365_multitask_eef_stats.npy").exists()  # default location NOT used
