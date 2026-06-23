"""Unit tests for the bespoke RoboCasa365 dataloader (raw LeRobot v2.1).

Builds a minimal RoboCasa365-shaped v2.1 bucket on disk (per-episode
``data/chunk-000/episode_NNNNNN.parquet`` with a 16-D ``observation.state`` column,
``meta/info.json`` with the v2.1 path templates, ``meta/episodes.jsonl`` with
per-episode task strings) and exercises the single-task + multi-task readers.
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
    state[:, 7:10] = rng.uniform(-1, 1, size=(n_rows, 3))  # eef_pos_rel
    q = rng.uniform(-1, 1, size=(n_rows, 4))
    state[:, 10:14] = q / np.linalg.norm(q, axis=1, keepdims=True)  # unit quat xyzw
    state[:, 14:16] = rng.uniform(0, 0.04, size=(n_rows, 2))  # gripper_qpos
    return state


def make_robocasa_bucket(tmp_path: Path, n_episodes: int = N_EPISODES) -> Path:
    """Write a minimal raw LeRobot v2.1 RoboCasa365 bucket; return its lerobot/ dir."""
    bucket = tmp_path / "OpenDrawer" / "20250816" / "lerobot"
    (bucket / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "PandaOmron",
        "chunks_size": 1000,
        "fps": 20,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info))

    with open(bucket / "meta" / "episodes.jsonl", "w") as f:
        for ep in range(n_episodes):
            data_dir = bucket / "data" / "chunk-000"
            data_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({"observation.state": list(_make_state(EP_LENGTH, seed=ep))})
            df.to_parquet(data_dir / f"episode_{ep:06d}.parquet")
            for cam in (HEAD_CAM, WRIST_CAM):
                vd = bucket / "videos" / "chunk-000" / cam
                vd.mkdir(parents=True, exist_ok=True)
                (vd / f"episode_{ep:06d}.mp4").write_bytes(b"")  # mocked decoder ignores content
            f.write(json.dumps({"episode_index": ep, "tasks": [PROMPT], "length": EP_LENGTH}) + "\n")
    return bucket


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

    def test_bad_temporal_contract_raises(self, tmp_path):
        # num_frames=9, video_stride=4 → 3 video frames; causal needs (3-1)%4==0 → fails.
        b = make_robocasa_bucket(tmp_path)
        with pytest.raises(ValueError, match="causal encoder"):
            RoboCasa365Dataset(data_root=str(b), normalize_mode=None, multiview=False, height=64, width=96,
                               num_frames=9, video_stride=4)


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

    def test_prompt_from_episodes_jsonl(self, tmp_path):
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
        deno = ds.denormalize_action(s["action"].numpy())
        assert deno.shape == s["action"].shape
        assert (deno[:, 10:] == 0).all()  # right arm stays zero

    def test_null_passthrough(self, tmp_path):
        b = make_robocasa_bucket(tmp_path)
        with _mock_video_decoder():
            ds = RoboCasa365Dataset(data_root=str(b), multiview=False, height=64, width=96, normalize_mode=None)
            sample = ds[0]
        assert ds.normalization_stats is None
        assert sample["action"].shape == (32, EEF_DIM)

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

    def test_multi_root_mode_discovers_buckets(self, tmp_path):
        # Two task dirs under a common root -> root-mode discovery.
        make_robocasa_bucket(tmp_path / "taskA")
        make_robocasa_bucket(tmp_path / "taskB")
        with _mock_video_decoder():
            ds = MultiTaskRoboCasa365Dataset(dataset_dir=str(tmp_path), multiview=False, height=64, width=96,
                                             normalize_mode=None)
        assert len(ds._datasets) == 2
        assert len(ds) == 2 * N_EPISODES * (EP_LENGTH - 1)

    def test_multi_root_uses_one_shared_stats(self, tmp_path):
        # Multi-bucket + normalization → ONE shared stats file pooled over all
        # buckets, forwarded to every sub-dataset (robotwin's shared-stats contract).
        make_robocasa_bucket(tmp_path / "taskA")
        make_robocasa_bucket(tmp_path / "taskB")
        with _mock_video_decoder():
            ds = MultiTaskRoboCasa365Dataset(dataset_dir=str(tmp_path), multiview=False, height=64, width=96,
                                             normalize_mode="min-max")
        shared = Path(tmp_path) / "robocasa365_multitask_eef_stats.npy"
        assert shared.exists(), "multi-bucket must pool ONE shared stats file at dataset_dir"
        # every sub-dataset points at the SAME shared file (not per-task stats)
        paths = {d.normalization_stats_path for d in ds._datasets}
        assert paths == {str(shared)}, paths
        # no per-task stats files were written under the buckets
        assert not list(Path(tmp_path).glob("**/taskA_eef_stats.npy"))

    def test_root_mode_filters_to_fixed_base(self, tmp_path):
        # Root discovery must drop non-fixed-base (mobile-base) buckets, which would violate
        # the base_motion=0 / control_mode=-1 design.
        from openwam.dataloader.robocasa365 import _fixed_base_task_names

        names = _fixed_base_task_names()
        assert "OpenDrawer" in names and len(names) == 112
        make_robocasa_bucket(tmp_path / "keep")  # -> .../OpenDrawer/.../lerobot (fixed-base)
        mobile = tmp_path / "drop" / "MobileNonFixedBaseTask" / "20250101" / "lerobot" / "meta"
        mobile.mkdir(parents=True)
        (mobile / "info.json").write_text("{}")  # just enough to be discovered
        roots = MultiTaskRoboCasa365Dataset._resolve_task_roots(str(tmp_path), None, None)
        assert {tn for tn, _ in roots} == {"OpenDrawer"}  # mobile task filtered out

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
