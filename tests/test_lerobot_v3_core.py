"""Core unit + integration tests for the LeRobot v3 dataloader.

Covers (no GPU / real dataset required):
  - ActionComposer: field extraction, multi-field concat, error handling
  - _flatten_layout: flat / 2-D / empty inputs
  - assemble_multiview_layout: L-shape geometry, missing camera, bad camera count
  - _pad_actions / _pad_video: T < target / T == target / T > target + mask correctness
  - LeRobot3Dataset.__getitem__: return dict keys/shapes, window split, proprio_source
  - Boundary: single-frame episode / last-frame pad
  - quat/euler -> rot6d identity correctness (EEF transform)
  - MultiTaskLeRobot3Dataset bisect boundary
  - MixtureDataset: virtual-len, dispatch, stats=None degradation, action-dim padding
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from tests.test_lerobot_v3_helpers import (
    _create_fake_task,
    _fake_task_and_ds,
    _make_lerobot3,
    _mkdir_subs,
    _mock_video,
)

# ---------------------------------------------------------------------------
# _flatten_layout
# ---------------------------------------------------------------------------


def test_flatten_layout_flat_list():
    from openwam.dataloader.lerobot_v3_base import _flatten_layout

    cams = ["cam_a", "cam_b", "cam_c"]
    assert _flatten_layout(cams) == cams


def test_flatten_layout_2d_list():
    from openwam.dataloader.lerobot_v3_base import _flatten_layout

    layout = [["cam_a", "cam_b"], ["cam_c", "cam_d"]]
    assert _flatten_layout(layout) == ["cam_a", "cam_b", "cam_c", "cam_d"]


def test_flatten_layout_empty():
    from openwam.dataloader.lerobot_v3_base import _flatten_layout

    assert _flatten_layout([]) == []


# ---------------------------------------------------------------------------
# ActionComposer
# ---------------------------------------------------------------------------


def test_action_composer_single_field_shape():
    from openwam.dataloader.lerobot_v3_base import ActionComposer

    info = {"action": {"shape": [14], "dtype": "float32"}}
    comp = ActionComposer(["action"], info)
    assert comp.action_dim == 14

    T = 5
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"action": [rng.standard_normal(14).astype(np.float32) for _ in range(T)]})
    out = comp.extract(df)
    assert out.shape == (T, 14)
    assert out.dtype == np.float32


def test_action_composer_multi_field_concat():
    from openwam.dataloader.lerobot_v3_base import ActionComposer

    info = {
        "action.left_arm": {"shape": [6], "dtype": "float32"},
        "action.right_arm": {"shape": [6], "dtype": "float32"},
        "action.left_gripper": {"shape": [1], "dtype": "float32"},
    }
    comp = ActionComposer(["action.left_arm", "action.right_arm", "action.left_gripper"], info)
    assert comp.action_dim == 13

    T = 4
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "action.left_arm": [rng.standard_normal(6).astype(np.float32) for _ in range(T)],
            "action.right_arm": [rng.standard_normal(6).astype(np.float32) for _ in range(T)],
            "action.left_gripper": rng.standard_normal(T).astype(np.float32),
        }
    )
    out = comp.extract(df)
    assert out.shape == (T, 13)


def test_action_composer_missing_field_raises():
    from openwam.dataloader.lerobot_v3_base import ActionComposer

    with pytest.raises(KeyError, match="action.missing"):
        ActionComposer(["action.missing"], {"action": {"shape": [14]}})


def test_action_composer_scalar_reshaped_to_column():
    from openwam.dataloader.lerobot_v3_base import ActionComposer

    info = {"grip": {"shape": [1], "dtype": "float32"}}
    comp = ActionComposer(["grip"], info)
    assert comp.action_dim == 1

    T = 3
    df = pd.DataFrame({"grip": np.array([0.1, 0.5, 0.9], dtype=np.float32)})
    out = comp.extract(df)
    assert out.shape == (T, 1)


# ---------------------------------------------------------------------------
# assemble_multiview_layout
# ---------------------------------------------------------------------------


def test_assemble_multiview_lshape_canvas_size():
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout

    cameras = ["top", "bot_l", "bot_r"]
    frames = {c: Image.new("RGB", (100, 100), color=(255, 0, 0)) for c in cameras}
    out = assemble_multiview_layout(frames, cameras, out_h=120, out_w=160)
    assert out.size == (160, 120)


def test_assemble_multiview_missing_camera_black_fill():
    """assemble_multiview_layout leaves a black region for any missing key.

    In normal __getitem__ flow the caller pre-pads short cameras with the last
    frame, so missing keys only occur on genuine data-absence.  The function
    itself still produces black for any absent camera_layout key.
    """
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout

    cameras = ["top", "bot_l", "bot_r"]
    # Only provide top camera; bot_l and bot_r are missing → black region
    frames = {"top": Image.new("RGB", (100, 100), color=(200, 200, 200))}
    out = assemble_multiview_layout(frames, cameras, out_h=60, out_w=60)
    arr = np.array(out)
    top_h = int(round(60 * 2 / 3))
    bottom_region = arr[top_h:, :, :]
    assert bottom_region.max() == 0, "Missing cameras should produce black regions"


def test_assemble_multiview_wrong_camera_count_raises():
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout

    with pytest.raises(ValueError, match="3 cameras"):
        assemble_multiview_layout({}, ["cam_a", "cam_b"], out_h=60, out_w=60)


# ---------------------------------------------------------------------------
# quat → rot6d correctness (critical path for EEF transform)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quat_convention,identity_quat",
    [("wxyz", [1, 0, 0, 0]), ("xyzw", [0, 0, 0, 1])],
)
def test_eef_transform_identity_pose(quat_convention, identity_quat):
    """Identity quaternion should produce rot6d = [1,0,0, 0,1,0] under either convention."""
    from openwam.dataloader.agibot import make_eef_transform

    transform = make_eef_transform(quat_convention=quat_convention)

    # 1-frame 40-D action, only left/right quaternions set to identity.
    # Layout: [gripper_l(1), gripper_r(1), xyz_l(3), xyz_r(3), q_l(4), q_r(4), ...]
    action = np.zeros((1, 40), dtype=np.float32)
    action[0, 8:12] = identity_quat  # q_l identity
    action[0, 12:16] = identity_quat  # q_r identity

    out = transform(action)  # (1, 20)
    assert out.shape == (1, 20)

    # left rot6d at cols 3:9, right rot6d at cols 13:19
    identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    np.testing.assert_allclose(out[0, 3:9], identity_rot6d, atol=1e-5)
    np.testing.assert_allclose(out[0, 13:19], identity_rot6d, atol=1e-5)


# ---------------------------------------------------------------------------
# OXE EEF transform correctness
# ---------------------------------------------------------------------------


def test_oxe_eef_transform_identity_euler():
    """Zero euler angles (identity rotation) should produce rot6d = [1,0,0, 0,1,0]."""
    from openwam.dataloader.oxe import _make_oxe_eef_transform

    transform, _mask = _make_oxe_eef_transform()

    # 1-frame, 7-D: xyz(3) + euler_rpy(3) + gripper(1)
    action = np.zeros((1, 7), dtype=np.float32)
    action[0, 0:3] = [1.0, 2.0, 3.0]  # xyz
    action[0, 3:6] = [0.0, 0.0, 0.0]  # identity euler
    action[0, 6] = 0.5  # gripper

    out = transform(action)
    assert out.shape == (1, 20)

    # xyz preserved
    np.testing.assert_allclose(out[0, 0:3], [1.0, 2.0, 3.0], atol=1e-5)
    # identity rot6d
    np.testing.assert_allclose(out[0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-5)
    # gripper preserved
    np.testing.assert_allclose(out[0, 9], 0.5, atol=1e-5)
    # second arm is all zeros
    np.testing.assert_array_equal(out[0, 10:], 0.0)


def test_oxe_eef_transform_mask_shape():
    """action_dim_mask should be (20,) bool with first 10 True, last 10 False."""
    from openwam.dataloader.oxe import _make_oxe_eef_transform

    _, mask = _make_oxe_eef_transform()
    assert mask.shape == (20,)
    assert mask.dtype == bool
    assert mask[:10].all()
    assert not mask[10:].any()


def test_oxe_eef_transform_multi_frame():
    """Transform should handle multi-frame input and output float32."""
    from openwam.dataloader.oxe import _make_oxe_eef_transform

    transform, _ = _make_oxe_eef_transform()
    rng = np.random.default_rng(42)
    actions = rng.standard_normal((10, 7)).astype(np.float32)
    out = transform(actions)
    assert out.shape == (10, 20)
    assert out.dtype == np.float32
    # second arm padding stays zero across all frames
    np.testing.assert_array_equal(out[:, 10:], 0.0)


# ---------------------------------------------------------------------------
# _pad_actions / _pad_video  (tested via dataset instance, no __getitem__ call)
# ---------------------------------------------------------------------------


def test_pad_actions_short(tmp_path):
    ds = _fake_task_and_ds(tmp_path)
    actions = np.ones((5, 14), dtype=np.float32)
    padded, mask = ds._pad_actions(actions, target=10)
    assert padded.shape == (10, 14)
    assert mask[:5].all()
    assert not mask[5:].any()
    # Padded tail repeats last real action
    assert (padded[5:] == actions[-1]).all()


def test_pad_actions_exact(tmp_path):
    ds = _fake_task_and_ds(tmp_path)
    actions = np.ones((8, 14), dtype=np.float32)
    padded, mask = ds._pad_actions(actions, target=8)
    assert padded.shape == (8, 14)
    assert mask.all()


def test_pad_actions_long(tmp_path):
    ds = _fake_task_and_ds(tmp_path)
    actions = np.ones((15, 14), dtype=np.float32)
    padded, mask = ds._pad_actions(actions, target=8)
    assert padded.shape == (8, 14)
    assert mask.all()


def test_pad_video_short(tmp_path):
    ds = _fake_task_and_ds(tmp_path)
    frames = [Image.new("RGB", (16, 16)) for _ in range(3)]
    padded, mask = ds._pad_video(frames, target=7)
    assert len(padded) == 7
    assert mask[:3].all()
    assert not mask[3:].any()


def test_pad_video_exact(tmp_path):
    ds = _fake_task_and_ds(tmp_path)
    frames = [Image.new("RGB", (16, 16)) for _ in range(5)]
    padded, mask = ds._pad_video(frames, target=5)
    assert len(padded) == 5
    assert mask.all()


# ---------------------------------------------------------------------------
# LeRobot3Dataset integration tests
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "video",
    "vace_video",
    "first_frame_image",
    "action",
    "action_mask",
    "video_mask",
    "proprio",
    "proprio_mask",
    "prompt",
    "episode_index",
    "episode_path",
    "start_frame",
    "end_frame",
    "episode_length",
    "task_name",
    "proprio_source",
}


def test_getitem_return_dict_keys(tmp_path):
    ds = _fake_task_and_ds(tmp_path)
    with _mock_video():
        sample = ds[0]
    assert _REQUIRED_KEYS.issubset(sample.keys()), f"Missing keys: {_REQUIRED_KEYS - sample.keys()}"


def test_getitem_action_and_proprio_shapes(tmp_path):
    """Window split: proprio=(1,D), action=(num_frames-1,D)."""
    num_frames = 9
    action_dim = 14

    ds = _fake_task_and_ds(tmp_path, task_kwargs={"ep_len": 30}, num_frames=num_frames)
    with _mock_video():
        sample = ds[0]

    assert sample["proprio"].shape == (1, action_dim)
    assert sample["proprio_mask"].shape == (1,)
    assert sample["action"].shape == (num_frames - 1, action_dim)
    assert sample["action_mask"].shape == (num_frames - 1,)


def test_getitem_action_dim_mask_fallback_all_true(tmp_path):
    """When the dataset doesn't supply an explicit action_dim_mask, every sample
    still carries an all-True mask of shape (action_dim,) so downstream loss
    code can apply mask-aware reductions uniformly without missing-key
    branches. Padded datasets (e.g. OXE) override this with a real mask."""
    action_dim = 14

    # action_dim_mask intentionally not passed
    ds = _fake_task_and_ds(tmp_path)
    assert ds._action_dim_mask is None
    with _mock_video():
        sample = ds[0]

    assert "action_dim_mask" in sample
    assert sample["action_dim_mask"].shape == (action_dim,)
    assert sample["action_dim_mask"].dtype == torch.bool
    assert sample["action_dim_mask"].all()


def test_getitem_proprio_source_action_fallback(tmp_path):
    """Without observation.state in info.json, proprio_source should be 'action_fallback'."""
    ds = _fake_task_and_ds(tmp_path, task_kwargs={"with_state": False})
    assert ds._state_composer is None
    with _mock_video():
        sample = ds[0]

    assert sample["proprio_source"] == "action_fallback"


def test_getitem_proprio_source_native_with_observation_state(tmp_path):
    """With observation.state in info.json, proprio_source should be 'native'."""
    # state_dim must equal action_dim (14) for the dimension check to pass
    state_dim = 14
    ds = _fake_task_and_ds(tmp_path, task_kwargs={"with_state": True, "state_dim": state_dim})
    assert ds._state_composer is not None
    with _mock_video():
        sample = ds[0]

    assert sample["proprio_source"] == "native"
    assert sample["proprio"].shape == (1, state_dim)


# ---------------------------------------------------------------------------
# Boundary: single-frame episode / last-frame pad
# ---------------------------------------------------------------------------


def test_single_frame_episode_pads_correctly(tmp_path):
    """Single-frame episode: action should be all-pad with mask all False."""
    ds = _fake_task_and_ds(tmp_path, task_kwargs={"ep_len": 1})
    with _mock_video():
        sample = ds[0]
    # action shape: (num_frames-1, D)
    assert sample["action"].shape[0] == 8
    # No real actions in a single-frame episode — mask should be all False
    assert not sample["action_mask"].any()


def test_last_frame_pad_repeats_last_action(tmp_path):
    """Episode shorter than num_frames: padded tail should repeat the last real action."""
    ds = _fake_task_and_ds(tmp_path, task_kwargs={"ep_len": 5})
    with _mock_video():
        sample = ds[0]
    action = sample["action"]  # (8, D)
    mask = sample["action_mask"]  # (8,)
    # ep_len=5: proprio=raw[0], action=raw[1:5] → 4 valid frames, 4 padded
    assert mask[:4].all()
    assert not mask[4:].any()
    # Padded frames must equal the last valid action frame
    last = action[3].numpy()
    np.testing.assert_array_equal(action[4:].numpy(), np.tile(last, (4, 1)))


# ---------------------------------------------------------------------------
# MultiTaskLeRobot3Dataset bisect boundary
# ---------------------------------------------------------------------------


def test_multitask_bisect_first_and_last_index(tmp_path):
    """idx=0 should dispatch to the first sub-dataset; idx=total-1 to the last."""
    from openwam.dataloader.lerobot_v3_base import MultiTaskLeRobot3Dataset

    d1, d2 = _mkdir_subs(tmp_path, "d1", "d2")
    _create_fake_task(str(d1), n_episodes=2, ep_len=20)
    _create_fake_task(str(d2), n_episodes=2, ep_len=20)
    ds1 = _make_lerobot3(d1)
    ds2 = _make_lerobot3(d2)
    multi = MultiTaskLeRobot3Dataset([ds1, ds2])

    total = len(multi)
    assert total == len(ds1) + len(ds2)

    with _mock_video():
        # idx=0 → first sample of ds1
        s0 = multi[0]
        assert s0["task_name"] == ds1.task_name

        # idx=total-1 → last sample of ds2
        s_last = multi[total - 1]
        assert s_last["task_name"] == ds2.task_name


# ---------------------------------------------------------------------------
# MixtureDataset
# ---------------------------------------------------------------------------


class _FakeDS:
    """Minimal fake dataset for MixtureDataset tests (no disk I/O)."""

    def __init__(self, n: int, action_dim: int, stats: dict | None = None):
        self._n = n
        self._action_dim = action_dim
        self._stats = stats

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {
            "action": torch.zeros(self._action_dim),
            "action_mask": torch.ones(self._action_dim, dtype=torch.bool),
        }

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_stats(self) -> dict | None:
        return self._stats

    def denormalize_action(self, x):
        return x


def _fake_stats(dim: int, mean: float = 0.0, std: float = 1.0) -> dict:
    return {
        "mean": np.full(dim, mean, dtype=np.float32),
        "std": np.full(dim, std, dtype=np.float32),
        "min": np.full(dim, -1.0, dtype=np.float32),
        "max": np.full(dim, 1.0, dtype=np.float32),
    }


def test_mixture_virtual_len_covers_total_real():
    """Total virtual samples should equal round(total_real * w) summed."""
    from openwam.dataloader.mixture import MixtureDataset

    ds_a = _FakeDS(100, 14)
    ds_b = _FakeDS(200, 14)
    mix = MixtureDataset([ds_a, ds_b], weights=[1.0, 1.0])
    total_real = 300
    # Each dataset gets ~round(total_real * 0.5) = 150 virtual samples
    # Total virtual = 300
    assert len(mix) == total_real


def test_mixture_dispatch_reaches_all_subdatasets():
    """Every sub-dataset index should appear in the virtual index map."""
    from openwam.dataloader.mixture import MixtureDataset

    ds_a = _FakeDS(50, 14)
    ds_b = _FakeDS(50, 14)
    mix = MixtureDataset([ds_a, ds_b], weights=[1.0, 1.0])
    seen_ds = {di for di, _ in mix._index_map}
    assert seen_ds == {0, 1}


def test_mixture_stats_none_when_any_subdataset_has_no_stats():
    """If any sub-dataset has action_stats=None, mixture stats should be None."""
    from openwam.dataloader.mixture import MixtureDataset

    ds_a = _FakeDS(50, 14, stats=_fake_stats(14))
    ds_b = _FakeDS(50, 14, stats=None)  # no stats
    mix = MixtureDataset([ds_a, ds_b], weights=[1.0, 1.0])
    assert mix.action_stats is None


def test_mixture_stats_computed_when_all_have_stats():
    """If all sub-datasets have stats, mixture stats should be non-None."""
    from openwam.dataloader.mixture import MixtureDataset

    ds_a = _FakeDS(50, 14, stats=_fake_stats(14))
    ds_b = _FakeDS(50, 14, stats=_fake_stats(14))
    mix = MixtureDataset([ds_a, ds_b], weights=[1.0, 1.0])
    assert mix.action_stats is not None
    assert "mean" in mix.action_stats and "std" in mix.action_stats


def test_mixture_action_dim_padding():
    """action from a smaller-dim sub-dataset must be zero-padded to action_dim_override."""
    from openwam.dataloader.mixture import MixtureDataset

    ds_small = _FakeDS(10, 14)  # 14-D
    ds_large = _FakeDS(10, 20)  # 20-D
    mix = MixtureDataset([ds_small, ds_large], weights=[1.0, 1.0], action_dim_override=20)

    # Find an index that dispatches to ds_small (di=0)
    target_idx = next(i for i, (di, _) in enumerate(mix._index_map) if di == 0)
    sample = mix[target_idx]
    assert sample["action"].shape[-1] == 20
    # Padded dims should be zero
    assert (sample["action"][14:] == 0.0).all()
