"""Unit tests for action normalization on the deployment path.

Covers _build_normalizer (reads normalization_stats.npy + cfg) and the
Normalizer normalize/unnormalize invariants used by deploy.

Pure CPU, no GPU, no network. Uses pytest's tmp_path fixture so there's no
dependency on any real checkpoint directory.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from openwam.dataloader.transforms.normalize import Normalizer
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec
from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.model_loader import _build_normalizer, _infer_raw_dim, _UnifyAwareNormalizer
from openwam.model.architectures.base import BaseWAMArchitecture

# --- Helper: build a realistic stats dict for a 20D eef action ---


def _eef_stats_min_max():
    """Build normalization_stats in the nested schema with eef range simulating real robot."""
    # Simulate a physical workspace roughly ±0.8 m for xyz, [-1, 1] for rot6d,
    # [0, 1] for gripper. 20D = [lxyz(3), lrot(6), lgrip(1), rxyz(3), rrot(6), rgrip(1)]
    lo = np.array([-0.8, -0.8, -0.2] + [-1.0] * 6 + [0.0] + [-0.8, -0.8, -0.2] + [-1.0] * 6 + [0.0], dtype=np.float32)
    hi = np.array([0.8, 0.8, 1.5] + [1.0] * 6 + [1.0] + [0.8, 0.8, 1.5] + [1.0] * 6 + [1.0], dtype=np.float32)
    mean = (lo + hi) / 2
    std = (hi - lo) / 4
    return {
        "mean": mean,
        "std": np.maximum(std, 1e-6),
        "min": lo,
        "max": hi,
        "q01": lo,
        "q99": hi,
    }


def _write_stats_file(tmp_path, mode_key: str = "eef"):
    """Write a nested-schema normalization_stats.npy into tmp_path and return its path."""
    stats = {mode_key: _eef_stats_min_max(), "num_timesteps": 1000}
    p = tmp_path / "normalization_stats.npy"
    np.save(str(p), stats, allow_pickle=True)
    return str(p)


# --- Normalizer round-trip tests ---


def test_normalizer_min_max_roundtrip():
    stats = _eef_stats_min_max()
    norm = Normalizer(mode="min_max", stats=stats)
    x = np.random.RandomState(0).uniform(-0.5, 0.5, size=(4, 20)).astype(np.float32)
    # Clamp to stats range so round-trip is well-defined
    x = np.clip(x, stats["min"], stats["max"])
    y = norm.normalize(x)
    x_back = norm.unnormalize(y)
    np.testing.assert_allclose(x, x_back, atol=1e-5)


def test_normalizer_zscore_roundtrip():
    stats = _eef_stats_min_max()
    norm = Normalizer(mode="mean_std", stats=stats)
    x = np.random.RandomState(1).uniform(-0.5, 0.5, size=(4, 20)).astype(np.float32)
    y = norm.normalize(x)
    x_back = norm.unnormalize(y)
    np.testing.assert_allclose(x, x_back, atol=1e-5)


# --- _build_normalizer branch tests ---


def test_build_normalizer_happy_path(tmp_path):
    _write_stats_file(tmp_path, mode_key="eef")
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "eef"}})
    normalizer = _build_normalizer(cfg, str(tmp_path))
    assert normalizer is not None
    assert isinstance(normalizer, Normalizer)

    # Feeding a normalized zero vector should map to the center of the range.
    # For min-max with [lo, hi], normalize(x) = 2*(x-lo)/(hi-lo) - 1, so x=0
    # (normalized) => x = (lo+hi)/2 (physical).
    out = normalizer.unnormalize(np.zeros(20, dtype=np.float32))
    stats = _eef_stats_min_max()
    expected = (stats["min"] + stats["max"]) / 2
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_build_normalizer_missing_stats_raises(tmp_path):
    # tmp_path is empty — no normalization_stats.npy
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "eef"}})
    with pytest.raises(FileNotFoundError, match="Missing required normalization_stats.npy"):
        _build_normalizer(cfg, str(tmp_path))


@pytest.mark.parametrize("disabled_value", [None, "none", "null", ""])
def test_build_normalizer_disabled_mode_returns_none(tmp_path, disabled_value):
    _write_stats_file(tmp_path)
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": disabled_value, "action_mode": "eef"}})
    normalizer = _build_normalizer(cfg, str(tmp_path))
    assert normalizer is None


@pytest.mark.parametrize("disabled_value", [None, "none", "null", ""])
def test_build_normalizer_disabled_mode_does_not_require_stats(tmp_path, disabled_value):
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": disabled_value, "action_mode": "eef"}})
    normalizer = _build_normalizer(cfg, str(tmp_path))
    assert normalizer is None


def test_build_normalizer_unknown_mode_returns_none(tmp_path):
    _write_stats_file(tmp_path)
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "bogus", "action_mode": "eef"}})
    normalizer = _build_normalizer(cfg, str(tmp_path))
    assert normalizer is None


def test_build_normalizer_wrong_action_mode_returns_none(tmp_path):
    # Stats file has only "eef" but config says action_mode="joint"
    _write_stats_file(tmp_path, mode_key="eef")
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "joint"}})
    normalizer = _build_normalizer(cfg, str(tmp_path))
    assert normalizer is None


# --- Deployment-path invariant: unnormalize must push xyz beyond [-1, 1] ---


def test_deployment_action_range_sanity(tmp_path):
    """Mirrors the action postprocessing in BaseWAMArchitecture.generate().

    Model output is in [-1, 1] (after flow-matching). After unnormalize, xyz
    dims must reach physical range (here: ±0.8 m). If unnormalize were missing,
    this guard would catch the regression.
    """
    _write_stats_file(tmp_path)
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "eef"}})
    normalizer = _build_normalizer(cfg, str(tmp_path))
    assert normalizer is not None, "prerequisite: action normalizer must build"

    # Simulate a model output batch: 33-step action chunk in [-1, 1].
    normalized = np.random.RandomState(42).uniform(-1.0, 1.0, size=(33, 20)).astype(np.float32)

    # Exact deploy action postprocessing: normalized model output -> physical units.
    actions = normalizer.unnormalize(normalized)

    # xyz indices in the 20D eef layout: left xyz = [0,1,2], right xyz = [10,11,12]
    xyz_abs_max = float(np.abs(actions[:, [0, 1, 2, 10, 11, 12]]).max())
    assert xyz_abs_max > 1.0, (
        f"xyz.abs().max()={xyz_abs_max:.4f} after unnormalize — expected > 1.0 "
        f"(stats x range is ±0.8 m but full span is ±1.5 m on z). This would fire if "
        f"the action normalizer stopped applying."
    )

    assert float(np.abs(actions).max()) > 1.0


class _TinyScheduler:
    num_train_timesteps = 1000

    def set_timesteps(self, n, shift=5.0):
        del shift
        self.timesteps = torch.linspace(1.0, 0.0, n)
        self.sigmas = torch.linspace(1.0, 0.0, n)

    def flow_step(self, noise_pred, sigma, sigma_next, sample):
        del sigma, sigma_next
        return sample + noise_pred


class _CaptureDeployArchitecture:
    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.video_scheduler = _TinyScheduler()
        self.action_scheduler = _TinyScheduler()
        self.seen_proprio = None

    def apply_compile_optimizations(self, compile_cfg):
        del compile_cfg

    def normalize_deploy_proprio(self, proprio):
        arr = np.asarray(proprio, dtype=np.float32)
        return torch.from_numpy(self.normalizer.normalize(arr))

    def generate(self, **kwargs):
        self.seen_proprio = kwargs["proprio"]
        return {"video": None, "actions": np.zeros((1, 20), dtype=np.float32)}


def test_joint_engine_normalizes_raw_deploy_state_before_generate():
    """JointInferenceEngine must pass normalized state into architecture.generate()."""
    stats = _eef_stats_min_max()
    normalizer = Normalizer(mode="min_max", stats=stats)
    arch = _CaptureDeployArchitecture(normalizer)
    cfg = OmegaConf.create(
        {
            "inference": {"denoise_steps": 2, "schedule_type": "sync", "shift": 5.0},
            "optimization": {"decode_video": False},
        }
    )
    engine = JointInferenceEngine(cfg=cfg, architecture=arch)

    raw_state = stats["max"].astype(np.float32)
    engine.generate({"observation": {"state": raw_state}, "num_frames": 2})

    assert arch.seen_proprio is not None
    np.testing.assert_allclose(arch.seen_proprio.numpy(), np.ones_like(raw_state), atol=1e-6)


class _TinyVideoBackbone:
    dim = 4

    def __init__(self):
        self.scheduler = _TinyScheduler()

    def preprocess_input_for_inference(self, *args, **kwargs):
        del args, kwargs
        return {"latents": torch.zeros(1, 1, 1, 1, 1)}

    def decode_video(self, latents, tiled=True):
        del latents, tiled
        return None


class _TinyGenerateArchitecture(BaseWAMArchitecture):
    def __init__(self, normalizer):
        nn.Module.__init__(self)
        self.cfg = None
        self.video_backbone = _TinyVideoBackbone()
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self.normalizer = normalizer
        self._use_proprioception_context = False

        class _ActionBackbone:
            action_dim = 20
            scheduler = _TinyScheduler()
            uses_proprioception = False

        self.action_backbone = _ActionBackbone()

    def forward(self, noisy_actions, action_timestep, **kwargs):
        del action_timestep, kwargs
        video_noise = torch.zeros(1, 1, 1, 1, 1)
        action_noise = torch.zeros_like(noisy_actions)
        return video_noise, action_noise


def test_base_generate_unnormalizes_deploy_actions():
    """BaseWAMArchitecture.generate() must return physical-unit actions when normalizer is attached."""
    stats = _eef_stats_min_max()
    normalizer = Normalizer(mode="min_max", stats=stats)
    arch = _TinyGenerateArchitecture(normalizer)

    result = arch.generate(
        schedule=[(0.0, 1.0), (0.0, 0.0)],
        prompt="",
        num_frames=2,
        decode_video=False,
        seed=123,
    )

    normalized = (
        torch.randn(
            1,
            1,
            20,
            generator=torch.Generator(device="cpu").manual_seed(123),
        )
        .squeeze(0)
        .numpy()
    )
    expected = normalizer.unnormalize(normalized)
    np.testing.assert_allclose(result["actions"], expected, atol=1e-6)


# --- _UnifyAwareNormalizer + unify dispatch (unify_action ckpts) ---

_UNIFY_MAP = ["0-9", "32-41"]  # 20-D raw eef -> unified slots [0:10) + [32:42)


def _unify_dst():
    return parse_unify_spec(_UNIFY_MAP, UNIFY_DIM)


def _clipped_raw(seed: int, n: int = 5):
    stats = _eef_stats_min_max()
    raw = np.random.RandomState(seed).uniform(-0.5, 0.5, size=(n, 20)).astype(np.float32)
    return np.clip(raw, stats["min"], stats["max"])


def test_unify_action_out_roundtrip():
    """action OUT: raw -> train forward (normalize->scatter) -> deploy inverse (gather->unnormalize)."""
    inner = Normalizer(mode="min_max", stats=_eef_stats_min_max())
    dst = _unify_dst()
    raw = _clipped_raw(0)
    unified, _ = map_to_unify(inner.normalize(raw), dst, UNIFY_DIM)  # what the model is trained on
    recovered = _UnifyAwareNormalizer(inner, dst, UNIFY_DIM).unnormalize(unified)
    assert recovered.shape == raw.shape
    np.testing.assert_allclose(recovered, raw, atol=1e-5)


def test_unify_proprio_in_matches_train_forward():
    """proprio IN: wrapper.normalize(raw) == map_to_unify(inner.normalize(raw))."""
    inner = Normalizer(mode="min_max", stats=_eef_stats_min_max())
    dst = _unify_dst()
    raw = np.random.RandomState(1).uniform(-0.5, 0.5, size=(3, 20)).astype(np.float32)
    expected, _ = map_to_unify(inner.normalize(raw), dst, UNIFY_DIM)
    np.testing.assert_allclose(_UnifyAwareNormalizer(inner, dst, UNIFY_DIM).normalize(raw), expected, atol=1e-6)


def test_unify_gather_only_when_inner_none():
    """inner=None: unnormalize only gathers (80->raw), normalize only scatters (raw->80)."""
    dst = _unify_dst()
    w = _UnifyAwareNormalizer(None, dst, UNIFY_DIM)
    raw = np.random.RandomState(2).uniform(-1, 1, size=(4, 20)).astype(np.float32)
    unified, _ = map_to_unify(raw, dst, UNIFY_DIM)
    np.testing.assert_allclose(w.unnormalize(unified), raw, atol=1e-6)
    np.testing.assert_allclose(w.normalize(raw), unified, atol=1e-6)


def test_unify_unnormalize_passthrough_when_not_unify_dim():
    """Defensive branch: last-dim != unify_dim -> skip gather, delegate to inner.unnormalize."""
    inner = Normalizer(mode="min_max", stats=_eef_stats_min_max())
    w = _UnifyAwareNormalizer(inner, _unify_dst(), UNIFY_DIM)
    raw_width = np.random.RandomState(3).uniform(-1, 1, size=(2, 20)).astype(np.float32)  # 20 != 80
    np.testing.assert_allclose(w.unnormalize(raw_width), inner.unnormalize(raw_width), atol=1e-6)


def test_infer_raw_dim():
    assert _infer_raw_dim(Normalizer(mode="min_max", stats=_eef_stats_min_max())) == 20
    assert _infer_raw_dim(None) is None


def test_build_normalizer_unify_off_returns_plain_inner(tmp_path):
    _write_stats_file(tmp_path, mode_key="eef")
    cfg = OmegaConf.create(
        {"dataloader": {"normalize_mode": "min-max", "action_mode": "eef", "unify_action": False}}
    )
    norm = _build_normalizer(cfg, str(tmp_path))
    assert isinstance(norm, Normalizer) and not isinstance(norm, _UnifyAwareNormalizer)


def test_build_normalizer_unify_on_wraps_and_roundtrips(tmp_path):
    _write_stats_file(tmp_path, mode_key="eef")
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "eef",
                "unify_action": True,
                "unify_action_map": _UNIFY_MAP,
            }
        }
    )
    norm = _build_normalizer(cfg, str(tmp_path))
    assert isinstance(norm, _UnifyAwareNormalizer)
    inner = Normalizer(mode="min_max", stats=_eef_stats_min_max())
    raw = _clipped_raw(7, n=4)
    unified, _ = map_to_unify(inner.normalize(raw), _unify_dst(), UNIFY_DIM)
    np.testing.assert_allclose(norm.unnormalize(unified), raw, atol=1e-5)


def test_build_normalizer_unify_identity_map_fallback(tmp_path):
    """unify on but no unify_action_map: raw dim inferred from stats (20) -> identity map 0..19."""
    _write_stats_file(tmp_path, mode_key="eef")
    cfg = OmegaConf.create(
        {"dataloader": {"normalize_mode": "min-max", "action_mode": "eef", "unify_action": True}}
    )
    norm = _build_normalizer(cfg, str(tmp_path))
    assert isinstance(norm, _UnifyAwareNormalizer)
    assert norm._dst_index.tolist() == list(range(20))


def test_build_normalizer_unify_no_map_no_stats_raises(tmp_path):
    """unify on, no map, normalize disabled (no stats to infer raw dim) -> ValueError."""
    cfg = OmegaConf.create(
        {"dataloader": {"normalize_mode": None, "action_mode": "eef", "unify_action": True}}
    )
    with pytest.raises(ValueError, match="unify_action_map is missing"):
        _build_normalizer(cfg, str(tmp_path))


# --- base_proprio_velocity: proprio IN carries [arm_raw, base_vel3] -> scatter base_vel to [68:71) ---


def _base_vel_stats_min_max():
    lo = np.array([-0.2, -0.2, -0.3], dtype=np.float32)
    hi = np.array([0.2, 0.2, 0.3], dtype=np.float32)
    return {"mean": (lo + hi) / 2, "std": np.maximum((hi - lo) / 4, 1e-6), "min": lo, "max": hi, "q01": lo, "q99": hi}


def _write_stats_file_with_base_vel(tmp_path):
    stats = {"eef": _eef_stats_min_max(), "base_vel": _base_vel_stats_min_max(), "num_timesteps": 1000}
    p = tmp_path / "normalization_stats.npy"
    np.save(str(p), stats, allow_pickle=True)
    return str(p)


def test_unify_proprio_in_with_base_velocity():
    """proprio IN, base_proprio_velocity: raw [arm20, base_vel3] -> normalize arm + scatter to arm
    slots, AND normalize base_vel with its own stats + scatter to [68:71). The dual of the action
    base command ([68:73) gather on OUT)."""
    from openwam.dataloader.robocasa365 import _UNIFY_BASE_VEL

    inner = Normalizer(mode="min_max", stats=_eef_stats_min_max())
    bv_norm = Normalizer(mode="min_max", stats=_base_vel_stats_min_max())
    dst = _unify_dst()
    w = _UnifyAwareNormalizer(inner, dst, UNIFY_DIM, base_vel_dst=_UNIFY_BASE_VEL, base_vel_normalizer=bv_norm)
    arm = _clipped_raw(5, n=2)                          # (2, 20) raw arm
    bv = np.array([[0.1, -0.05, 0.2], [0.0, 0.1, -0.1]], np.float32)  # raw base velocity
    out = w.normalize(np.concatenate([arm, bv], axis=-1))            # (2, 23) -> (2, 80)
    assert out.shape == (2, UNIFY_DIM)
    expected_arm, _ = map_to_unify(inner.normalize(arm), dst, UNIFY_DIM)  # arm-only forward
    np.testing.assert_allclose(out[..., dst], expected_arm[..., dst], atol=1e-6)   # arm slots match
    np.testing.assert_allclose(out[..., _UNIFY_BASE_VEL], bv_norm.normalize(bv), atol=1e-6)  # [68:71) = norm bv


def test_build_normalizer_base_proprio_velocity_scatters(tmp_path):
    """cfg base_proprio_velocity=True + a 'base_vel' stats block -> _build_normalizer wires a base_vel
    scatter so normalize([arm20, base_vel3]) fills [68:71) with the normalized velocity."""
    from openwam.dataloader.robocasa365 import _UNIFY_BASE_VEL

    _write_stats_file_with_base_vel(tmp_path)
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "eef",
                "unify_action": True,
                "unify_action_map": _UNIFY_MAP,
                "base_proprio_velocity": True,
            }
        }
    )
    norm = _build_normalizer(cfg, str(tmp_path))
    assert isinstance(norm, _UnifyAwareNormalizer)
    arm = _clipped_raw(9, n=2)
    bv = np.array([[0.1, -0.05, 0.2], [0.0, 0.1, -0.1]], np.float32)
    out = norm.normalize(np.concatenate([arm, bv], axis=-1))  # (2, 23) -> (2, 80)
    assert out.shape == (2, UNIFY_DIM)
    inner_bv = Normalizer(mode="min_max", stats=_base_vel_stats_min_max())
    np.testing.assert_allclose(out[..., _UNIFY_BASE_VEL], inner_bv.normalize(bv), atol=1e-6)


def test_build_normalizer_base_proprio_velocity_missing_block_raises(tmp_path):
    """base_proprio_velocity=True but no 'base_vel' block in stats -> raise (no silent fallback)."""
    _write_stats_file(tmp_path, mode_key="eef")  # eef only, no base_vel
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "eef",
                "unify_action": True,
                "unify_action_map": _UNIFY_MAP,
                "base_proprio_velocity": True,
            }
        }
    )
    with pytest.raises(ValueError, match="base_vel"):
        _build_normalizer(cfg, str(tmp_path))
