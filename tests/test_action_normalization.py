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
from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.model_loader import _build_normalizer
from openwam.model.architectures.architecture_base import BaseWAMArchitecture

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
            has_latent_decoder = False

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


class _LatentDecoderActionBackbone(nn.Module):
    """Stub action backbone exposing a latent->action decoder for generate()."""

    action_dim = 20  # latent token_dim in latent mode

    def __init__(self, num_query, real_action_dim, proprio_dim=20):
        super().__init__()
        self.scheduler = _TinyScheduler()
        self.uses_proprioception = False
        self.num_query = num_query
        self.real_action_dim = real_action_dim
        self.proprio_dim = proprio_dim
        self.seen_proprio = None

    @property
    def has_latent_decoder(self):
        return True

    def decode_latent_to_action(self, latent, proprio=None):
        # Record proprio to assert it was routed in; emit a fixed (B, num_query,
        # real_action_dim) so the test checks shape + unnormalize, not values.
        self.seen_proprio = proprio
        b = latent.shape[0]
        base = torch.arange(self.num_query * self.real_action_dim, dtype=latent.dtype)
        return base.view(1, self.num_query, self.real_action_dim).expand(b, -1, -1)


class _TinyLatentGenerateArchitecture(_TinyGenerateArchitecture):
    def __init__(self, normalizer, num_query, real_action_dim):
        super().__init__(normalizer)
        self.action_backbone = _LatentDecoderActionBackbone(num_query, real_action_dim)


def test_base_generate_latent_decodes_then_unnormalizes():
    """Latent mode: generate() decodes latent->action AND unnormalizes (same path as explicit)."""
    num_query, real_action_dim = 5, 20
    stats = _eef_stats_min_max()
    normalizer = Normalizer(mode="min_max", stats=stats)
    arch = _TinyLatentGenerateArchitecture(normalizer, num_query, real_action_dim)

    proprio = np.zeros(real_action_dim, dtype=np.float32)
    result = arch.generate(
        schedule=[(0.0, 1.0), (0.0, 0.0)],
        prompt="",
        num_frames=2,
        decode_video=False,
        seed=123,
        proprio=torch.from_numpy(proprio),
    )

    # Decoded shape (num_query, real_action_dim), proprio was routed to the decoder.
    assert result["actions"].shape == (num_query, real_action_dim)
    assert arch.action_backbone.seen_proprio is not None
    # Output went through unnormalize (decoder emits an arange; unnormalize maps it).
    raw = torch.arange(num_query * real_action_dim, dtype=torch.float32).view(num_query, real_action_dim).numpy()
    expected = normalizer.unnormalize(raw)
    np.testing.assert_allclose(result["actions"], expected, atol=1e-5)
