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


def test_build_normalizer_unknown_mode_raises(tmp_path):
    # An ACTIVE-but-unrecognized normalize_mode must fail fast, not silently disable
    # the normalizer (that would send the model's normalized outputs to the robot as
    # physical commands).
    _write_stats_file(tmp_path)
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "bogus", "action_mode": "eef"}})
    with pytest.raises(ValueError, match="not a recognized mode"):
        _build_normalizer(cfg, str(tmp_path))


def test_build_normalizer_wrong_action_mode_raises(tmp_path):
    # Stats file has only "eef" but config says action_mode="joint": the requested
    # key is missing → refuse to deploy with normalization silently disabled.
    _write_stats_file(tmp_path, mode_key="eef")
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "joint"}})
    with pytest.raises(KeyError, match="no 'joint' entry"):
        _build_normalizer(cfg, str(tmp_path))


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
            "inference": {"denoise_steps": 2, "denoise_mode": "sync", "shift": 5.0},
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


def test_unify_supports_distinct_action15_and_state19_maps():
    def flat(dim, low, high):
        lo = np.full(dim, low, np.float32)
        hi = np.full(dim, high, np.float32)
        return {
            "mean": (lo + hi) / 2,
            "std": np.ones(dim, np.float32),
            "min": lo,
            "max": hi,
            "q01": lo,
            "q99": hi,
        }

    action_stats = flat(15, -1.0, 1.0)
    state_stats = flat(19, -2.0, 2.0)
    inner = Normalizer(mode="min_max", stats={**action_stats, "_normalize_stats": state_stats})
    action_dst = parse_unify_spec(["0-9", "68-72"], UNIFY_DIM)
    state_dst = parse_unify_spec(["0-9", "68-76"], UNIFY_DIM)
    wrapper = _UnifyAwareNormalizer(inner, action_dst, UNIFY_DIM, state_dst_index=state_dst)

    raw_state = np.linspace(-2, 2, 19, dtype=np.float32)[None]
    expected_state, _ = map_to_unify(inner.normalize(raw_state), state_dst, UNIFY_DIM)
    np.testing.assert_allclose(wrapper.normalize(raw_state), expected_state, atol=1e-6)

    raw_action = np.linspace(-1, 1, 15, dtype=np.float32)[None]
    unified_action, _ = map_to_unify(raw_action, action_dst, UNIFY_DIM)
    np.testing.assert_allclose(wrapper.unnormalize(unified_action), raw_action, atol=1e-6)


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
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "eef", "unify_action": False}})
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
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": "min-max", "action_mode": "eef", "unify_action": True}})
    norm = _build_normalizer(cfg, str(tmp_path))
    assert isinstance(norm, _UnifyAwareNormalizer)
    assert norm._dst_index.tolist() == list(range(20))


def test_build_normalizer_unify_no_map_no_stats_raises(tmp_path):
    """unify on, no map, normalize disabled (no stats to infer raw dim) -> ValueError."""
    cfg = OmegaConf.create({"dataloader": {"normalize_mode": None, "action_mode": "eef", "unify_action": True}})
    with pytest.raises(ValueError, match="unify_action_map is missing"):
        _build_normalizer(cfg, str(tmp_path))


# --- mobile (25-D [arm20, base5]): the base is folded INTO the raw vector, mapped via ONE map, and
#     un-normalized with ONE combined 'eef_base' stats block — the deploy normalizer is fully generic
#     (no base special-casing). Verifies the wayrise convergence (base is no longer a bypass channel). ---

_UNIFY_MAP_MOBILE = ["0-9", "32-41", "68-72"]  # raw 25-D [arm20, base5] -> unified [0:10)+[32:42)+[68:73)


def _base5_stats_min_max():
    lo = np.array([-1.0, -1.0, -1.0, 0.0, -1.0], dtype=np.float32)  # x/y/yaw vel, torso, control_mode
    hi = np.array([1.0, 1.0, 1.0, 0.34, 1.0], dtype=np.float32)
    return {"mean": (lo + hi) / 2, "std": np.maximum((hi - lo) / 4, 1e-6), "min": lo, "max": hi, "q01": lo, "q99": hi}


def _eef_base_stats_min_max():
    """Combined 25-D stats = 20-D arm ++ 5-D base command (what the mobile reader/deploy persist)."""
    a, b = _eef_stats_min_max(), _base5_stats_min_max()
    return {k: np.concatenate([a[k], b[k]]).astype(np.float32) for k in ("mean", "std", "min", "max", "q01", "q99")}


def _write_stats_file_eef_base(tmp_path):
    stats = {"eef_base": _eef_base_stats_min_max(), "num_timesteps": 1000}
    p = tmp_path / "normalization_stats.npy"
    np.save(str(p), stats, allow_pickle=True)
    return str(p)


def _unify_dst_mobile():
    return parse_unify_spec(_UNIFY_MAP_MOBILE, UNIFY_DIM)


def _clipped_raw25(seed: int, n: int = 2):
    """A 25-D raw [arm20, base5] clipped into the combined stats range (clean round-trip)."""
    arm = _clipped_raw(seed, n)
    b = _base5_stats_min_max()
    base = np.random.RandomState(seed + 100).uniform(b["min"], b["max"], size=(n, 5)).astype(np.float32)
    return np.concatenate([arm, base], axis=-1)


def test_unify_action_out_with_mobile_base():
    """action OUT, mobile: the model's 80-D unified action is gathered back to the 25-D [arm20, base5]
    raw vector via the ONE map, then un-normalized with the single combined 'eef_base' stats — no base
    special-casing (generic _UnifyAwareNormalizer). mobile_base: true is the robocasa365.yaml default."""
    inner = Normalizer(mode="min_max", stats=_eef_base_stats_min_max())  # 25-D combined
    dst = _unify_dst_mobile()
    raw25 = _clipped_raw25(3)
    unified, _ = map_to_unify(inner.normalize(raw25), dst, UNIFY_DIM)  # what the model is trained on
    out = _UnifyAwareNormalizer(inner, dst, UNIFY_DIM).unnormalize(unified)  # (2, 80) → (2, 25)
    assert out.shape == (2, 25)  # [arm20, base5]
    np.testing.assert_allclose(out, raw25, atol=1e-5)  # whole vector round-trips (arm + base)


def test_unify_proprio_in_mobile_roundtrip():
    """proprio IN, mobile: wrapper.normalize(raw25) == map_to_unify(inner.normalize(raw25)) — the base
    velocity (raw slots 20:23) scatters through the SAME map, no split-off special-casing."""
    inner = Normalizer(mode="min_max", stats=_eef_base_stats_min_max())
    dst = _unify_dst_mobile()
    raw25 = _clipped_raw25(1, n=3)
    expected, _ = map_to_unify(inner.normalize(raw25), dst, UNIFY_DIM)
    np.testing.assert_allclose(_UnifyAwareNormalizer(inner, dst, UNIFY_DIM).normalize(raw25), expected, atol=1e-6)


def test_build_normalizer_mobile_base_unnormalizes(tmp_path):
    """cfg action_mode='eef_base' + the 25-D mobile map → generic _UnifyAwareNormalizer; unnormalize(80)
    returns the 25-D [arm20, base5] raw vector the client bridges (arm→OSC, base5 direct)."""
    _write_stats_file_eef_base(tmp_path)
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "eef_base",
                "unify_action": True,
                "unify_action_map": _UNIFY_MAP_MOBILE,
                "mobile_base": True,
            }
        }
    )
    norm = _build_normalizer(cfg, str(tmp_path))
    assert isinstance(norm, _UnifyAwareNormalizer)
    inner = Normalizer(mode="min_max", stats=_eef_base_stats_min_max())
    raw25 = _clipped_raw25(7)
    unified, _ = map_to_unify(inner.normalize(raw25), _unify_dst_mobile(), UNIFY_DIM)
    out = norm.unnormalize(unified)
    assert out.shape == (2, 25)
    np.testing.assert_allclose(out, raw25, atol=1e-5)


def test_build_normalizer_mobile_base_missing_block_raises(tmp_path):
    """action_mode='eef_base' but the stats file only has 'eef' → KeyError (no silent disable): the
    combined 25-D block is required, so a stale arm-only file must fail loud, not serve un-normalized."""
    _write_stats_file(tmp_path, mode_key="eef")  # eef only, no 'eef_base'
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "eef_base",
                "unify_action": True,
                "unify_action_map": _UNIFY_MAP_MOBILE,
                "mobile_base": True,
            }
        }
    )
    with pytest.raises(KeyError, match="no 'eef_base' entry"):
        _build_normalizer(cfg, str(tmp_path))


# --- _CommandAwareNormalizer (binary_action_dims + base_proprio=global_pose) ---


def _eef_base_stats_25d(base_lo=None, base_hi=None):
    """25-D [arm20, base5] stats: the 20-D arm block + a base5 block (default the ±1 command range).
    The gripper dim (9) is pinned to the [-1, +1] command range, like the production stats
    (_pin_gripper_stats), so its min-max normalization is the identity."""
    arm = {k: np.array(v, np.float32) for k, v in _eef_stats_min_max().items()}
    for k, v in (("min", -1.0), ("max", 1.0), ("q01", -1.0), ("q99", 1.0), ("mean", 0.0), ("std", 1.0)):
        arm[k][9] = v
    lo5 = np.asarray(base_lo if base_lo is not None else [-1.0] * 5, np.float32)
    hi5 = np.asarray(base_hi if base_hi is not None else [1.0] * 5, np.float32)
    return {
        "mean": np.concatenate([arm["mean"], (lo5 + hi5) / 2]).astype(np.float32),
        "std": np.concatenate([arm["std"], np.maximum((hi5 - lo5) / 4, 1e-6)]).astype(np.float32),
        "min": np.concatenate([arm["min"], lo5]).astype(np.float32),
        "max": np.concatenate([arm["max"], hi5]).astype(np.float32),
        "q01": np.concatenate([arm["q01"], lo5]).astype(np.float32),
        "q99": np.concatenate([arm["q99"], hi5]).astype(np.float32),
    }


def test_command_aware_binary_snap(tmp_path):
    """binary_action_dims: the decoded output is snapped to exact ±1 at threshold 0.5 — the same
    boundary the downstream bridge (confident-close >0.5) and env (control_mode >=0.5) apply, so an
    uncertain mid-range output still lands on the safe side (open / arm mode)."""
    stats = {"eef_base": _eef_base_stats_25d(), "num_timesteps": 10}
    np.save(str(tmp_path / "normalization_stats.npy"), stats, allow_pickle=True)
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "eef_base",
                "binary_action_dims": [9, 24],
            }
        }
    )
    normalizer = _build_normalizer(cfg, str(tmp_path))
    x = np.zeros(25, dtype=np.float32)
    x[9], x[24] = 0.6, 0.4  # gripper leaning close; mode below threshold
    out = normalizer.unnormalize(x)
    assert out[9] == 1.0 and out[24] == -1.0
    x[9], x[24] = 0.5, 0.51  # 0.5 exactly is NOT confident -> -1 (strict >)
    out = normalizer.unnormalize(x)
    assert out[9] == -1.0 and out[24] == 1.0
    # non-binary dims untouched by the snap
    assert out[20] == pytest.approx(0.0, abs=1e-6)


def test_command_aware_global_pose_split_stats(tmp_path):
    """base_proprio='global_pose': proprio normalizes with the 'eef_base_pose_proprio' block (pose
    range), actions keep un-normalizing with the 'eef_base' command block."""
    pose_block = _eef_base_stats_25d(base_lo=[-10.0, -10.0, -1.0, -1.0, 0.0], base_hi=[10.0, 10.0, 1.0, 1.0, 1.0])
    stats = {"eef_base": _eef_base_stats_25d(), "eef_base_pose_proprio": pose_block, "num_timesteps": 10}
    np.save(str(tmp_path / "normalization_stats.npy"), stats, allow_pickle=True)
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "eef_base",
                "base_proprio": "global_pose",
            }
        }
    )
    normalizer = _build_normalizer(cfg, str(tmp_path))
    raw = np.zeros(25, dtype=np.float32)
    raw[20] = 5.0  # x = 5 m -> min-max over [-10, 10] -> 0.5 under the POSE block
    assert normalizer.normalize(raw)[20] == pytest.approx(0.5, abs=1e-5)
    # action direction: normalized 0.5 on a command dim (±1 range) -> physical 0.5, NOT 5 m
    a = np.zeros(25, dtype=np.float32)
    a[20] = 0.5
    assert normalizer.unnormalize(a)[20] == pytest.approx(0.5, abs=1e-5)


def test_command_aware_missing_pose_block_raises(tmp_path):
    stats = {"eef_base": _eef_base_stats_25d(), "num_timesteps": 10}
    np.save(str(tmp_path / "normalization_stats.npy"), stats, allow_pickle=True)
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "eef_base",
                "base_proprio": "global_pose",
            }
        }
    )
    with pytest.raises(KeyError, match="eef_base_pose_proprio"):
        _build_normalizer(cfg, str(tmp_path))


def test_command_aware_null_normalize_is_supported(tmp_path):
    """normalize_mode=null + binary/global_pose is a CONSISTENT config (training allows it): every
    dim serves in raw space (binary targets are raw ±1 by construction, pose proprio is raw meters)
    and the legality projection lives in WAMPolicy, independent of the normalizer. Deploy must
    return None, not raise — a trainable config must be deployable."""
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": None,
                "action_mode": "eef_base",
                "binary_action_dims": [9, 24],
                "base_proprio": "global_pose",
            }
        }
    )
    assert _build_normalizer(cfg, str(tmp_path)) is None  # no stats file needed either


def test_command_aware_binary_survives_nonneutral_zscore(tmp_path):
    """THE B1 regression (wayrise): binary targets bypassed normalization at train time, so the model
    emits them in raw ±1 space — the stats inverse must NOT touch them. With class-imbalanced z-score
    stats (control_mode +1 ≈ 7% → mean≈-0.86, std≈0.51) the old post-inverse judgment turned a
    correct +1 into (1·0.51 - 0.86) = -0.35 → snapped to the WRONG class."""
    stats = _eef_base_stats_25d()
    for d in (9, 24):
        stats["mean"][d], stats["std"][d] = -0.86, 0.51
    np.save(str(tmp_path / "normalization_stats.npy"), {"eef_base": stats, "num_timesteps": 10}, allow_pickle=True)
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "z-score",
                "action_mode": "eef_base",
                "binary_action_dims": [9, 24],
            }
        }
    )
    normalizer = _build_normalizer(cfg, str(tmp_path))
    x = np.zeros(25, dtype=np.float32)
    x[9], x[24] = 1.0, 1.0
    out = normalizer.unnormalize(x)
    assert out[9] == 1.0 and out[24] == 1.0  # class preserved (old code: -1)
    x[9], x[24] = -1.0, -1.0
    out = normalizer.unnormalize(x)
    assert out[9] == -1.0 and out[24] == -1.0
    # continuous dims still go through the stats inverse (z-score: x*std + mean)
    x2 = np.zeros(25, dtype=np.float32)
    x2[20] = 1.0
    assert _build_normalizer(cfg, str(tmp_path)).unnormalize(x2)[20] == pytest.approx(
        1.0 * stats["std"][20] + stats["mean"][20], abs=1e-5
    )


def test_command_aware_binary_zscore_under_unify(tmp_path):
    """Same regression through the unify wrapper: gather 80→25 happens BEFORE the command-aware
    inner, so the pre-inverse judgment must still see the raw ±1 (unified dim 72 → raw 24)."""
    stats = _eef_base_stats_25d()
    for d in (9, 24):
        stats["mean"][d], stats["std"][d] = -0.86, 0.51
    np.save(str(tmp_path / "normalization_stats.npy"), {"eef_base": stats, "num_timesteps": 10}, allow_pickle=True)
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "z-score",
                "action_mode": "eef_base",
                "binary_action_dims": [9, 24],
                "unify_action": True,
                "unify_action_map": ["0-9", "34-43", "68-72"],
            }
        }
    )
    normalizer = _build_normalizer(cfg, str(tmp_path))
    u = np.zeros(80, dtype=np.float32)
    u[9], u[72] = 1.0, -1.0  # unified: l_grip at 9, control_mode at 72
    out = normalizer.unnormalize(u)
    assert out.shape[-1] == 25
    assert out[9] == 1.0 and out[24] == -1.0
