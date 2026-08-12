"""Smoke tests for DualSystem IDM (Inverse Dynamics Model) architecture.

Tests the IDM-specific features:
  - Teacher-forcing attention mask layout
  - IDM architecture construction and registration
  - ActionDiT pre/post_attn round-trip under IDM
  - IDMMoTDriver teacher-forcing mask correctness
"""

from unittest.mock import MagicMock

import torch
import torch.nn as nn


def _make_idm(dim=64, video_dim=64):
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "idm",
        "action_dim": 7,
        "dim": dim,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": video_dim,
        "bridge_layers": (0, 2),
        "idm_video_cond_noise_prob": 0.5,
    }
    return build_architecture("dual_system_idm", cfg)


def test_idm_registered():
    """IDM variant should be in the supported architecture registry."""
    from openwam.model import list_supported_architectures

    supported = list_supported_architectures()
    assert "dual_system_idm" in supported


def test_idm_construction():
    """IDM architecture constructs with expected components."""
    arch = _make_idm()
    assert arch.action_backbone is not None
    assert arch.action_backbone.variant == "idm"
    assert hasattr(arch, "_mot_driver")
    assert hasattr(arch, "video_cond_noise_prob")
    assert arch.video_cond_noise_prob == 0.5


def test_idm_rejects_out_of_range_cond_noise_prob():
    """idm_video_cond_noise_prob outside [0, 1] silently breaks the Bernoulli gate."""
    import pytest

    from openwam.model import build_architecture

    base_cfg = {
        "framework": "dual_system",
        "variant": "idm",
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 64,
        "bridge_layers": (0, 2),
    }
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="idm_video_cond_noise_prob must be in"):
            build_architecture("dual_system_idm", {**base_cfg, "idm_video_cond_noise_prob": bad})


def test_idm_legacy_cond_noise_key_still_works():
    """Keep the old unprefixed key working while the config migrates."""
    from openwam.model import build_architecture

    arch = build_architecture(
        "dual_system_idm",
        {
            "framework": "dual_system",
            "variant": "idm",
            "action_dim": 7,
            "dim": 64,
            "ffn_dim": 128,
            "num_heads": 4,
            "video_dim": 64,
            "bridge_layers": (0, 2),
            "video_cond_noise_prob": 0.25,
        },
    )
    assert arch.video_cond_noise_prob == 0.25


def test_idm_ignores_attention_mask_mode():
    """IDM owns its teacher-forcing / Stage-2 masks and ignores attention_mask_mode.

    A stale cross-modal mode (previously a hard error) must now construct fine
    and be dropped — never forwarded to the driver kwargs.
    """
    from openwam.model import build_architecture

    arch = build_architecture(
        "dual_system_idm",
        {
            "framework": "dual_system",
            "variant": "idm",
            "action_dim": 7,
            "dim": 64,
            "ffn_dim": 128,
            "num_heads": 4,
            "video_dim": 64,
            "bridge_layers": (0, 2),
            "attention_mask_mode": "mutual",
        },
    )
    assert "attention_mask_mode" not in arch._mot_driver_kwargs


def test_idm_action_dit_roundtrip():
    """ActionDiT pre/post_attn round-trip works under IDM architecture."""
    arch = _make_idm()
    ab = arch.action_backbone
    B, T_action = 2, 5

    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])
    context = torch.randn(B, 4, ab.text_dim)
    context_mask = torch.ones(B, 4, dtype=torch.bool)

    astate = ab.prepare_state(noisy_actions, timestep, context=context, context_mask=context_mask)

    for layer_id in range(ab.num_layers):
        q, k, v, post = ab.pre_attn_at_layer(layer_id, astate)
        assert q.shape == k.shape == v.shape == (B, T_action, ab.num_heads * ab.head_dim)
        attn_out = torch.randn_like(q)
        astate = ab.post_attn_at_layer(layer_id, astate, attn_out, post)

    pred = ab.extract_prediction(astate)
    assert pred.shape == (B, T_action, 7)


def test_idm_teacher_forcing_mask():
    """Teacher-forcing mask has correct layout."""
    from openwam.model.architectures.dual_system.idm import IDMMoTDriver

    # Mock video backbone with a simple bidirectional v2v mask
    vb = MagicMock()
    vb.num_layers = 2
    vb.num_heads = 4
    vb.head_dim = 16
    vb.dim = 64

    def mock_v2v_mask(video_seq_len, video_tokens_per_frame, device):
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    vb.build_video_to_video_mask = mock_v2v_mask
    vb.video_attention_mask_mode = "bidirectional"

    ab = MagicMock()
    ab.num_layers = 2
    ab.num_heads = 4
    ab.head_dim = 16
    ab.training = False

    driver = IDMMoTDriver(vb, ab, attention_mask_mode="action_sees_video")

    s_noisy = 10
    s_cond = 10
    s_action = 5
    mask = driver._build_teacher_forcing_mask(
        s_noisy_video=s_noisy,
        s_cond_video=s_cond,
        s_action=s_action,
        video_tokens_per_frame=5,
        device=torch.device("cpu"),
    )

    total = s_noisy + s_cond + s_action
    assert mask.shape == (total, total)

    # Check noisy_video ↔ noisy_video (True for bidirectional)
    assert mask[:s_noisy, :s_noisy].all()
    # Check cond_video ↔ cond_video (True for bidirectional)
    assert mask[s_noisy : s_noisy + s_cond, s_noisy : s_noisy + s_cond].all()
    # Check action ↔ action (True)
    assert mask[s_noisy + s_cond :, s_noisy + s_cond :].all()
    # Check action → cond_video (True)
    assert mask[s_noisy + s_cond :, s_noisy : s_noisy + s_cond].all()

    # Check blocked paths
    # noisy_video → cond_video (False)
    assert not mask[:s_noisy, s_noisy : s_noisy + s_cond].any()
    # noisy_video → action (False)
    assert not mask[:s_noisy, s_noisy + s_cond :].any()
    # cond_video → noisy_video (False)
    assert not mask[s_noisy : s_noisy + s_cond, :s_noisy].any()
    # cond_video → action (False)
    assert not mask[s_noisy : s_noisy + s_cond, s_noisy + s_cond :].any()
    # action → noisy_video (False)
    assert not mask[s_noisy + s_cond :, :s_noisy].any()


def test_idm_resolve_from_framework_variant():
    """IDM can be resolved from (framework='dual_system', variant='idm')."""
    from openwam.model.architectures.registry import _FRAMEWORK_VARIANT_INDEX

    assert ("dual_system", "idm") in _FRAMEWORK_VARIANT_INDEX
    assert _FRAMEWORK_VARIANT_INDEX[("dual_system", "idm")] == "dual_system_idm"


class _CapturePrepareVideoBackbone(nn.Module):
    """Tiny video backbone that records per-prepare timesteps for IDM loss tests."""

    dim = 8
    num_layers = 1
    num_heads = 2
    head_dim = 4
    submodule_names = []

    def __init__(self):
        super().__init__()
        from tests.test_openwam_trainer import _MockScheduler

        self.scheduler = _MockScheduler()
        self.prepare_timesteps = []
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    @property
    def video_attention_mask_mode(self):
        return "bidirectional"

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode):
        del mode

    def set_dtype_device(self, dtype, device):
        self._dtype = dtype
        self._device = device

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    def prepare(self, **kw):
        from openwam.model.video_backbone.base import BlockLoopState

        latents = kw["latents"]
        timestep = kw["timestep"]
        self.prepare_timesteps.append(timestep.detach().clone())
        B, _, f, h, w = latents.shape
        s = f * h * w
        x = latents[:, :1].reshape(B, s, 1).expand(B, s, self.dim).contiguous()
        freqs = torch.polar(torch.ones(s, 1, self.head_dim // 2), torch.zeros(s, 1, self.head_dim // 2))
        t_mod = timestep.view(B, 1, 1, 1).expand(B, s, 6, self.dim).contiguous()
        context = torch.zeros(B, 2, self.dim)
        context_mask = torch.ones(B, 2, dtype=torch.bool)
        return BlockLoopState(
            hidden_states=x,
            time_mod=t_mod,
            rope_freqs=freqs,
            context=context,
            context_mask=context_mask,
            grid_frames=f,
            grid_height=h,
            grid_width=w,
        )

    def merge_idm_video_branches(self, noisy, cond):
        """Drive the *production* Wan merge (flat (B, L, D), 4D time_mod, rope_freqs,
        VACE hints) rather than a copy, so its validation stays covered. ``WanBase``'s
        implementation touches only the two states (never ``self``), so an unbound
        call on this fake is exact."""
        from openwam.model.video_backbone.wan_backbone import WanBase

        return WanBase.merge_idm_video_branches(self, noisy, cond)

    def split_idm_video_branches(self, merged, noisy, cond):
        from openwam.model.video_backbone.wan_backbone import WanBase

        return WanBase.split_idm_video_branches(self, merged, noisy, cond)

    def pre_attn_at_layer(self, layer_id, state):
        del layer_id
        return state.hidden_states, state.hidden_states, state.hidden_states, {"residual": state.hidden_states}

    def post_attn_at_layer(self, layer_id, state, attn_out, post_state):
        del layer_id
        state.hidden_states = post_state["residual"] + attn_out
        return state

    def run_block(self, block_id, state):
        del block_id
        state.hidden_states = state.hidden_states + 1
        return state

    def finalize(self, state):
        B = state.hidden_states.shape[0]
        return (
            state.hidden_states[:, :, :1]
            .transpose(1, 2)
            .reshape(B, 1, state.grid_frames, state.grid_height, state.grid_width)
        )

    def decode_video(self, latents, *, tiled=True):
        del latents, tiled
        return None


def _make_idm_with_video(vb, *, action_dim=3, text_dim=8):
    from openwam.model.architectures.dual_system.idm import DualSystemIDMArchitecture

    cfg = {
        "framework": "dual_system",
        "variant": "idm",
        "action_dim": action_dim,
        "dim": 8,
        "ffn_dim": 16,
        "num_heads": 2,
        "attn_head_dim": 4,
        "video_dim": 8,
        "text_dim": text_dim,
        "bridge_layers": (0,),
        "idm_video_cond_noise_prob": 0.0,
        "mot_checkpoint_mixed_attn": False,
    }
    arch = DualSystemIDMArchitecture(cfg)
    arch.video_backbone = vb
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    arch.build_mot_driver()
    return arch


def test_idm_clean_cond_video_uses_zero_timestep():
    """Clean teacher-forcing cond video must use literal t=0, not scheduler index 0."""
    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.init_training_schedulers(10)

    result = arch.compute_loss(
        input_latents=torch.randn(2, 1, 1, 1, 1),
        context=torch.randn(2, 2, arch.action_backbone.text_dim),
        context_mask=torch.ones(2, 2, dtype=torch.bool),
        actions=torch.randn(2, 3, 3),
        lambda_video=1.0,
        lambda_action=1.0,
    )

    assert torch.isfinite(result["loss"])
    assert len(vb.prepare_timesteps) >= 2
    cond_timesteps = vb.prepare_timesteps[1]
    assert torch.equal(cond_timesteps, torch.zeros_like(cond_timesteps))


def test_idm_training_requires_tokenwise_video_t_mod():
    """IDM should fail loudly if a backbone cannot represent noisy/cond timesteps in one sequence.

    The check now lives in the backbone's ``merge_idm_video_branches`` (the
    driver delegates branch concatenation to the backbone), so a non-4D
    ``time_mod`` must raise there and propagate through the driver loop.
    """
    from openwam.model.video_backbone.base import BlockLoopState

    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb)
    driver = arch._mot_driver

    # 3D time_mod (not the token-wise 4D shape IDM requires).
    vstate = BlockLoopState(
        hidden_states=torch.zeros(1, 2, vb.dim),
        time_mod=torch.zeros(1, 6, vb.dim),
        rope_freqs=torch.zeros(2, 1, 2),
        context=torch.zeros(1, 1, vb.dim),
        grid_frames=2,
        grid_height=1,
        grid_width=1,
    )
    astate = MagicMock()
    astate.payload.x_action = torch.zeros(1, 1, vb.dim)

    import pytest

    with pytest.raises(ValueError, match="token-wise video t_mod"):
        driver.run_idm_training_loop(vstate, vstate, astate)


class _GenerateVideoBackbone(_CapturePrepareVideoBackbone):
    def __init__(self):
        super().__init__()
        self.prepare_count = 0
        self.prepare_context_lengths = []

    def preprocess_input_for_inference(self, *args, **kwargs):
        del args, kwargs
        return {
            "latents": torch.ones(1, 1, 1, 1, 1),
            "context": torch.randn(1, 2, self.dim),
            "context_mask": torch.ones(1, 2, dtype=torch.bool),
        }

    def prepare(self, **kw):
        self.prepare_count += 1
        self.prepare_context_lengths.append(int(kw["context"].shape[1]))
        return super().prepare(**kw)

    def finalize(self, state):
        B = state.hidden_states.shape[0]
        return torch.zeros(B, 1, state.grid_frames, state.grid_height, state.grid_width)


def test_idm_generate_honors_action_num_frames_and_prefills_video_once(monkeypatch):
    """OpenWAM deploy passes raw action_num_frames separately from video_num_frames."""
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    vb = _GenerateVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.eval()

    result = arch.generate(
        schedule=[(1.0, 1.0), (0.0, 0.5), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
    )

    assert result["video"] is None
    assert result["actions"].shape == (4, 3)
    # One prepare in stage 1 and one video-cache prefill prepare before action denoising.
    assert vb.prepare_count == 2


def test_idm_generate_runs_attached_normalizer_on_actions(monkeypatch):
    """IDM's own generate() must unnormalize the output via the attached normalizer.

    Regression: the action_normalizer -> normalizer rename left idm.py reading a
    stale getattr key, silently skipping unnormalization for IDM deploys.
    """
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)

    class _AddConstNormalizer:
        def unnormalize(self, x):
            return x + 100.0

    torch.manual_seed(0)
    arch = _make_idm_with_video(_GenerateVideoBackbone())
    arch.eval()
    gen_kwargs = dict(
        schedule=[(1.0, 1.0), (0.0, 0.5), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
    )

    # Same global + per-call seed → identical denoised actions; only the normalizer differs.
    torch.manual_seed(42)
    raw = arch.generate(**gen_kwargs)["actions"]
    arch.normalizer = _AddConstNormalizer()
    torch.manual_seed(42)
    normalized = arch.generate(**gen_kwargs)["actions"]

    torch.testing.assert_close(torch.from_numpy(normalized), torch.from_numpy(raw) + 100.0)


def test_idm_generate_pins_inactive_unified_action_dims(monkeypatch):
    """IDM's stage-2 loop must keep unsupervised unified dims on the analytic
    sigma * eps0 path, exactly like BaseWAMArchitecture.generate."""
    import numpy as np

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    gen_kwargs = dict(
        schedule=[(1000.0, 1000.0), (0.0, 500.0), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
    )

    arch = _make_idm_with_video(_GenerateVideoBackbone())
    arch.eval()
    torch.manual_seed(42)
    baseline = arch.generate(**gen_kwargs)["actions"]
    # The randomly initialized ActionDiT emits nonzero flow on every channel,
    # so unpinned inactive dims do NOT land on 0.
    assert np.abs(baseline[:, 1:]).max() > 0.0

    torch.manual_seed(42)
    pinned = arch.generate(**gen_kwargs, active_action_mask=torch.tensor([True, False, False]))["actions"]
    # sigma_end == 0 -> pinned inactive dims land exactly on 0; the active dim
    # still integrates real flow.
    assert np.abs(pinned[:, 1:]).max() == 0.0
    assert np.abs(pinned[:, 0]).max() > 0.0


def test_idm_generate_with_proprio_appends_context_once(monkeypatch):
    """IDM generate appends proprio once even though stage 2 bypasses forward()."""
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    vb = _GenerateVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch._use_proprioception_context = True
    arch.proprio_dim = 3
    arch.context_dim = arch.action_backbone.text_dim
    arch.proprio_encoder = nn.Linear(3, arch.context_dim)
    arch.eval()

    result = arch.generate(
        schedule=[(1.0, 1.0), (0.0, 0.5), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
        proprio=torch.randn(1, 3),
    )

    assert result["video"] is None
    assert result["actions"].shape == (4, 3)
    assert vb.prepare_count == 2
    assert vb.prepare_timesteps[0].shape == torch.Size([1])
    assert vb.prepare_context_lengths == [3, 3]


class _ReuseOnceVideoCache:
    """Tiny cache mock that recomputes twice, then reuses the previous velocity."""

    def __init__(self):
        self.should_calls = []
        self.update_calls = []
        self.get_cached_calls = 0
        self._cached = None

    def should_recompute(self, sigma):
        self.should_calls.append(float(sigma))
        return len(self.should_calls) < 3

    def update(self, velocity, sigma):
        self.update_calls.append(float(sigma))
        self._cached = velocity.detach()

    def get_cached(self):
        self.get_cached_calls += 1
        return self._cached


def test_idm_video_cache_matches_joint_loop():
    """The cached Stage-2 path must equal a full joint loop on the action output.

    Stage 2 reuses ``prefill_video_cache`` + ``run_action_with_video_cache``
    instead of running joint attention every action step. That's only valid
    when the joint mask blocks ``v→a`` — see
    ``DualSystemMoTDriver._build_attention_mask``. If that invariant ever regresses
    (or the cache layout drifts from what the joint K/V would be), the
    cached action output will diverge from the joint action output. This
    test pins the equivalence numerically.
    """
    import copy as _copy

    torch.manual_seed(0)
    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.eval()
    driver = arch._mot_driver

    B = 1
    latents = torch.randn(B, 1, 1, 1, 1)
    vstate_joint = vb.prepare(latents=latents, timestep=torch.zeros(B))
    vstate_cache = _copy.copy(vstate_joint)
    vstate_cache.hidden_states = vstate_joint.hidden_states.clone()

    action_latents = torch.randn(B, 3, arch.action_backbone.action_dim)
    a_timestep = torch.tensor([0.5])
    context = torch.randn(B, 2, arch.action_backbone.text_dim)
    context_mask = torch.ones(B, 2, dtype=torch.bool)

    astate_joint = arch.action_backbone.prepare_state(
        action_latents, a_timestep, context=context, context_mask=context_mask
    )
    astate_cache = arch.action_backbone.prepare_state(
        action_latents, a_timestep, context=context, context_mask=context_mask
    )

    _, astate_joint = driver.run_joint_loop(vstate_joint, astate_joint)
    pred_joint = arch.action_backbone.extract_prediction(astate_joint)

    video_kv_cache, _, _ = driver.prefill_video_cache(vstate_cache)
    astate_cache = driver.run_action_with_video_cache(
        astate_cache,
        video_kv_cache=video_kv_cache,
        video_seq_len=int(vstate_cache.hidden_states.shape[1]),
    )
    pred_cache = arch.action_backbone.extract_prediction(astate_cache)

    assert torch.allclose(pred_joint, pred_cache, atol=1e-5, rtol=1e-5)


def test_idm_generate_forwards_the_prefix_gate_to_stage_two(monkeypatch):
    """generate() must hand Stage 2 the gate ``prefill_video_cache`` returned.

    The driver-level tests cover ``build_cached_action_mask`` and
    ``run_action_with_video_cache``; this covers the caller that connects them.
    Setting either forwarding site in ``generate()`` to ``None`` restores the
    original und-padding leak with every other test green, because nothing else
    asserts the value ever leaves ``prefill_video_cache``.
    """
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    vb = _GenerateVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.eval()
    driver = arch._mot_driver

    sentinel = torch.tensor([[True, True, False]])
    seen = {}
    real_prefill = driver.prefill_video_cache
    real_stage2 = driver.run_action_with_video_cache

    def spy_prefill(vstate):
        cache, key_len, _gate = real_prefill(vstate)
        seen["produced"] = sentinel  # stand in for a padded batch's gate
        return cache, key_len, sentinel

    def spy_stage2(astate, *, video_kv_cache, video_seq_len, prefix_kv_mask=None):
        seen["forwarded"] = prefix_kv_mask
        # The real Stage 2 would size its mask from the gate; the fake backbone
        # has no prefix, so just record and run without it.
        return real_stage2(astate, video_kv_cache=video_kv_cache, video_seq_len=video_seq_len)

    monkeypatch.setattr(driver, "prefill_video_cache", spy_prefill)
    monkeypatch.setattr(driver, "run_action_with_video_cache", spy_stage2)

    arch.generate(
        schedule=[(1.0, 1.0), (0.0, 0.5), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
    )

    assert "forwarded" in seen, "Stage 2 was never reached"
    assert seen["forwarded"] is sentinel, (
        "generate() dropped prefill_video_cache's prefix gate instead of forwarding it to Stage 2"
    )


def test_idm_generate_forwards_the_prefix_gate_on_the_compiled_path(monkeypatch):
    """Same contract for the compiled Stage-2 branch, which builds its own mask.

    The eager branch delegates to ``run_action_with_video_cache``; the compiled
    branch calls ``build_cached_action_mask`` directly, so it is a second,
    independently-mutable forwarding site.
    """
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    vb = _GenerateVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.eval()
    driver = arch._mot_driver

    sentinel = torch.tensor([[True, True, False]])
    calls = []
    real_prefill = driver.prefill_video_cache
    real_build = driver.build_cached_action_mask

    def spy_prefill(vstate):
        cache, key_len, _gate = real_prefill(vstate)
        return cache, key_len, sentinel

    def spy_build(*, s_action, video_key_len, device, prefix_kv_mask=None):
        # Record every call: the forced fallback re-enters via the eager branch,
        # which would otherwise overwrite the compiled branch's argument.
        calls.append(prefix_kv_mask)
        return real_build(s_action=s_action, video_key_len=video_key_len, device=device, prefix_kv_mask=None)

    def _boom(*_a, **_k):
        raise RuntimeError("force the documented eager fallback after the mask is built")

    monkeypatch.setattr(driver, "prefill_video_cache", spy_prefill)
    monkeypatch.setattr(driver, "build_cached_action_mask", spy_build)
    arch._compiled_idm_action_cache_loop = _boom

    arch.generate(
        schedule=[(1.0, 1.0), (0.0, 0.5), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
    )

    assert calls, "the compiled Stage-2 branch was never entered"
    assert calls[0] is sentinel, "the compiled path built its action mask without prefill_video_cache's prefix gate"
