"""CPU smoke for CosmosPredict25VideoBackbone.

Builds the wrapper around a hand-rolled fake ``net`` that mimics just the
public surface of upstream ``MinimalV1LVGDiT`` (``prepare_embedded_sequence``,
``t_embedder``, ``t_embedding_norm``, ``blocks``, ``final_layer``,
``unpatchify``, ``crossattn_proj``, ``timestep_scale``,
``concat_padding_mask``, ``use_crossattn_projection``,
``crossattn_proj_in_channels``). Lets us verify the prepare → run_block →
finalize wiring + preprocess_input validation without requiring
``cosmos_predict2`` to be installed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone


class _FakeBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x_B_T_H_W_D,
        t_embedding_B_T_D,
        crossattn_emb,
        *,
        rope_emb_L_1_1_D=None,
        adaln_lora_B_T_3D=None,
        extra_per_block_pos_emb=None,
    ):
        # Add a context-driven gate so cross-attn shape mismatches blow up here.
        ctx_mean = crossattn_emb.mean(dim=1)  # (B, ctx_dim) — we don't strictly check ctx_dim
        return self.proj(x_B_T_H_W_D) + 0.0 * ctx_mean.sum()


class _FakeFinalLayer(nn.Module):
    def __init__(self, dim: int, out_per_patch: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, out_per_patch, bias=False)

    def forward(self, x_B_T_H_W_D, t_embedding_B_T_D, *, adaln_lora_B_T_3D=None):
        return self.linear(x_B_T_H_W_D)


class _FakeMiniDIT(nn.Module):
    """Mimics MinimalV1LVGDiT surface enough for the wrapper to drive it."""

    def __init__(
        self,
        *,
        dim: int = 32,
        num_blocks: int = 4,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        out_channels: int = 16,
        ctx_dim_post: int = 24,
        ctx_dim_pre: int = 64,
        concat_padding_mask: bool = True,
        use_crossattn_projection: bool = True,
    ) -> None:
        super().__init__()
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.timestep_scale = 1.0
        self.concat_padding_mask = concat_padding_mask
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = ctx_dim_pre
        self.crossattn_proj = nn.Sequential(nn.Linear(ctx_dim_pre, ctx_dim_post, bias=True))
        # ``t_embedder`` returns (t_emb, adaln_lora). Use a simple Module list mimic.
        self.t_embedder = _FakeTEmbedder(dim=dim, lora_dim=dim * 3)
        self.t_embedding_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([_FakeBlock(dim) for _ in range(num_blocks)])
        self.final_layer = _FakeFinalLayer(
            dim=dim, out_per_patch=patch_spatial * patch_spatial * patch_temporal * out_channels
        )
        self._patched_dim = dim
        self._out_channels = out_channels

    def prepare_embedded_sequence(self, x_B_C_T_H_W, *, fps=None, padding_mask=None):
        if self.concat_padding_mask:
            assert padding_mask is not None, "fake DIT: padding_mask must be auto-filled by wrapper"
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        B, C, T, H, W = x_B_C_T_H_W.shape
        f = T // self.patch_temporal
        h = H // self.patch_spatial
        w = W // self.patch_spatial
        # Synthetic patchify: deterministic content from input.
        x_B_T_H_W_D = torch.zeros(B, f, h, w, self._patched_dim, dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
        x_B_T_H_W_D[..., 0] = x_B_C_T_H_W[:, 0, :: self.patch_temporal, :: self.patch_spatial, :: self.patch_spatial]
        rope_emb = torch.zeros(f * h * w, 1, 1, self._patched_dim, dtype=torch.float32, device=x_B_C_T_H_W.device)
        return x_B_T_H_W_D, rope_emb, None

    def unpatchify(self, x_B_T_H_W_O):
        # Inverse of fake patchify: reshape into (B, C, T, H, W)
        B, f, h, w, O = x_B_T_H_W_O.shape
        # O = patch_spatial^2 * patch_temporal * out_channels
        out = x_B_T_H_W_O.reshape(
            B, f, h, w, self._out_channels, self.patch_spatial, self.patch_spatial, self.patch_temporal
        )
        # (B, f, h, w, C, p, p, t) -> (B, C, f*t, h*p, w*p)
        out = out.permute(0, 4, 1, 7, 2, 5, 3, 6).contiguous()
        out = out.reshape(
            B, self._out_channels, f * self.patch_temporal, h * self.patch_spatial, w * self.patch_spatial
        )
        return out


class _FakeTEmbedder(nn.Module):
    def __init__(self, *, dim: int, lora_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.lora_dim = lora_dim
        self.t_proj = nn.Linear(1, dim, bias=False)
        self.adaln_proj = nn.Linear(1, lora_dim, bias=False)

    def forward(self, timesteps_B_T):
        # timesteps_B_T is (B, T) after wrapper's unsqueeze
        t_f = timesteps_B_T.to(torch.float32).unsqueeze(-1)
        emb = self.t_proj(t_f)
        adaln = self.adaln_proj(t_f)
        return emb, adaln


@pytest.fixture
def fake_wrapper():
    torch.manual_seed(0)
    net = _FakeMiniDIT(dim=32, num_blocks=4, patch_spatial=2, patch_temporal=1, out_channels=16)
    return CosmosPredict25VideoBackbone(
        net=net,
        vae=None,
        text_encoder=None,
        dim=32,
        num_layers=4,
        num_heads=4,
        head_dim=8,
        context_dim=24,
        shift_video=3.0,
    )


def _make_inputs(B=1, C=16, T=2, H=4, W=4, ctx_dim_post=24):
    latents = torch.randn(B, C, T, H, W)
    context = torch.randn(B, 8, ctx_dim_post)  # already post-projection
    timestep = torch.randint(0, 1000, (B,))
    return latents, context, timestep


def test_wrapper_attributes_propagate(fake_wrapper):
    assert fake_wrapper.dim == 32
    assert fake_wrapper.num_layers == 4
    assert fake_wrapper.num_heads == 4
    assert fake_wrapper.head_dim == 8
    assert fake_wrapper.text_dim == 24  # context_dim exposed via base text_dim property
    assert fake_wrapper._shift_video == 3.0  # private on the backbone


def test_prepare_run_finalize_shape_conservation(fake_wrapper):
    latents, context, timestep = _make_inputs()
    state = fake_wrapper.prepare(input_latents=latents, context=context, timestep=timestep)
    assert isinstance(state, BlockLoopState)
    # Cosmos-specific extras are populated
    assert "t_embedding_B_T_D" in state.extras
    assert "adaln_lora_B_T_3D" in state.extras
    assert "rope_emb_L_1_1_D" in state.extras
    # x is 5D after patchify
    assert state.hidden_states.dim() == 5
    B, f, h, w, D = state.hidden_states.shape
    assert D == fake_wrapper.dim

    for i in range(fake_wrapper.num_layers):
        state = fake_wrapper.run_block(i, state)
    assert state.hidden_states.shape == (B, f, h, w, D)

    out = fake_wrapper.finalize(state)
    # finalize produces (B, C, T, H, W) at input latent resolution
    assert out.shape == latents.shape


def test_prepare_auto_fills_condition_and_padding_masks(fake_wrapper):
    latents, context, timestep = _make_inputs()
    # Don't pass condition_mask / padding_mask — wrapper must synthesize zeros.
    state = fake_wrapper.prepare(input_latents=latents, context=context, timestep=timestep)
    assert state.hidden_states.shape[-1] == fake_wrapper.dim


def test_crossattn_projection_only_when_dim_matches_pre(fake_wrapper):
    """Wrapper should only call crossattn_proj when context arrives in the
    pre-projection dim (64), and pass post-projection (24) through unchanged."""
    latents, _post, timestep = _make_inputs()
    pre_ctx = torch.randn(1, 8, 64)
    state_pre = fake_wrapper.prepare(input_latents=latents, context=pre_ctx, timestep=timestep)
    # After projection: (B, L, 24)
    assert state_pre.context.shape[-1] == 24

    post_ctx = torch.randn(1, 8, 24)
    state_post = fake_wrapper.prepare(input_latents=latents, context=post_ctx, timestep=timestep)
    assert state_post.context.shape[-1] == 24
    # The 1D check is on the projection NOT firing: post_ctx should be the same tensor
    # (object-identity if no projection was applied).
    assert torch.equal(state_post.context, post_ctx)


def test_preprocess_input_rejects_vace():
    wrapper = _wrapper_with_fake_live_encoder()
    latents = torch.randn(1, 16, 2, 4, 4)
    # Dataset always emits a list (possibly all-None for "no data"); rejection
    # must trigger only when an actual entry is non-None.
    with pytest.raises(NotImplementedError, match="VACE"):
        wrapper._preprocess_input(input_latents=latents, text=["x"], vace_videos=[object()])
    # All-None VACE list passes through (RoboTwin's default).
    out = wrapper._preprocess_input(input_latents=latents, text=["x"], vace_videos=[None])
    assert "input_latents" in out


def test_preprocess_input_passes_through_when_ref_images_absent():
    """T2V path: ``ref_images=None`` or list-of-``None`` produces no TI2V keys.

    ``FirstFrameConditioningTransform`` skips populating ``first_frame_image``
    for samples with an empty video, so the adapter forwards
    ``ref_images=None`` (or list-of-``None``s when only some samples have a
    reference, which ``base.py`` already rejects). Either way, the wrapper
    must not crash on a missing VAE and must not emit TI2V plumbing.
    """
    wrapper = _wrapper_with_fake_live_encoder()
    latents = torch.randn(1, 16, 2, 4, 4)
    out_none = wrapper._preprocess_input(input_latents=latents, text=["x"], ref_images=None)
    assert "input_latents" in out_none
    assert "first_frame_latents" not in out_none
    assert "condition_mask" not in out_none
    out_listed_none = wrapper._preprocess_input(input_latents=latents, text=["x"], ref_images=[None])
    assert "first_frame_latents" not in out_listed_none


def test_preprocess_input_requires_text(fake_wrapper):
    latents = torch.randn(1, 16, 2, 4, 4)
    with pytest.raises(ValueError, match="requires `text`"):
        fake_wrapper._preprocess_input(input_latents=latents)


def test_preprocess_input_requires_configured_encoder(fake_wrapper):
    latents = torch.randn(1, 16, 2, 4, 4)
    with pytest.raises(ValueError, match="text encoder"):
        fake_wrapper._preprocess_input(input_latents=latents, text=["x"])


def test_preprocess_input_returns_required_keys():
    wrapper = _wrapper_with_fake_live_encoder()
    latents = torch.randn(1, 16, 2, 4, 4)
    out = wrapper._preprocess_input(input_latents=latents, text=["x"])
    assert set(out) >= {"input_latents", "context", "context_mask", "seq_lens", "num_frames", "height", "width"}
    assert out["num_frames"] == 2 and out["height"] == 4 and out["width"] == 4
    assert out["seq_lens"].tolist() == [8]


def test_decode_video_without_vae_raises(fake_wrapper):
    latents = torch.randn(1, 16, 2, 4, 4)
    with pytest.raises(RuntimeError, match="VAE"):
        fake_wrapper.decode_video(latents)


# ----------------------------------------------------------------------
# Phase 4 — VAE encode + decode plumbing
# ----------------------------------------------------------------------


class _FakeWanVAEModel:
    """Mimics ``Wan2pt1VAEInterface.model`` for CPU tests."""

    def __init__(self) -> None:
        # A trivial inner module so `_vae_device` can read params.
        self.model: nn.Module = nn.Linear(1, 1)
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16


class _FakeVAEInterface:
    """Mimics ``Wan2pt1VAEInterface`` encode/decode API."""

    def __init__(self) -> None:
        self.model = _FakeWanVAEModel()
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, video):
        self.encode_calls += 1
        B, C, T, H, W = video.shape
        assert C == 3, f"VAE encode expects (B, 3, T, H, W); got C={C}"
        T_lat = 1 + (T - 1) // 4
        return torch.randn(B, 16, T_lat, H // 8, W // 8, dtype=video.dtype, device=video.device)

    def decode(self, latents):
        self.decode_calls += 1
        B, C, T_lat, H_lat, W_lat = latents.shape
        assert C == 16, f"VAE decode expects (B, 16, T_lat, H_lat, W_lat); got C={C}"
        T = (T_lat - 1) * 4 + 1
        return torch.randn(B, 3, T, H_lat * 8, W_lat * 8, dtype=latents.dtype, device=latents.device)


def _make_pil_video(B: int, T: int, H: int, W: int):
    import numpy as np
    from PIL import Image

    return [
        [Image.fromarray((np.ones((H, W, 3), dtype=np.uint8) * (i * 25 % 256))) for i in range(T)] for _ in range(B)
    ]


def _wrapper_with_fake_vae() -> CosmosPredict25VideoBackbone:
    net = _FakeMiniDIT(dim=32, num_blocks=4)
    # Encoder emits post-projection dim (24) directly: the fake VAE produces
    # bf16 latents while the fake DiT's crossattn_proj Linear is float32, so
    # this test skips the projection and exercises only the VAE path.
    return CosmosPredict25VideoBackbone(
        net=net,
        vae=_FakeVAEInterface(),
        text_encoder=_FakeLiveTextEncoder(L=8, ctx_dim_pre=24),
        dim=32,
        num_layers=4,
        num_heads=4,
        head_dim=8,
        context_dim=24,
        shift_video=5.0,
    )


def test_preprocess_input_with_fake_vae_returns_real_latents():
    wrapper = _wrapper_with_fake_vae()
    frames = _make_pil_video(B=1, T=5, H=64, W=64)
    out = wrapper._preprocess_input(frames=frames, text=["x"])

    # Wan2pt1 stride: T_lat = 1 + (5-1)//4 = 2; H_lat=8; W_lat=8; C_z=16.
    assert out["input_latents"].shape == (1, 16, 2, 8, 8)
    assert wrapper._vae_iface.encode_calls == 1
    assert wrapper._vae_iface.decode_calls == 0
    assert torch.isfinite(out["input_latents"]).all()


def test_decode_video_returns_pil_list():
    from PIL import Image

    wrapper = _wrapper_with_fake_vae()
    latents = torch.randn(1, 16, 2, 4, 4)
    frames = wrapper.decode_video(latents)

    assert wrapper._vae_iface.decode_calls == 1
    assert isinstance(frames, list)
    # T_pix = (2 - 1) * 4 + 1 = 5
    assert len(frames) == 5
    for frame in frames:
        assert isinstance(frame, Image.Image)
        assert frame.size == (4 * 8, 4 * 8)  # PIL (W, H)


def test_decode_video_batch_gt_1_raises():
    wrapper = _wrapper_with_fake_vae()
    latents = torch.randn(2, 16, 2, 4, 4)
    with pytest.raises(NotImplementedError, match=r"B=1"):
        wrapper.decode_video(latents)


# ----------------------------------------------------------------------
# §14 — live text-encoder dispatch (Reason1LiveTextEncoder is the real one;
# CPU tests use a callable shim returning a fixed pre-projection tensor).
# ----------------------------------------------------------------------


class _FakeLiveTextEncoder:
    """Callable stand-in matching the real ``Reason1LiveTextEncoder.__call__``
    contract: pre-projection ``(B, L, ctx_dim_pre)`` so the wrapper's auto-gate
    in ``prepare_block_loop`` fires and projects to ``ctx_dim_post``."""

    def __init__(self, *, L: int, ctx_dim_pre: int):
        self.L = L
        self.ctx_dim_pre = ctx_dim_pre
        self.calls = []

    def __call__(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        self.calls.append(list(prompts))
        B = len(prompts)
        return torch.randn(B, self.L, self.ctx_dim_pre)


def _wrapper_with_fake_live_encoder():
    net = _FakeMiniDIT(dim=32, num_blocks=4)  # ctx_dim_pre=64, ctx_dim_post=24
    return CosmosPredict25VideoBackbone(
        net=net,
        vae=None,
        text_encoder=_FakeLiveTextEncoder(L=8, ctx_dim_pre=64),
        dim=32,
        num_layers=4,
        num_heads=4,
        head_dim=8,
        context_dim=24,
        shift_video=5.0,
    )


def test_preprocess_input_uses_live_text_encoder_when_text_provided():
    """Live encoder dispatch path: caller provides `text=[...]`. Wrapper must:
    1. Invoke `self.text_encoder(text)` (records via the fake's `.calls`)
    2. Apply `net.crossattn_proj` in-line (64 → 24 on FakeMiniDIT) so the
       architecture's downstream `_append_proprio_context_token` sees the
       expected post-projection dim.
    """
    wrapper = _wrapper_with_fake_live_encoder()
    latents = torch.randn(1, 16, 2, 4, 4)

    out = wrapper._preprocess_input(input_latents=latents, text=["pick up the block"])

    # The fake encoder records each call so we can confirm the dispatch.
    assert wrapper.text_encoder.calls == [["pick up the block"]], (
        "preprocess_input should call self.text_encoder(text) when only `text=` is given."
    )
    # In-line projection applied: 64 (ctx_dim_pre) → 24 (ctx_dim_post).
    assert out["context"].shape == (1, 8, 24), (
        f"context shape: {tuple(out['context'].shape)} — expected (1, 8, 24) "
        "after preprocess_input's in-line `net.crossattn_proj` projection."
    )
    # `seq_lens` reflects the live encoder's L axis (8).
    assert out["seq_lens"].tolist() == [8]


# ----------------------------------------------------------------------
# §14.7 — CFG dropout for the live encoder path.
# ----------------------------------------------------------------------


def _wrapper_with_dropout(*, p: float, seed=None) -> CosmosPredict25VideoBackbone:
    net = _FakeMiniDIT(dim=32, num_blocks=4)
    return CosmosPredict25VideoBackbone(
        net=net,
        vae=None,
        text_encoder=_FakeLiveTextEncoder(L=8, ctx_dim_pre=64),
        dim=32,
        num_layers=4,
        num_heads=4,
        head_dim=8,
        context_dim=24,
        shift_video=5.0,
        text_dropout_p=p,
        text_dropout_seed=seed,
    )


def test_text_dropout_training_substitutes_empty_strings():
    """With `text_dropout_p=1.0` and the wrapper in training mode every prompt
    must be substituted with `""` before the encoder is called (canonical
    empty-embedding contract)."""
    wrapper = _wrapper_with_dropout(p=1.0, seed=0)
    assert wrapper.training, "fresh nn.Module is in training mode by default"
    latents = torch.randn(2, 16, 2, 4, 4)

    wrapper._preprocess_input(input_latents=latents, text=["pick up the block", "stack the red cube"])

    assert wrapper.text_encoder.calls == [["", ""]], (
        f"All prompts must be replaced with '' under p=1.0 + training; got {wrapper.text_encoder.calls}"
    )


def test_text_dropout_eval_passthrough():
    """In eval mode dropout must NEVER fire — even if `text_dropout_p=1.0`.
    Guards against CFG dropout leaking into inference and silently degrading
    samples. `deploy/model_loader.py:161` and `base.py::generate` both flip
    eval; the wrapper inherits via nn.Module."""
    wrapper = _wrapper_with_dropout(p=1.0, seed=0)
    wrapper.eval()
    latents = torch.randn(2, 16, 2, 4, 4)

    wrapper._preprocess_input(input_latents=latents, text=["pick up the block", "stack the red cube"])

    assert wrapper.text_encoder.calls == [["pick up the block", "stack the red cube"]], (
        f"Eval mode must pass prompts through unchanged regardless of p; got {wrapper.text_encoder.calls}"
    )


def test_text_dropout_seed_reproducible():
    """Two wrappers built with the same `text_dropout_seed` must produce
    identical substitution patterns on the same input — needed for
    deterministic CI smoke runs."""
    prompts = [f"prompt_{i}" for i in range(32)]
    latents = torch.randn(32, 16, 2, 4, 4)

    w1 = _wrapper_with_dropout(p=0.5, seed=42)
    w2 = _wrapper_with_dropout(p=0.5, seed=42)
    w1._preprocess_input(input_latents=latents, text=list(prompts))
    w2._preprocess_input(input_latents=latents, text=list(prompts))

    assert w1.text_encoder.calls == w2.text_encoder.calls, "Same seed must yield identical substitution pattern."
    # Sanity check: with p=0.5 over 32 samples the pattern should be
    # non-degenerate — neither all-real nor all-empty.
    pattern = w1.text_encoder.calls[0]
    n_empty = sum(1 for p in pattern if p == "")
    assert 0 < n_empty < 32, f"p=0.5 should produce a mixed pattern; got {n_empty}/32 empty"


def test_text_dropout_p_zero_passthrough():
    """Default `text_dropout_p=0.0` must be a strict no-op regardless of
    training mode — preserves backwards compatibility for runs that opted
    into the live encoder before §14.7."""
    wrapper = _wrapper_with_dropout(p=0.0)
    assert wrapper.training
    latents = torch.randn(2, 16, 2, 4, 4)

    wrapper._preprocess_input(input_latents=latents, text=["pick up the block", "stack the red cube"])

    assert wrapper.text_encoder.calls == [["pick up the block", "stack the red cube"]]


def test_text_dropout_p_out_of_range_rejected():
    """`text_dropout_p` outside [0, 1] must fail at construction time, not
    silently at the first call (mirrors the builder-side validator)."""
    net = _FakeMiniDIT(dim=32, num_blocks=4)
    common = dict(
        net=net,
        vae=None,
        text_encoder=None,
        dim=32,
        num_layers=4,
        num_heads=4,
        head_dim=8,
        context_dim=24,
        shift_video=5.0,
    )
    with pytest.raises(ValueError, match=r"text_dropout_p"):
        CosmosPredict25VideoBackbone(text_dropout_p=1.5, **common)
    with pytest.raises(ValueError, match=r"text_dropout_p"):
        CosmosPredict25VideoBackbone(text_dropout_p=-0.1, **common)


# ----------------------------------------------------------------------
# Reason1 state-dict registration (Reason1LiveTextEncoder is a plain Python
# class; its inner ``nn.Module`` must be registered as ``reason1`` on
# the wrapper so its weights ride into the unified safetensors).
# ----------------------------------------------------------------------


class _FakeReason1WithInnerModule:
    """Plain Python class with a ``self.model`` ``nn.Module`` child —
    mirrors the real ``Reason1LiveTextEncoder`` shape (which holds the
    Qwen2.5-VL ``nn.Module`` at ``self.model``)."""

    def __init__(self) -> None:
        # A tiny stand-in for Qwen2.5-VL — what matters is that it is an
        # actual ``nn.Module`` with at least one parameter that should turn
        # up under the ``reason1.`` prefix in ``state_dict()``.
        self.model = nn.Linear(4, 4)


class _FakeReason1WithoutInnerModule:
    """Plain Python class with NO ``self.model`` — exercises the
    no-registration fallback (mirrors a callable-only stub like
    ``_FakeLiveTextEncoder`` used elsewhere in this file)."""

    def __init__(self) -> None:
        self.tokenizer = object()


def testreason1_appears_in_state_dict():
    """The wrapper must register a Reason1-shaped encoder's inner
    ``nn.Module`` as ``reason1`` so its parameters flow into the
    unified safetensors. Without this, deploy hosts would still need an
    external Cosmos-Reason1 bundle (the pre-fix behaviour)."""
    net = _FakeMiniDIT(dim=32, num_blocks=4)
    encoder = _FakeReason1WithInnerModule()
    wrapper = CosmosPredict25VideoBackbone(
        net=net,
        vae=None,
        text_encoder=encoder,
        dim=32,
        num_layers=4,
        num_heads=4,
        head_dim=8,
        context_dim=24,
        shift_video=5.0,
    )

    state_keys = list(wrapper.state_dict().keys())
    reason1_keys = [k for k in state_keys if k.startswith("reason1.")]
    assert reason1_keys, (
        f"Expected `reason1.*` keys in wrapper.state_dict(); got prefixes "
        f"{sorted({k.split('.', 1)[0] for k in state_keys})}"
    )
    # The inner module is reachable as an attribute (single registration
    # point — the same object as ``encoder.model``).
    assert wrapper.reason1 is encoder.model
    # And it is the ONLY registration path for that module — no duplicate
    # via ``self.text_encoder`` (which is a plain attribute, not an
    # ``nn.Module``, so it does not enter ``_modules``).
    assert "text_encoder" not in wrapper._modules


def test_reason1_no_inner_module_means_no_registration():
    """Encoders without a ``self.model`` ``nn.Module`` (e.g. callable-only
    test shims) must not crash, and must not add a ``reason1`` entry
    — there is nothing to register."""
    net = _FakeMiniDIT(dim=32, num_blocks=4)
    encoder = _FakeReason1WithoutInnerModule()
    wrapper = CosmosPredict25VideoBackbone(
        net=net,
        vae=None,
        text_encoder=encoder,
        dim=32,
        num_layers=4,
        num_heads=4,
        head_dim=8,
        context_dim=24,
        shift_video=5.0,
    )

    assert "reason1" not in wrapper._modules
    state_keys = list(wrapper.state_dict().keys())
    assert not any(k.startswith("reason1.") for k in state_keys)


def testreason1_state_dict_roundtrip():
    """Round-trip the wrapper's ``state_dict`` through a fresh wrapper of
    the same shape and confirm the Reason1 inner module's weights match
    bit-for-bit. This is the unit-test analogue of the safetensors
    save/load round trip exercised at training time."""
    net1 = _FakeMiniDIT(dim=32, num_blocks=4)
    enc1 = _FakeReason1WithInnerModule()
    w1 = CosmosPredict25VideoBackbone(
        net=net1,
        vae=None,
        text_encoder=enc1,
        dim=32,
        num_layers=4,
        num_heads=4,
        head_dim=8,
        context_dim=24,
        shift_video=5.0,
    )
    # Capture the source weights, then build a second wrapper and load.
    src_weight = enc1.model.weight.detach().clone()
    sd = w1.state_dict()

    net2 = _FakeMiniDIT(dim=32, num_blocks=4)
    enc2 = _FakeReason1WithInnerModule()
    w2 = CosmosPredict25VideoBackbone(
        net=net2,
        vae=None,
        text_encoder=enc2,
        dim=32,
        num_layers=4,
        num_heads=4,
        head_dim=8,
        context_dim=24,
        shift_video=5.0,
    )
    # Sanity check: independently initialised, weights must differ.
    assert not torch.equal(enc2.model.weight, src_weight)
    missing, unexpected = w2.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    assert torch.equal(w2.reason1.weight, src_weight)
