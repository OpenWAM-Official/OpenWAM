"""Phase 0 smoke tests for ``SanaVideoBackbone``.

What's covered without GPU + real weights:
- Module import + registration (the basic plumbing works)
- ``attn_kernel`` property advertises ``linear_relu``
- ``pre_attn_at_layer`` returns the dual-track ``q_unrot/k_unrot`` contract

What requires SANA importable (``third_party/Sana`` initialised + working timm):
- Mini-model construction
- Numerical equivalence: ``prepare → run_block × N → finalize`` ≡ raw
  ``SanaMSVideo.forward`` (rtol=1e-4, atol=1e-5)
- Numerical equivalence at the sub-block level: ``pre_attn + native_attn +
  post_attn`` ≡ ``block.forward`` (rtol=1e-4, atol=1e-5)

The full-weight equivalence on the published 2B 480p ckpt is in
``tests/gpu/test_sana_backbone_2b.py`` (not part of this Phase 0 commit; needs
deploy-side downloader).
"""

from __future__ import annotations

import pytest


def _sana_importable() -> bool:
    """Probe whether ``third_party/Sana`` is initialized and imports cleanly.

    Two failure modes:
      - submodule not init'd → ``ModuleNotFoundError: diffusion``
      - broken timm in venv (see docs/sana_vendor.md) → ``ModuleNotFoundError: timm.version``
    """
    try:
        import diffusion.model.nets.sana_multi_scale_video  # noqa: F401
    except Exception:
        return False
    return True


SANA_AVAILABLE = _sana_importable()
requires_sana = pytest.mark.skipif(
    not SANA_AVAILABLE,
    reason="third_party/Sana not importable (submodule init or timm version.py missing)",
)


def test_sana_video_2b_is_registered():
    """The registry lookup works without importing SANA upstream itself."""
    from openwam.model.video_backbone import _VIDEO_BACKBONE_REGISTRY

    assert "sana_video_2b" in _VIDEO_BACKBONE_REGISTRY


def test_sana_backbone_class_imports():
    """Adapter class is importable regardless of SANA's runtime availability."""
    from openwam.model.video_backbone.sana import SanaMSVideoSplit, SanaVideoBackbone

    assert SanaVideoBackbone.__name__ == "SanaVideoBackbone"
    assert SanaMSVideoSplit.__name__ == "SanaMSVideoSplit"


@requires_sana
def test_mini_pipeline_attn_kernel():
    """attn_kernel returns ``linear_relu`` — driver dispatch contract."""
    from openwam.model.video_backbone.sana import SanaVideoBackbone

    bb = SanaVideoBackbone.from_mini_config()
    assert bb.attn_kernel == "linear_relu"


@requires_sana
def test_mini_pipeline_basic_properties():
    """Basic ABC properties resolve against a mini SANA DiT."""
    from openwam.model.video_backbone.sana import SanaVideoBackbone

    bb = SanaVideoBackbone.from_mini_config(depth=2, hidden_size=128, num_heads=4, linear_head_dim=32)
    assert bb.num_layers == 2
    assert bb.dim == 128
    # 128 // 32 = 4 head splits for LiteLAReLURope
    assert bb.num_heads == 4
    assert bb.head_dim == 32
    assert bb.video_attention_mask_mode == "first_frame_causal"
    assert "dit" in bb.submodule_names


@requires_sana
def test_split_eq_native():
    """``prepare → run_block × N → finalize`` ≡ raw ``SanaMSVideo.forward``."""
    import torch

    from openwam.model.video_backbone.sana import SanaVideoBackbone

    if not torch.cuda.is_available():
        pytest.skip(
            "Numerical equivalence requires CUDA — fp32 CPU paths in SANA's "
            "depth_conv can disagree from the cuda kernel by >rtol."
        )

    f, h, w = 4, 8, 8
    bb = SanaVideoBackbone.from_mini_config(
        depth=2, hidden_size=128, num_heads=4, linear_head_dim=32, f=f, h=h, w=w, device="cuda", dtype=torch.float32
    )
    dit = bb._dit

    # Input shape mirrors upstream: (B, C, T, H, W). patch_size = (1, 2, 2).
    B = 1
    x = torch.randn(B, 16, f, h, w, device="cuda", dtype=torch.float32)
    timestep = torch.tensor([100], device="cuda")
    y = torch.randn(B, 1, 8, 64, device="cuda", dtype=torch.float32)
    mask = torch.ones(B, 1, 1, 8, dtype=torch.int16, device="cuda")

    with torch.no_grad():
        y_native = dit(x, timestep=timestep, y=y, mask=mask)

        state = bb.prepare(x=x, timestep=timestep, y=y, mask=mask)
        for i in range(bb.num_layers):
            state = bb.run_block(i, state)
        y_split = bb.finalize(state)

    torch.testing.assert_close(y_native, y_split, rtol=1e-4, atol=1e-5)


@requires_sana
def test_pre_attn_returns_unrot_qk():
    """``pre_attn_at_layer`` honors the linear-attn contract (q_unrot/k_unrot)."""
    import torch

    from openwam.model.video_backbone.sana import SanaVideoBackbone

    bb = SanaVideoBackbone.from_mini_config(depth=1, hidden_size=128, num_heads=4, linear_head_dim=32)
    f, h, w = 4, 8, 8
    B = 1
    x = torch.randn(B, 16, f, h, w, dtype=torch.float32)
    timestep = torch.tensor([100])
    y = torch.randn(B, 1, 8, 64, dtype=torch.float32)
    mask = torch.ones(B, 1, 1, 8, dtype=torch.int16)

    state = bb.prepare(x=x, timestep=timestep, y=y, mask=mask)
    q, k, v, post = bb.pre_attn_at_layer(0, state)

    # MoT contract: q/k/v shaped (B, S, H*D)
    assert q.dim() == 3 and q.shape[0] == B
    assert q.shape == k.shape == v.shape

    # Linear-attn contract for Phase 3 driver dispatch
    assert post.get("uses_linear_attn") is True
    assert "q_unrot" in post and post["q_unrot"].shape == q.shape
    assert "k_unrot" in post and post["k_unrot"].shape == k.shape
    # Block ref is present so post_attn_at_layer can finish without re-running modulation
    assert post["block"] is bb._dit.blocks[0]


@requires_sana
def test_block_pre_post_eq_block_forward():
    """``block_pre_attn + native_attn + block_post_attn`` ≡ raw block.forward."""
    import torch

    from openwam.model.video_backbone.sana import SanaVideoBackbone

    if not torch.cuda.is_available():
        pytest.skip("See test_split_eq_native — CUDA-only.")

    f, h, w = 4, 8, 8
    bb = SanaVideoBackbone.from_mini_config(
        depth=1, hidden_size=128, num_heads=4, linear_head_dim=32, f=f, h=h, w=w, device="cuda", dtype=torch.float32
    )
    B = 1
    x = torch.randn(B, 16, f, h, w, device="cuda", dtype=torch.float32)
    timestep = torch.tensor([100], device="cuda")
    y = torch.randn(B, 1, 8, 64, device="cuda", dtype=torch.float32)
    mask = torch.ones(B, 1, 1, 8, dtype=torch.int16, device="cuda")

    state = bb.prepare(x=x, timestep=timestep, y=y, mask=mask)
    split = state.extras["split"]
    block = bb._dit.blocks[0]

    # ``prepare`` patchifies x to (B, S, C); the post-patchify spatial dims
    # (state.f / state.h / state.w) are what GLUMBConvTemp's HW reshape
    # needs — not the raw latent (f, h, w) we passed in above.
    f_p, h_p, w_p = state.f, state.h, state.w

    # Native block call
    with torch.no_grad():
        y_block = block(
            state.x.clone(),
            state.context,
            state.t_mod,
            state.context_mask,
            (f_p, h_p, w_p),
            state.freqs,
        )

        # Split + recompose
        q, k, v, post = split.block_pre_attn(0, state.x.clone(), state.t_mod, state.freqs)
        attn_out = split.native_attn(post)
        y_split = split.block_post_attn(
            0,
            attn_out,
            post,
            y=state.context,
            y_lens=state.context_mask,
            f=f_p,
            h=h_p,
            w=w_p,
        )

    torch.testing.assert_close(y_block, y_split, rtol=1e-4, atol=1e-5)
