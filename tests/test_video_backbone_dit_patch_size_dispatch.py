"""Per-token ``t_mod`` token-count dispatch on ``dit_patch_size``.

Drives commit A2: the two ``tokens_per_frame = latents.shape[3] *
latents.shape[4] // 4`` hardcodes inside :meth:`WanVideoBackbone.prepare`
(TI2V branch + ``force_per_token_t_mod`` broadcast branch) used to assume
Wan's native ``(1, 2, 2)`` first-layer patch. Once an external encoder
pre-patchifies the latent grid, its declared ``dit_patch_size`` may be
``(1, 1, 1)`` (ViT-style — DiT first conv becomes a channel projection) or
``(1, 4, 4)`` (hypothetical 4x DiT patch), and the hardcoded ``// 4``
silently produces the wrong per-token sequence length.

These parametrized tests construct a tiny CPU WanModel + matching
``external_encoder.properties.dit_patch_size`` and verify the per-token ``t_mod``
length equals ``f_lat * (H * W // (ps[1] * ps[2]))`` for three patch
configurations. Native ``(1, 2, 2)`` is included so the existing behavior
is locked in as a regression guard.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from openwam.model.video_backbone.encoder.base import VideoEncoderProperties
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v


def _mock_encoder_with_patch(dit_patch_size: tuple[int, int, int]) -> SimpleNamespace:
    """Duck-typed encoder exposing only ``encoder.properties.dit_patch_size``.

    ``WanVideoBackbone.__init__`` reads exactly this one attribute when
    ``external_encoder is not None`` — the full :class:`VideoEncoder` ABC is
    not needed for the dispatch path under test.
    """
    properties = VideoEncoderProperties(
        z_dim=16,
        spatial_compression=8,
        temporal_compression=4,
        causal_temporal=True,
        dit_patch_size=dit_patch_size,
    )
    return SimpleNamespace(properties=properties)


def _build_backbone_with_patch(
    patch_size: tuple[int, int, int],
    *,
    ti2v: bool = False,
) -> tuple[Wan22Ti2v, WanModel]:
    """Tiny CPU backbone whose DiT patchifies at ``patch_size`` and whose
    adapter's ``_dit_patch_size`` matches via a mock external encoder.

    ``ti2v=True`` flips ``seperated_timestep`` + ``fuse_vae_embedding_in_latents``
    so the TI2V branch of ``prepare()`` fires (line 469 in main); ``ti2v=False``
    lets the ``force_per_token_t_mod`` elif fire (line 495).
    """
    model = WanModel(
        dim=64,
        in_dim=4,
        ffn_dim=128,
        out_dim=4,
        text_dim=32,
        freq_dim=32,
        eps=1e-6,
        patch_size=patch_size,
        num_heads=4,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=ti2v,
        fuse_vae_embedding_in_latents=ti2v,
    )
    pipe = SimpleNamespace(
        dit=model,
        motion_controller=None,
        vace=None,
        use_unified_sequence_parallel=False,
    )
    encoder = _mock_encoder_with_patch(patch_size)
    return Wan22Ti2v(pipe, external_encoder=encoder), model


# H = W = 8 is divisible by every patch under test (2, 1, 4); F = 2 keeps
# the runtime cheap while still letting f_lat * tokens_per_frame stay
# distinguishable across the parameters.
_F, _H, _W = 2, 8, 8


@pytest.mark.parametrize(
    "patch_size,expected_div",
    [
        ((1, 2, 2), 4),  # Wan native (and CosmosPredict25 native) — regression guard
        ((1, 1, 1), 1),  # ViT-style pre-patchified encoder, DiT 1st conv is channel proj
        ((1, 4, 4), 16),  # hypothetical 4x DiT patch
    ],
)
def test_force_per_token_t_mod_dispatches_on_dit_patch_size(patch_size, expected_div):
    """The ``force_per_token_t_mod`` branch must size ``t_mod`` to
    ``f_lat * (H*W // (ps[1]*ps[2]))`` rows so a non-(1,2,2) external encoder
    produces a t_mod row count that matches the DiT's actual post-patchify
    token count."""
    backbone, model = _build_backbone_with_patch(patch_size, ti2v=False)
    latents = torch.randn(1, model.in_dim, _F, _H, _W)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 3, 32)

    state = backbone.prepare(
        latents=latents,
        timestep=timestep,
        context=context,
        force_per_token_t_mod=True,
    )

    expected_L = _F * (_H * _W // expected_div)
    assert state.time_mod.dim() == 4, f"force_per_token_t_mod must yield 4D t_mod; got dim={state.time_mod.dim()}"
    assert state.time_mod.shape == (1, expected_L, 6, model.dim), (
        f"patch={patch_size} expected L={expected_L}; got t_mod shape={tuple(state.time_mod.shape)}"
    )
    # Cross-check against the DiT's actual post-patchify token count.
    assert state.hidden_states.shape[1] == expected_L, (
        f"patch={patch_size} post-patchify L={state.hidden_states.shape[1]} disagrees with "
        f"per-token t_mod L={expected_L} — dispatch is out of sync with patchify"
    )


@pytest.mark.parametrize(
    "patch_size,expected_div",
    [
        ((1, 2, 2), 4),
        ((1, 1, 1), 1),
        ((1, 4, 4), 16),
    ],
)
def test_ti2v_branch_dispatches_on_dit_patch_size(patch_size, expected_div):
    """The TI2V branch (``seperated_timestep`` + ``fuse_vae_embedding_in_latents``)
    must also dispatch ``tokens_per_frame`` through ``self._dit_patch_size``.
    Same arithmetic as the ``force_per_token_t_mod`` branch, different gate."""
    backbone, model = _build_backbone_with_patch(patch_size, ti2v=True)
    latents = torch.randn(1, model.in_dim, _F, _H, _W)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 3, 32)

    state = backbone.prepare(
        latents=latents,
        timestep=timestep,
        context=context,
        fuse_vae_embedding_in_latents=True,
        num_clean_prefix_frames=1,
    )

    expected_L = _F * (_H * _W // expected_div)
    assert state.time_mod.dim() == 4
    assert state.time_mod.shape == (1, expected_L, 6, model.dim), (
        f"patch={patch_size} expected L={expected_L}; got t_mod shape={tuple(state.time_mod.shape)}"
    )
    assert state.hidden_states.shape[1] == expected_L
