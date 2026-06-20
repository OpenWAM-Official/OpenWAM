"""DINOv3 video encoder tests.

Covers the parts of ``openwam/model/video_backbone/encoder/dinov3.py``
that have non-trivial logic and were previously untested:

- D1   registry round-trip (``register_video_encoder("dinov3")``)
- D2   spec invariants (z_dim/patch from ctor kwargs, fixed temporal=4,
       causal=True, dit_patch_size=(1,2,2), pixel_decode=False)
- D3   ImageNet preprocess shape + normalisation (gray frame ≈ 0)
- D4   ``batch_encode`` shape contract on T_pixel ∈ {1, 5, 9}
- D5   causal mean pool semantics (frame 0 kept verbatim, rest 4-grouped)
- D6   register-token drop (CLS + R register tokens skipped)
- D7   fail-fasts: T ≢ 1 mod 4, H not divisible by patch_size,
       wrong patch-token count
- D8   non-standard ``forward`` returns: bare Tensor accepted, foreign
       dataclass without ``last_hidden_state`` raises TypeError
- D9   ``decode`` / ``to_frames`` raise NotImplementedError (irreversible)
- D10  ``from_skeleton`` fail-fast: missing encoder_cfg / non-existent
       model_path → FileNotFoundError
- D11  ``_extract_structural_fields`` fail-fast: hidden_size missing /
       non-positive → ValueError; falsy ``hidden_size`` does NOT silently
       fall back to ``embed_dim`` (regression for the None-vs-falsy fix)

All tests use a tiny CPU mock ViT — no transformers / no weights load —
so they run on the lint job's CPU PyTorch image.
"""

from __future__ import annotations

import json
import types

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

# ---------------------------------------------------------------------------
# Mock ViT
# ---------------------------------------------------------------------------


class _MockDinoViT(nn.Module):
    """Tiny CPU stand-in for an HF DINOv3 ViT.

    Reproduces only what ``DinoV3VideoEncoder.batch_encode`` consumes from
    the ViT forward — a ``BaseModelOutput``-like object exposing
    ``last_hidden_state`` of shape ``(B*T, 1 + R + Hp*Wp, D)``. CLS and
    register tokens are filled with sentinel values so tests can verify
    they were sliced off; patch tokens carry a deterministic per-position
    pattern so the temporal pool can be eye-checked.

    Has at least one parameter so ``next(self.parameters())`` yields a
    device/dtype anchor (matches what ``preprocess_video`` /
    ``batch_encode`` rely on).
    """

    def __init__(self, embed_dim: int = 8, patch: int = 16, num_register_tokens: int = 0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch = int(patch)
        self.num_register_tokens = int(num_register_tokens)
        # Real parameter so the module has a sane device/dtype.
        self._proj = nn.Linear(1, embed_dim, bias=False)
        with torch.no_grad():
            self._proj.weight.fill_(1.0)

    def forward(self, x: Tensor):
        BT, _C, H, W = x.shape
        h = H // self.patch
        w = W // self.patch
        n_patch = h * w
        n_total = 1 + self.num_register_tokens + n_patch
        out = torch.zeros(BT, n_total, self.embed_dim, device=x.device, dtype=x.dtype)
        # CLS = -1.0 (sentinel), register = -2.0 (sentinel), patches = +1.0.
        # If batch_encode incorrectly forgot to drop CLS/register, the
        # output would mix in these negative sentinels and the post-LN
        # mean would not be near zero with the patch-only signal we set.
        out[:, 0, :] = -1.0
        if self.num_register_tokens > 0:
            out[:, 1 : 1 + self.num_register_tokens, :] = -2.0
        out[:, 1 + self.num_register_tokens :, :] = 1.0
        # Use the param so dtype/device propagate through autograd if ever.
        anchor = self._proj(torch.zeros(1, 1, device=x.device, dtype=x.dtype))
        return types.SimpleNamespace(last_hidden_state=out + anchor.sum() * 0.0)


def _build_encoder(*, embed_dim: int = 8, patch: int = 16, num_register_tokens: int = 0):
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

    vit = _MockDinoViT(embed_dim=embed_dim, patch=patch, num_register_tokens=num_register_tokens)
    return DinoV3VideoEncoder(
        vit,
        embed_dim=embed_dim,
        patch_size=patch,
        num_register_tokens=num_register_tokens,
    )


# ---------------------------------------------------------------------------
# D1: registration
# ---------------------------------------------------------------------------


def test_D1_dinov3_registration_round_trip():
    """``register_video_encoder("dinov3")`` exposes the class via the registry."""
    from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder  # noqa: F401

    assert "dinov3" in _VIDEO_ENCODER_REGISTRY
    assert _VIDEO_ENCODER_REGISTRY["dinov3"] is DinoV3VideoEncoder


# ---------------------------------------------------------------------------
# D2: spec invariants
# ---------------------------------------------------------------------------


def test_D2_dinov3_spec_invariants():
    """Spec fields fixed by ctor kwargs + class invariants. Token-count
    parity with Wan VAE depends on ``temporal_compression=4``,
    ``causal_temporal=True``, and ``dit_patch_size=(1,2,2)`` — these MUST
    NOT drift, so they're pinned here.
    """
    enc = _build_encoder(embed_dim=768, patch=16, num_register_tokens=4)
    properties = enc.properties
    assert properties.z_dim == 768
    assert properties.spatial_compression == 16
    assert properties.temporal_compression == 4
    assert properties.causal_temporal is True
    assert properties.dit_patch_size == (1, 2, 2)
    assert properties.pixel_decode is False


# ---------------------------------------------------------------------------
# D3: preprocess
# ---------------------------------------------------------------------------


def test_D3_dinov3_preprocess_imagenet_normalize():
    """``preprocess_video`` ImageNet-normalizes — uniform 0.5-gray frames
    land near zero, std == 0 (constant input)."""
    enc = _build_encoder()
    frames = [Image.new("RGB", (32, 32), color=(128, 128, 128)) for _ in range(3)]
    video = enc.preprocess_video(frames)
    assert video.shape == (1, 3, 3, 32, 32)
    flat = video.reshape(3, -1)
    # 0.5 input - mean (~0.45) / std (~0.22) ≈ small magnitude
    assert flat.abs().max() < 1.0
    # Constant input ⇒ per-channel std == 0 (post-normalize).
    assert flat.std(dim=1).max() == 0.0


# ---------------------------------------------------------------------------
# D4: batch_encode shape contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T_pixel, T_lat",
    [(1, 1), (5, 2), (9, 3)],
)
def test_D4_dinov3_batch_encode_t_lat_shapes(T_pixel, T_lat):
    """``batch_encode`` shape contract:
    ``(B, 3, T, H, W) → (B, D, 1 + (T-1)/4, H/ps, W/ps)``.
    """
    enc = _build_encoder(embed_dim=8, patch=16)
    v = torch.randn(2, 3, T_pixel, 32, 48)
    z = enc.batch_encode(v)
    assert z.shape == (2, 8, T_lat, 32 // 16, 48 // 16)


# ---------------------------------------------------------------------------
# D5: causal pool semantics
# ---------------------------------------------------------------------------


def test_D5_dinov3_causal_pool_keeps_frame0_and_means_rest():
    """``_causal_temporal_pool``: frame 0 verbatim, rest in groups of 4 -> mean.
    Per-channel constant input 1, 2, 3, 4, 5 across T should yield
    pooled[:, :, 0] = 1 and pooled[:, :, 1] = mean(2, 3, 4, 5) = 3.5.
    """
    from openwam.model.video_backbone.encoder.dinov3 import _causal_temporal_pool

    # x: (B=1, D=2, T=5, H=1, W=1) with x[:, :, t] = t + 1.
    base = torch.arange(1, 6, dtype=torch.float32).view(1, 1, 5, 1, 1)
    x = base.expand(1, 2, 5, 1, 1).contiguous()
    pooled = _causal_temporal_pool(x)
    assert pooled.shape == (1, 2, 2, 1, 1)
    assert torch.allclose(pooled[0, :, 0, 0, 0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(pooled[0, :, 1, 0, 0], torch.tensor([3.5, 3.5]))


def test_D5b_dinov3_causal_pool_t1_keeps_lone_frame():
    """``T == 1`` boundary: ``rest`` is empty, output equals frame 0."""
    from openwam.model.video_backbone.encoder.dinov3 import _causal_temporal_pool

    x = torch.randn(1, 4, 1, 2, 2)
    pooled = _causal_temporal_pool(x)
    assert pooled.shape == x.shape
    assert torch.equal(pooled, x)


# ---------------------------------------------------------------------------
# D6: register-token drop
# ---------------------------------------------------------------------------


def test_D6_dinov3_register_tokens_dropped_from_grid():
    """The mock ViT puts a sentinel ``-1`` in CLS and ``-2`` in register
    tokens, ``+1`` in patch tokens. After ``batch_encode``, the pooled
    grid has gone through a non-affine LayerNorm over D=embed_dim with
    embed_dim=8 — patch tokens are all-ones along D, so LN of a constant
    vector is exactly 0. If CLS/register were NOT dropped, the per-token
    D-vector would mix sentinels (e.g. ``[-1, -1, -1, 1, 1, ...]``),
    breaking the all-zero post-LN invariant.
    """
    enc = _build_encoder(embed_dim=8, patch=16, num_register_tokens=4)
    z = enc.batch_encode(torch.zeros(1, 3, 5, 32, 32))
    # All patch tokens identical along D ⇒ non-affine LN ⇒ exact 0.
    assert torch.allclose(z, torch.zeros_like(z), atol=1e-5)


# ---------------------------------------------------------------------------
# D7: fail-fasts
# ---------------------------------------------------------------------------


def test_D7a_dinov3_T_not_1_mod_4_raises():
    """``T_pixel`` must satisfy ``(T - 1) % 4 == 0``."""
    enc = _build_encoder()
    with pytest.raises(ValueError, match=r"T ≡ 1 \(mod 4\)"):
        enc.batch_encode(torch.randn(1, 3, 4, 16, 16))  # (4-1)%4 = 3 ≠ 0


def test_D7b_dinov3_H_not_divisible_by_patch_raises():
    """H / W must be divisible by ``spec.spatial_compression`` (== patch_size)."""
    enc = _build_encoder(patch=16)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        enc.batch_encode(torch.randn(1, 3, 1, 18, 32))  # 18 % 16 ≠ 0


def test_D7c_dinov3_wrong_patch_token_count_raises():
    """If the ViT returns more tokens than ``1 + R + Hp*Wp``, the
    register-token-aware slice over-shoots and ``patch_tokens.shape[1]``
    won't match ``Hp*Wp`` — fail fast with a register-count diagnostic.
    """
    enc = _build_encoder(embed_dim=8, patch=16, num_register_tokens=8)
    # Replace the mock ViT with one that pretends register_tokens=0 → the
    # encoder's ``n_drop = 1 + 8`` will bite into patch tokens.
    enc._m = _MockDinoViT(embed_dim=8, patch=16, num_register_tokens=0)
    with pytest.raises(ValueError, match="patch tokens but expected"):
        enc.batch_encode(torch.randn(1, 3, 1, 32, 32))


# ---------------------------------------------------------------------------
# D8: non-standard ViT outputs
# ---------------------------------------------------------------------------


class _BareTensorViT(_MockDinoViT):
    """Some HF mirrors return ``last_hidden_state`` directly as a Tensor."""

    def forward(self, x: Tensor):
        out = super().forward(x)
        return out.last_hidden_state


class _ForeignReturnViT(nn.Module):
    """Returns an object with neither ``last_hidden_state`` nor Tensor-ness
    — should trip the defensive type guard."""

    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self._p = nn.Linear(1, embed_dim, bias=False)

    def forward(self, x: Tensor):
        return {"unexpected": x.shape}  # plain dict — no last_hidden_state


def test_D8a_dinov3_accepts_bare_tensor_forward():
    """Bare-Tensor return path: ``getattr(outputs, 'last_hidden_state', None)``
    is None, falls through to ``torch.is_tensor(outputs)`` branch."""
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

    vit = _BareTensorViT(embed_dim=8)
    enc = DinoV3VideoEncoder(vit, embed_dim=8, patch_size=16, num_register_tokens=0)
    z = enc.batch_encode(torch.zeros(1, 3, 1, 32, 32))
    assert z.shape == (1, 8, 1, 2, 2)


def test_D8b_dinov3_foreign_forward_raises_typeerror():
    """Non-tensor, non-BaseModelOutput-like return → TypeError with hint."""
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

    vit = _ForeignReturnViT(embed_dim=8)
    enc = DinoV3VideoEncoder(vit, embed_dim=8, patch_size=16, num_register_tokens=0)
    with pytest.raises(TypeError, match="last_hidden_state"):
        enc.batch_encode(torch.zeros(1, 3, 1, 32, 32))


# ---------------------------------------------------------------------------
# D9: decode/to_frames raise (pixel_decode=False)
# ---------------------------------------------------------------------------


def test_D9_dinov3_decode_and_to_frames_raise():
    """Irreversible encoder: ``decode`` / ``to_frames`` use the ABC's
    default which raises NotImplementedError citing ``pixel_decode``."""
    enc = _build_encoder()
    with pytest.raises(NotImplementedError, match="pixel_decode"):
        enc.decode(torch.zeros(1, 8, 1, 2, 2))
    with pytest.raises(NotImplementedError, match="pixel_decode"):
        enc.to_frames(torch.zeros(1, 3, 1, 32, 32))


# ---------------------------------------------------------------------------
# D10: from_skeleton fail-fast
# ---------------------------------------------------------------------------


def test_D10a_dinov3_from_skeleton_no_encoder_cfg_raises():
    """``from_skeleton(ckpt_dir=None)`` → no readable config.json (deploy is
    strictly self-contained, no ``encoder.model_path`` fallback) →
    FileNotFoundError before any HF call."""
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

    with pytest.raises(FileNotFoundError, match=r"no readable config\.json"):
        DinoV3VideoEncoder.from_skeleton({}, device="cpu", encoder_cfg=None, ckpt_dir=None)


def test_D10b_dinov3_from_skeleton_missing_path_raises(tmp_path):
    """``from_skeleton`` ignores ``encoder_cfg.model_path`` (batch 4 dropped the
    fallback): even with a ``model_path`` set, ``ckpt_dir=None`` fails fast in
    ``_resolve_config_dir`` before any ``AutoConfig.from_pretrained`` call."""
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

    bogus = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match=r"no readable config\.json"):
        DinoV3VideoEncoder.from_skeleton(
            {},
            device="cpu",
            encoder_cfg={"model_path": str(bogus)},
            ckpt_dir=None,
        )


# ---------------------------------------------------------------------------
# D11: structural-field extraction
# ---------------------------------------------------------------------------


def test_D11a_dinov3_extract_structural_fields_happy_path():
    """``hidden_size`` is preferred and ``num_register_tokens`` defaults to 0."""
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

    cfg = types.SimpleNamespace(hidden_size=1024, patch_size=14)
    embed, patch, regs = DinoV3VideoEncoder._extract_structural_fields(cfg, "fake/path")
    assert (embed, patch, regs) == (1024, 14, 0)


def test_D11b_dinov3_extract_structural_fields_falls_back_to_embed_dim():
    """``embed_dim`` fallback only fires when ``hidden_size`` is absent."""
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

    cfg = types.SimpleNamespace(embed_dim=512, patch_size=16, num_register_tokens=4)
    embed, patch, regs = DinoV3VideoEncoder._extract_structural_fields(cfg, "fake/path")
    assert (embed, patch, regs) == (512, 16, 4)


def test_D11c_dinov3_extract_structural_fields_missing_raises():
    """Neither ``hidden_size`` nor ``embed_dim`` → ValueError."""
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

    cfg = types.SimpleNamespace(patch_size=16)
    with pytest.raises(ValueError, match="hidden_size, embed_dim"):
        DinoV3VideoEncoder._extract_structural_fields(cfg, "fake/path")


def test_D11d_dinov3_extract_structural_fields_zero_hidden_does_not_fall_back():
    """Regression for the explicit-None fix: ``hidden_size=0`` MUST surface
    a ValueError, not silently fall through to ``embed_dim``. Previously
    this used a falsy-or fallback that would have silently overwritten 0
    with whatever ``embed_dim`` happened to be (or 0 by default), masking
    a misconfigured snapshot.
    """
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

    cfg = types.SimpleNamespace(hidden_size=0, embed_dim=512, patch_size=16)
    with pytest.raises(ValueError, match="non-positive"):
        DinoV3VideoEncoder._extract_structural_fields(cfg, "fake/path")


# ---------------------------------------------------------------------------
# DS: optional S-VAE feature reducer (mirrors tests/test_vjepa_svae.py)
# ---------------------------------------------------------------------------


def _svae_cfg(input_dim: int, latent_dim: int = 48) -> dict:
    return dict(
        input_dim=input_dim, latent_dim=latent_dim, num_heads=2, num_layers=1, intermediate_size=16, dropout=0.0
    )


def _enc_svae(*, embed_dim: int = 1408, svae: bool = False, latent_dim: int = 48, svae_target_dim=None):
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

    kw = {}
    if svae:
        kw["svae_config"] = _svae_cfg(embed_dim, latent_dim)
    if svae_target_dim is not None:
        kw["svae_target_dim"] = svae_target_dim
    vit = _MockDinoViT(embed_dim=embed_dim, patch=16, num_register_tokens=0)
    return DinoV3VideoEncoder(vit, embed_dim=embed_dim, patch_size=16, num_register_tokens=0, **kw)


def test_DS1_zdim_and_out_norm_track_reducer():
    """``properties.z_dim`` and the non-affine ``_out_norm`` both rebuild against
    the reducer's ``latent_dim`` when an S-VAE is attached, else the raw dim."""
    raw = _enc_svae(embed_dim=1408, svae=False)
    assert raw.properties.z_dim == 1408
    assert raw._out_norm.normalized_shape == (1408,)

    red = _enc_svae(embed_dim=1408, svae=True, latent_dim=48)
    assert red.properties.z_dim == 48
    assert red._out_norm.normalized_shape == (48,)


def test_DS2_batch_encode_output_is_reduced():
    red = _enc_svae(embed_dim=1408, svae=True, latent_dim=48)
    z = red.batch_encode(torch.randn(1, 3, 9, 32, 32))
    assert z.shape == (1, 48, 3, 2, 2)  # (B, latent, T_lat=3, H/16, W/16)


def test_DS3_svae_runs_after_pool_on_raw_dim():
    # The reducer must receive the POST-POOL (T_lat=3), RAW-dim (1408) tensor —
    # not pre-pool (T=9) nor an already-reduced one.
    red = _enc_svae(embed_dim=1408, svae=True)
    captured = {}
    orig = red._svae.encode_mean

    def spy(z):
        captured["shape"] = tuple(z.shape)
        return orig(z)

    red._svae.encode_mean = spy
    red.batch_encode(torch.randn(1, 3, 9, 32, 32))
    assert captured["shape"] == (1, 1408, 3, 2, 2)


def test_DS4_pooled_for_training_raw_vs_reduced():
    raw = _enc_svae(embed_dim=1408, svae=False)
    out = raw.batch_encode_pooled_for_svae_training(torch.randn(1, 3, 9, 32, 32))
    assert out.shape == (1, 1408, 3, 2, 2)  # raw dim, post-pool, pre-LayerNorm

    red = _enc_svae(embed_dim=1408, svae=True)
    with pytest.raises(RuntimeError, match="raw encoder"):
        red.batch_encode_pooled_for_svae_training(torch.randn(1, 3, 9, 32, 32))


def test_DS5_svae_target_dim_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        _enc_svae(embed_dim=1408, svae=True, latent_dim=48, svae_target_dim=24)


def test_DS6_reducer_is_frozen_eval():
    red = _enc_svae(embed_dim=1408, svae=True)
    assert red._svae.training is False
    assert all(not p.requires_grad for p in red._svae.parameters())


def test_DS7_deploy_sidecar_write_and_read(tmp_path):
    from openwam.model.video_backbone.encoder.svae import _CHECKPOINT_FORMAT_VERSION, reducer

    red = _enc_svae(embed_dim=1408, svae=True)
    out = tmp_path / "ckpt"
    out.mkdir()
    # save_deploy_assets is strictly self-contained: it copies the DINOv3
    # config.json from encoder.model_path (hard error if absent) AND writes the
    # S-VAE sidecar. Provide a dummy config.json so the call succeeds; this test
    # asserts on the S-VAE sidecar half.
    src = tmp_path / "enc_src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            video_backbone=types.SimpleNamespace(encoder=types.SimpleNamespace(model_path=str(src)))
        )
    )
    red.save_deploy_assets(str(out), cfg)

    sidecar = out / "svae_config.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["format_version"] == _CHECKPOINT_FORMAT_VERSION
    assert payload["model_config"] == red._svae.config_dict()
    assert reducer.read_sidecar(str(out)) == red._svae.config_dict()
    assert reducer.read_sidecar(str(tmp_path / "absent")) is None


def test_DS8_deploy_skeleton_strict_load_roundtrip():
    enc1 = _enc_svae(embed_dim=1408, svae=True)
    # non-trivial reducer buffers so the round-trip actually exercises them
    enc1._svae.set_input_stats(torch.randn(1408).abs() + 0.1, torch.rand(1408) + 0.5)

    skeleton = _enc_svae(embed_dim=1408, svae=True)
    missing_unexpected = skeleton.load_state_dict(enc1.state_dict(), strict=True)
    assert not missing_unexpected.missing_keys and not missing_unexpected.unexpected_keys

    enc1.eval()
    skeleton.eval()
    v9 = torch.randn(1, 3, 9, 32, 32)
    assert torch.allclose(enc1.batch_encode(v9), skeleton.batch_encode(v9), atol=1e-5)


def test_DS9_disabled_is_passthrough():
    raw = _enc_svae(embed_dim=8, svae=False)
    z = raw.batch_encode(torch.zeros(1, 3, 9, 32, 32))
    assert z.shape == (1, 8, 3, 2, 2)  # unchanged raw-dim behaviour
    assert raw._svae is None
