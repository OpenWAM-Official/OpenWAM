"""Smoke tests for the video backbone module.

Verifies that the video backbone can be imported and basic model
instantiation works without requiring GPU or model weights.
"""

import torch


def test_import_pipeline():
    """WanVideoPipeline should be importable from the first-class Wan backbone package."""
    from openwam.model.video_backbone.wan.pipeline import WanVideoPipeline

    assert WanVideoPipeline is not None
    assert hasattr(WanVideoPipeline, "from_pretrained")


def test_import_wan_model():
    """WanModel (video DiT) should be importable."""
    from openwam.model.video_backbone.wan.dit import WanModel

    assert WanModel is not None


def test_wan_model_has_dim():
    """WanModel instance should expose .dim attribute for video_dim derivation."""
    from openwam.model.video_backbone.wan.dit import WanModel

    # Tiny model for testing (not real weights)
    model = WanModel(
        dim=64,
        in_dim=4,
        ffn_dim=128,
        out_dim=4,
        text_dim=64,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=2,
        has_image_input=False,
    )
    assert model.dim == 64


def test_import_vae():
    """WanVideoVAE should be importable."""
    from openwam.model.video_backbone.wan.vae import WanVideoVAE

    assert WanVideoVAE is not None


def test_import_text_encoder():
    """WanTextEncoder should be importable."""
    from openwam.model.video_backbone.wan.text_encoder import WanTextEncoder

    assert WanTextEncoder is not None


def test_wan_video_backbone_adapter_freq_helpers():
    """extend_freqs_with_action_tokens appends identity rotations for action positions."""
    from types import SimpleNamespace

    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

    pipe = SimpleNamespace(dit=None, use_unified_sequence_parallel=False)
    adapter = WanVideoBackbone(pipe)

    freqs = torch.polar(torch.ones(4, 1, 6), torch.zeros(4, 1, 6))
    extended = adapter._extend_freqs_with_action_tokens(freqs, 2)
    assert extended.shape == (6, 1, 6)
    assert torch.allclose(extended[-2:].real, torch.ones_like(extended[-2:].real))
    assert torch.allclose(extended[-2:].imag, torch.zeros_like(extended[-2:].imag))

    # n_action_tokens=0 is a passthrough.
    assert adapter._extend_freqs_with_action_tokens(freqs, 0) is freqs


def test_wan_video_backbone_adapter_reference_prefix_len():
    """_compute_reference_prefix_len mirrors the ref_conv flatten path's prefix length."""
    from types import SimpleNamespace

    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

    adapter = WanVideoBackbone(SimpleNamespace(dit=None, use_unified_sequence_parallel=False))

    assert adapter._compute_reference_prefix_len(None) == 0
    # 5D (B, C, 1, H, W) — H*W tokens after ref_conv flatten.
    assert adapter._compute_reference_prefix_len(torch.zeros(1, 4, 1, 6, 8)) == 48
    # 4D (B, C, H, W).
    assert adapter._compute_reference_prefix_len(torch.zeros(1, 4, 6, 8)) == 48


def test_wan_video_backbone_is_ti2v():
    """_is_ti2v returns True when fuse_vae_embedding_in_latents is set."""
    from types import SimpleNamespace

    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

    pipe_a = SimpleNamespace(
        dit=SimpleNamespace(seperated_timestep=True, fuse_vae_embedding_in_latents=True),
        use_unified_sequence_parallel=False,
    )
    adapter_a = WanVideoBackbone(pipe_a)
    assert adapter_a._is_ti2v is True

    pipe_b = SimpleNamespace(
        dit=SimpleNamespace(seperated_timestep=False),
        use_unified_sequence_parallel=False,
    )
    adapter_b = WanVideoBackbone(pipe_b)
    assert adapter_b._is_ti2v is False


def test_license_exists():
    """Apache 2.0 LICENSE file must exist in the extracted Wan license directory."""
    from pathlib import Path

    license_path = (
        Path(__file__).resolve().parents[1] / "openwam" / "model" / "video_backbone" / "wan" / "license" / "LICENSE"
    )
    assert license_path.exists(), f"LICENSE not found at {license_path}"
    content = license_path.read_text()
    assert "Apache License" in content


if __name__ == "__main__":
    test_import_pipeline()
    test_import_wan_model()
    test_wan_model_has_dim()
    test_import_vae()
    test_import_text_encoder()
    test_wan_video_backbone_adapter_freq_helpers()
    test_wan_video_backbone_adapter_reference_prefix_len()
    test_wan_video_backbone_is_ti2v()
    test_license_exists()
    print("All smoke tests passed.")
