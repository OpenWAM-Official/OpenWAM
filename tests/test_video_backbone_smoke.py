"""Smoke tests for the video backbone module.

Verifies that the video backbone can be imported and basic model
instantiation works without requiring GPU or model weights.
"""


def test_import_pipeline():
    """WanVideoPipeline should be importable from video_backbone."""
    from openwam.model.video_backbone import WanVideoPipeline

    assert WanVideoPipeline is not None
    assert hasattr(WanVideoPipeline, "from_pretrained")


def test_import_wan_model():
    """WanModel (video DiT) should be importable."""
    from openwam.model.video_backbone.diffsynth.models.wan_video_dit import WanModel

    assert WanModel is not None


def test_wan_model_has_dim():
    """WanModel instance should expose .dim attribute for video_dim derivation."""
    from openwam.model.video_backbone.diffsynth.models.wan_video_dit import WanModel

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
    from openwam.model.video_backbone.diffsynth.models.wan_video_vae import WanVideoVAE

    assert WanVideoVAE is not None


def test_import_text_encoder():
    """WanTextEncoder should be importable."""
    from openwam.model.video_backbone.diffsynth.models.wan_video_text_encoder import WanTextEncoder

    assert WanTextEncoder is not None


def test_license_exists():
    """Apache 2.0 LICENSE file must exist in the diffsynth directory."""
    from pathlib import Path

    license_path = (
        Path(__file__).resolve().parents[1] / "openwam" / "model" / "video_backbone" / "diffsynth" / "LICENSE"
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
    test_license_exists()
    print("All smoke tests passed.")
