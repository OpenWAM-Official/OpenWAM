"""GPU smoke for the real Cosmos-Predict2.5-2B checkpoint.

Loads the post-trained 2B EMA weights from the canonical asset path and
drives ``prepare → 28 × run_block → finalize`` on tiny dummy latents,
asserting shape conservation (B, C, T, H, W).

Skip conditions:
- No CUDA available.
- ``cosmos_predict2`` is not installed (run ``bash scripts/install_cosmos25.sh`` first).
- The asset bundle at ``COSMOS25_ASSET_PATH`` (default
  ``/path/to/assets/Cosmos-Predict2.5-2B``) does not exist on the host.

The test is annotated ``@pytest.mark.gpu`` so the default CPU CI run picks
it up via the marker filter; explicit invocation:

    .venv/bin/python -m pytest -q -m gpu tests/test_cosmos25_real_load.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

ASSET_PATH = Path(os.environ.get("COSMOS25_ASSET_PATH", "/path/to/assets/Cosmos-Predict2.5-2B"))


def _skip_unless_runnable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if not ASSET_PATH.exists():
        pytest.skip(f"Cosmos asset bundle missing at {ASSET_PATH}.")
    try:
        import cosmos_predict2  # noqa: F401
    except ImportError:
        pytest.skip("cosmos_predict2 not installed; run scripts/install_cosmos25.sh first.")


@pytest.mark.parametrize("sac_mode", ["none", "mm_only"])
def test_real_load_block_loop_preserves_shape(sac_mode):
    _skip_unless_runnable()
    from openwam.model.video_backbone import build_video_backbone

    cfg = {
        "video_backbone": {
            "name": "cosmos25_predict_2b",
            "model_path": str(ASSET_PATH),
            "model_variant": "base/post-trained",
            "text_encoder": "none",
            "freeze": True,
            "sac_mode": sac_mode,
        }
    }
    # `build_video_backbone(name, cfg, device=...)` only forwards `device` on
    # the deploy path (when `source` is given). For training-style
    # instantiation we move the backbone explicitly afterwards.
    vb = build_video_backbone("cosmos25_predict_2b", cfg)
    vb.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))

    # Geometry probed from the checkpoint.
    assert vb.dim == 2048
    assert vb.num_layers == 28
    assert vb.num_heads == 16
    assert vb.head_dim == 128
    assert vb.context_dim == 1024

    # Confirm SAC wrap state matches the config. `ptd_checkpoint_wrapper`
    # wraps each block in a `CheckpointWrapper(_checkpoint_wrapped_module=block)`
    # — `CheckpointWrapper.state_dict()` strips the prefix on the way out, so
    # the only reliable signal is the `_checkpoint_wrapped_module` attribute
    # itself.
    net = vb._pipe.net
    block0 = net.blocks[0]
    is_sac_wrapped = hasattr(block0, "_checkpoint_wrapped_module")
    if sac_mode == "none":
        assert not is_sac_wrapped, f"sac_mode='none' should leave blocks raw, got type={type(block0).__name__}"
    else:
        assert is_sac_wrapped, f"sac_mode={sac_mode!r} should wrap each block, got type={type(block0).__name__}"

    # Tiny dummy inputs — minimum shapes that exercise the 5D block forward.
    # Latent T=2, spatial 8x8 → after patch (1, 2, 2): (B=1, T=2, H=4, W=4, D=2048).
    B, C, T, H, W = 1, 16, 2, 8, 8
    latents = torch.randn(B, C, T, H, W, dtype=torch.bfloat16, device="cuda:0")
    # `text_encoder: none` ⇒ caller provides post-projection context (1024-dim).
    context = torch.randn(B, 16, 1024, dtype=torch.bfloat16, device="cuda:0")
    timestep = torch.randint(0, 1000, (B,), device="cuda:0").to(torch.bfloat16)

    state = vb.prepare(input_latents=latents, context=context, timestep=timestep)
    assert state.x.dim() == 5
    for i in range(vb.num_layers):
        state = vb.run_block(i, state)
    out = vb.finalize(state)
    assert out.shape == (B, C, T, H, W), f"shape changed: in={latents.shape} out={out.shape}"
    # Real flow-matching velocity output, sanity-check finite.
    assert torch.isfinite(out).all()


# ----------------------------------------------------------------------
# Phase 4 — real Wan2pt1 VAE encode + decode round-trip
# ----------------------------------------------------------------------


def _build_backbone_with_real_vae():
    from openwam.model.video_backbone import build_video_backbone

    cfg = {
        "video_backbone": {
            "name": "cosmos25_predict_2b",
            "model_path": str(ASSET_PATH),
            "model_variant": "base/post-trained",
            "text_encoder": "none",
            "vae": "wan2pt1",
            "freeze": True,
            "sac_mode": "none",
        }
    }
    vb = build_video_backbone("cosmos25_predict_2b", cfg)
    vb.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
    return vb


def test_real_vae_load_and_shape_round_trip():
    """Encode random pixels through the real Wan2pt1 VAE and decode back."""
    _skip_unless_runnable()
    if not (ASSET_PATH / "tokenizer.pth").exists():
        pytest.skip(f"tokenizer.pth missing at {ASSET_PATH}.")

    vb = _build_backbone_with_real_vae()
    vae = vb._pipe.vae
    assert vae is not None, "vae='wan2pt1' should populate the wrapper VAE slot"

    # T_pix=5 → T_lat = 1 + (5-1)//4 = 2; spatial 64→8 (stride 8); z_dim=16.
    pixels = torch.randn(1, 3, 5, 64, 64, dtype=torch.bfloat16, device="cuda:0")
    latents = vae.encode(pixels)
    assert latents.shape == (1, 16, 2, 8, 8), f"encode latent shape={latents.shape}"
    assert torch.isfinite(latents).all()

    recon = vae.decode(latents)
    assert recon.shape == (1, 3, 5, 64, 64), f"decode pixel shape={recon.shape}"
    assert torch.isfinite(recon).all()


def test_real_preprocess_input_full_path():
    """End-to-end: PIL frames → preprocess_input → Wan2pt1 encode → latents."""
    _skip_unless_runnable()
    if not (ASSET_PATH / "tokenizer.pth").exists():
        pytest.skip(f"tokenizer.pth missing at {ASSET_PATH}.")

    import numpy as np
    from PIL import Image

    vb = _build_backbone_with_real_vae()
    # 5 frames at 64×64 — Wan2pt1 internal conv3d wants T>=3 after chunking;
    # 5 frames is the smallest size that works through `temporal_window=4`.
    frames = [[Image.fromarray((np.ones((64, 64, 3), dtype=np.uint8) * (i * 30 % 256))) for i in range(5)]]
    pre_text = torch.randn(1, 16, 1024, dtype=torch.bfloat16, device="cuda:0")

    out = vb.preprocess_input(frames=frames, text=None, pre_encoded_text=pre_text)
    # T_lat = 1 + (5-1)//4 = 2; spatial 64/8=8; C_z=16.
    assert out["input_latents"].shape == (1, 16, 2, 8, 8), f"input_latents shape={out['input_latents'].shape}"
    assert torch.isfinite(out["input_latents"]).all()
    assert out["context"].shape == (1, 16, 1024)
