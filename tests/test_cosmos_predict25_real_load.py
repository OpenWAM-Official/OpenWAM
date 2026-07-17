"""GPU smoke for the real Cosmos-Predict2.5-2B checkpoint.

Loads the post-trained 2B EMA weights from the canonical asset path and
drives ``prepare → 28 × run_block → finalize`` on tiny dummy latents,
asserting shape conservation (B, C, T, H, W). Also round-trips the real
Wan2pt1 VAE and the ``preprocess_input_for_train`` full path.

Skip conditions:
- No CUDA available.
- ``cosmos_predict2`` is not installed (run ``bash scripts/install_cosmos_predict25.sh`` first).
- The asset bundle at ``COSMOS25_ASSET_PATH`` (default
  ``/path/to/assets/Cosmos-Predict2.5-2B``) does not exist on the host.

Annotated ``@pytest.mark.gpu``; explicit invocation:

    .venv/bin/python -m pytest -q tests/test_cosmos_predict25_real_load.py
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
        pytest.skip("cosmos_predict2 not installed; run scripts/install_cosmos_predict25.sh first.")


@pytest.mark.parametrize("sac_mode", ["none", "mm_only"])
def test_real_load_block_loop_preserves_shape(sac_mode, stub_reason1):
    _skip_unless_runnable()
    from openwam.model.video_backbone import build_video_backbone

    cfg = {
        "video_backbone": {
            "name": "cosmos_predict25_2b",
            "model_path": str(ASSET_PATH),
            "model_variant": "base/post-trained",
            "text_encoder_path": "/stub",
            "sac_mode": sac_mode,
        }
    }
    vb = build_video_backbone("cosmos_predict25_2b", cfg)
    vb.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))

    # Geometry probed from the checkpoint.
    assert vb.dim == 2048
    assert vb.num_layers == 28
    assert vb.num_heads == 16
    assert vb.head_dim == 128
    assert vb.text_dim == 1024  # CosmosPredict25 context dim, exposed via the base text_dim property

    # Confirm SAC wrap state matches the config.
    net = vb.dit
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
    # `prepare` consumes post-projection context (1024-dim) directly.
    context = torch.randn(B, 16, 1024, dtype=torch.bfloat16, device="cuda:0")
    timestep = torch.randint(0, 1000, (B,), device="cuda:0").to(torch.bfloat16)

    state = vb.prepare(input_latents=latents, context=context, timestep=timestep)
    assert state.hidden_states.dim() == 5
    for i in range(vb.num_layers):
        state = vb.run_block(i, state)
    out = vb.finalize(state)
    assert out.shape == (B, C, T, H, W), f"shape changed: in={latents.shape} out={out.shape}"
    # Real flow-matching velocity output, sanity-check finite.
    assert torch.isfinite(out).all()


# ----------------------------------------------------------------------
# Real Wan2pt1 VAE encode + decode round-trip
# ----------------------------------------------------------------------


def _build_backbone_with_real_vae():
    from openwam.model.video_backbone import build_video_backbone

    cfg = {
        "video_backbone": {
            "name": "cosmos_predict25_2b",
            "model_path": str(ASSET_PATH),
            "model_variant": "base/post-trained",
            "text_encoder_path": "/stub",
            "vae": "wan2pt1",
            "sac_mode": "none",
        }
    }
    vb = build_video_backbone("cosmos_predict25_2b", cfg)
    vb.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
    return vb


def test_real_vae_load_and_shape_round_trip(stub_reason1):
    """Encode random pixels through the real Wan2pt1 VAE and decode back."""
    _skip_unless_runnable()
    if not (ASSET_PATH / "tokenizer.pth").exists():
        pytest.skip(f"tokenizer.pth missing at {ASSET_PATH}.")

    vb = _build_backbone_with_real_vae()
    # Use the Wan2pt1VAEInterface facade — the registered `vb.vae` child is the
    # raw inner `WanVAE_` whose encode() takes an explicit `scale`.
    vae = vb._vae_iface
    assert vae is not None, "vae='wan2pt1' should populate the wrapper VAE slot"
    assert vb.vae is not None, "inner WanVAE_ must be registered for state_dict"

    # T_pix=5 → T_lat = 1 + (5-1)//4 = 2; spatial 64→8 (stride 8); z_dim=16.
    pixels = torch.randn(1, 3, 5, 64, 64, dtype=torch.bfloat16, device="cuda:0")
    latents = vae.encode(pixels)
    assert latents.shape == (1, 16, 2, 8, 8), f"encode latent shape={latents.shape}"
    assert torch.isfinite(latents).all()

    recon = vae.decode(latents)
    assert recon.shape == (1, 3, 5, 64, 64), f"decode pixel shape={recon.shape}"
    assert torch.isfinite(recon).all()


def test_real_preprocess_input_full_path(stub_reason1):
    """End-to-end: PIL frames → preprocess_input_for_train → Wan2pt1 encode → latents."""
    _skip_unless_runnable()
    if not (ASSET_PATH / "tokenizer.pth").exists():
        pytest.skip(f"tokenizer.pth missing at {ASSET_PATH}.")

    import numpy as np
    from PIL import Image

    vb = _build_backbone_with_real_vae()
    # 5 frames at 64×64 — Wan2pt1 internal conv3d wants T>=3 after chunking;
    # 5 frames is the smallest size that works through `temporal_window=4`.
    frames = [[Image.fromarray((np.ones((64, 64, 3), dtype=np.uint8) * (i * 30 % 256))) for i in range(5)]]

    out = vb.preprocess_input_for_train(frames=frames, text=["pick up the block"])
    # T_lat = 1 + (5-1)//4 = 2; spatial 64/8=8; C_z=16.
    assert out["input_latents"].shape == (1, 16, 2, 8, 8), f"input_latents shape={out['input_latents'].shape}"
    assert torch.isfinite(out["input_latents"]).all()
    # Stub encoder emits (1, 512, 100352); the DiT-owned crossattn_proj lands at 1024.
    assert out["context"].shape == (1, 512, 1024)
