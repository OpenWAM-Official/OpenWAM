"""GPU smoke for the real Cosmos-Predict2.5-2B inference path.

Exercises ``preprocess_input_for_inference`` (T2V + CFG) on real DiT/VAE
weights (Reason1 is stubbed — 512-token pre-projection output, projected
through the real DiT-owned ``crossattn_proj``) and feeds its
``inputs_shared`` through the real DiT block loop. Complements
``test_cosmos_predict25_real_load.py`` (the train/forward chain).

Skip conditions identical to ``test_cosmos_predict25_real_load.py``.
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


def _build_backbone():
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


def test_real_inference_t2v_drives_block_loop(stub_reason1):
    """T2V (cfg_scale=1.0): inference dict drives the real DiT."""
    _skip_unless_runnable()
    vb = _build_backbone()

    inputs_shared = vb.preprocess_input_for_inference(
        prompt="a robot stacking blocks",
        num_frames=5,
        height=64,
        width=64,
        num_inference_steps=2,
        cfg_scale=1.0,
    )

    assert inputs_shared["uncond_context"] is None  # no CFG at scale 1.0
    assert inputs_shared["latents"].shape == (1, 16, 2, 8, 8)
    # Stub emits (1, 512, 100352); the real crossattn_proj lands at 1024.
    assert inputs_shared["context"].shape == (1, 512, 1024)

    # Feed the inference latents/context through the real block loop.
    timestep = torch.randint(0, 1000, (1,), device="cuda:0").to(torch.bfloat16)
    state = vb.prepare(
        input_latents=inputs_shared["latents"],
        context=inputs_shared["context"],
        timestep=timestep,
    )
    for i in range(vb.num_layers):
        state = vb.run_block(i, state)
    out = vb.finalize(state)
    assert out.shape == (1, 16, 2, 8, 8)
    assert torch.isfinite(out).all()


def test_real_inference_cfg_materializes_uncond_context(stub_reason1):
    """cfg_scale>1.0 materialises uncond via ``text_encoder("")`` matching the
    cond context shape/dtype/device."""
    _skip_unless_runnable()
    vb = _build_backbone()

    inputs_shared = vb.preprocess_input_for_inference(
        prompt="a robot stacking blocks",
        num_frames=5,
        height=64,
        width=64,
        cfg_scale=2.0,
    )
    uncond = inputs_shared["uncond_context"]
    cond = inputs_shared["context"]
    assert uncond is not None
    assert uncond.shape == cond.shape  # (1, 512, 1024)
    assert uncond.dtype == cond.dtype and uncond.device == cond.device
    assert torch.isfinite(uncond.float()).all()
