"""GPU real-weights load test for cosmos3_edge (predict2.5 real_load parity).

Gated on CUDA + the Cosmos3-Edge bundle + diffusers. Asserts the released
geometry, runs preprocess_input_for_train end-to-end on synthetic PIL frames,
drives the full 28-block loop shape-conservingly, and round-trips the VAE.
"""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

ASSET_PATH = Path(os.environ.get("COSMOS3_EDGE_ASSET_PATH", "/path/to/assets/Cosmos3-Edge"))


def _skip_unless_runnable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if not ASSET_PATH.exists():
        pytest.skip(f"Cosmos3-Edge bundle missing at {ASSET_PATH}.")
    pytest.importorskip("diffusers")


@pytest.fixture(scope="module")
def vb():
    _skip_unless_runnable()
    from openwam.model.video_backbone import build_video_backbone

    cfg = {"model": {"video_backbone": {"name": "cosmos3_edge", "model_path": str(ASSET_PATH)}}}
    backbone = build_video_backbone("cosmos3_edge", cfg)
    backbone.set_dtype_device(torch.bfloat16, torch.device("cuda"))
    return backbone


def test_released_geometry(vb):
    assert vb.dim == 2048
    assert vb.num_layers == 28
    assert vb.num_heads == 16
    assert vb.head_dim == 128
    assert vb.text_dim == 2048
    assert vb.temporal_compression == 4
    assert vb.causal_temporal is True
    assert getattr(vb.dit, "lm_head", None) is None


def test_preprocess_and_block_loop_shapes(vb):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    clip = [Image.fromarray(rng.integers(0, 255, (96, 160, 3), dtype=np.uint8)) for _ in range(9)]
    with torch.no_grad():
        inputs = vb.preprocess_input_for_train(frames=[clip], text=["pick up the bottle"])
        lat = inputs["input_latents"]
        assert lat.shape[0] == 1 and lat.shape[1] == 48
        assert lat.shape[2] == 1 + (9 - 1) // 4  # 3 latent frames
        assert lat.shape[3] == 96 // 16 and lat.shape[4] == 160 // 16
        assert inputs["context"].shape[-1] == 2048
        assert len(inputs["und_kv"]) == 28
        assert inputs["num_clean_prefix_frames"] == 1

        state = vb.prepare(
            latents=lat.to(vb.dtype),
            timestep=torch.tensor([500.0], device=lat.device),
            context=inputs["context"],
            und_mask=inputs["und_mask"],
            und_kv=inputs["und_kv"],
            vision_positions=inputs["vision_positions"],
            num_clean_prefix_frames=1,
        )
        assert state.prefix_kv_len == inputs["context"].shape[1]
        for i in range(vb.num_layers):
            state = vb.run_block(i, state)
        out = vb.finalize(state)
    assert out.shape == lat.shape
    assert torch.isfinite(out.float()).all()


def test_vae_roundtrip(vb):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(1)
    clip = [Image.fromarray(rng.integers(0, 255, (96, 160, 3), dtype=np.uint8)) for _ in range(5)]
    with torch.no_grad():
        inputs = vb.preprocess_input_for_train(frames=[clip], text=["x"])
        frames = vb.decode_video(inputs["input_latents"].to(vb.dtype))
    assert len(frames) == 5
    assert frames[0].size == (160, 96)
