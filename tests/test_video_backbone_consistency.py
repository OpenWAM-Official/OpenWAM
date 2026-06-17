"""Consistency test: original model_fn_wan_video vs new prepare/run_block/finalize.

Loads real Wan checkpoints and verifies that the decomposed three-step interface
produces bit-exact (within floating-point tolerance) results compared to the
original monolithic forward function.

Requires GPU. Mark with @pytest.mark.gpu.

Tests pure DiT forward only (no VACE, no SP).
These optional modules are additive — if the core DiT path is bit-exact, the
optional paths will be too since run_block faithfully reproduces the same logic.
"""

import gc
import json
import os

import pytest
import torch
from safetensors.torch import load_file

from openwam.model.video_backbone.wan._reference import model_fn_wan_video
from openwam.model.video_backbone.wan_backbone import Wan21

# Override via env vars on machines that mount the checkpoints elsewhere; the
# defaults match the shared dev box but skipif() makes a missing path a skip,
# not a hard failure.
WAN21_VACE_1_3B = os.environ.get("WAN21_VACE_1_3B", "/path/to/Wan2.1-VACE-1.3B")
WAN22_TI2V_5B = os.environ.get("WAN22_TI2V_5B", "/path/to/Wan2.2-TI2V-5B")
CUDA_AVAILABLE = torch.cuda.is_available()


def _load_dit_only(model_dir: str, device: str = "cuda:0"):
    """Load just the DiT model from a checkpoint directory."""
    from openwam.model.video_backbone.wan.models.dit import WanModel

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "t2v")
    has_image_input = model_type in ("i2v", "ti2v", "flf2v")

    dit = WanModel(
        has_image_input=has_image_input,
        patch_size=[1, 2, 2],
        in_dim=cfg["in_dim"],
        dim=cfg["dim"],
        ffn_dim=cfg["ffn_dim"],
        freq_dim=cfg["freq_dim"],
        text_dim=4096,
        out_dim=cfg["out_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        eps=cfg.get("eps", 1e-6),
        seperated_timestep=cfg.get("seperated_timestep", False) or False,
    )

    safetensors_files = sorted(
        f for f in os.listdir(model_dir) if f.startswith("diffusion_pytorch_model") and f.endswith(".safetensors")
    )
    state_dict = {}
    for sf in safetensors_files:
        state_dict.update(load_file(os.path.join(model_dir, sf)))

    try:
        dit.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        stripped = {k.replace("dit.", "", 1) if k.startswith("dit.") else k: v for k, v in state_dict.items()}
        dit.load_state_dict(stripped, strict=False)

    dit = dit.to(device=device, dtype=torch.bfloat16)
    dit.eval()
    return dit


class _FakePipe:
    """Minimal duck-typed pipeline for Wan21."""

    def __init__(self, dit):
        self.dit = dit
        self.use_unified_sequence_parallel = False
        self.in_iteration_models = ["dit"]


def _run_original(dit, inputs):
    with torch.no_grad():
        return model_fn_wan_video(dit=dit, **inputs)


def _run_decomposed(pipe, inputs):
    backbone = Wan21(pipe)
    with torch.no_grad():
        state = backbone.prepare(dit=pipe.dit, **inputs)
        for block_id in range(backbone.num_layers):
            state = backbone.run_block(block_id, state)
        return backbone.finalize(state)


@pytest.mark.gpu
@pytest.mark.skipif(
    not CUDA_AVAILABLE or not os.path.isdir(WAN21_VACE_1_3B),
    reason=f"CUDA unavailable or checkpoint not mounted: {WAN21_VACE_1_3B}",
)
def test_consistency_wan21_vace_1_3b():
    """Wan2.1-VACE-1.3B: standard timestep path (pure DiT, no VACE)."""
    device = "cuda:0"
    dit = _load_dit_only(WAN21_VACE_1_3B, device=device)
    pipe = _FakePipe(dit)

    B, C_in, T, H, W = 1, 16, 5, 12, 20
    inputs = {
        "latents": torch.randn(B, C_in, T, H, W, device=device, dtype=torch.bfloat16),
        "timestep": torch.tensor([500.0], device=device, dtype=torch.bfloat16),
        "context": torch.randn(B, 512, 4096, device=device, dtype=torch.bfloat16),
    }

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    out_original = _run_original(dit, inputs)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    out_decomposed = _run_decomposed(pipe, inputs)

    assert out_original.shape == out_decomposed.shape, f"Shape mismatch: {out_original.shape} vs {out_decomposed.shape}"
    max_diff = (out_original - out_decomposed).abs().max().item()
    assert torch.allclose(out_original, out_decomposed, atol=1e-4, rtol=1e-4), (
        f"Output mismatch! Max diff: {max_diff:.6e}"
    )
    print(f"[1.3B] Max diff: {max_diff:.6e} — PASS")

    del dit, pipe
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.skipif(
    not CUDA_AVAILABLE or not os.path.isdir(WAN22_TI2V_5B),
    reason=f"CUDA unavailable or checkpoint not mounted: {WAN22_TI2V_5B}",
)
def test_consistency_wan22_ti2v_5b():
    """Wan2.2-TI2V-5B: standard + per-token timestep paths."""
    device = "cuda:0"
    dit = _load_dit_only(WAN22_TI2V_5B, device=device)
    pipe = _FakePipe(dit)

    B, C_in, T, H, W = 1, 48, 5, 12, 20
    inputs = {
        "latents": torch.randn(B, C_in, T, H, W, device=device, dtype=torch.bfloat16),
        "timestep": torch.tensor([500.0], device=device, dtype=torch.bfloat16),
        "context": torch.randn(B, 512, 4096, device=device, dtype=torch.bfloat16),
        "clip_feature": torch.randn(B, 1, 1280, device=device, dtype=torch.bfloat16),
    }

    # Standard path
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    out_original = _run_original(dit, inputs)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    out_decomposed = _run_decomposed(pipe, inputs)

    assert out_original.shape == out_decomposed.shape
    max_diff = (out_original - out_decomposed).abs().max().item()
    assert torch.allclose(out_original, out_decomposed, atol=1e-4, rtol=1e-4), (
        f"Output mismatch (standard)! Max diff: {max_diff:.6e}"
    )
    print(f"[5B standard] Max diff: {max_diff:.6e} — PASS")

    # Per-token timestep path
    if getattr(dit, "seperated_timestep", False):
        inputs_fuse = {**inputs, "fuse_vae_embedding_in_latents": True, "num_clean_prefix_frames": 1}

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        out_original_fuse = _run_original(dit, inputs_fuse)

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        out_decomposed_fuse = _run_decomposed(pipe, inputs_fuse)

        assert out_original_fuse.shape == out_decomposed_fuse.shape
        max_diff_fuse = (out_original_fuse - out_decomposed_fuse).abs().max().item()
        assert torch.allclose(out_original_fuse, out_decomposed_fuse, atol=1e-4, rtol=1e-4), (
            f"Output mismatch (fuse)! Max diff: {max_diff_fuse:.6e}"
        )
        print(f"[5B fuse] Max diff: {max_diff_fuse:.6e} — PASS")

    del dit, pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=== Testing Wan2.1-VACE-1.3B ===")
    test_consistency_wan21_vace_1_3b()
    print("\n=== Testing Wan2.2-TI2V-5B ===")
    test_consistency_wan22_ti2v_5b()
    print("\nAll consistency tests passed!")
