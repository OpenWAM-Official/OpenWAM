"""CPU smoke for ``pre_encoded_text`` threading through
``BaseWAMArchitecture.prepare_inputs`` → ``video_backbone.preprocess_input``.

When a dataloader/transform attaches a cached Reason1 embedding to each
sample dict under the ``pre_encoded_text`` key, the architecture must
stack it across the batch and forward it as a kwarg to ``preprocess`` so
the Cosmos25 wrapper uses the cache. This file exercises:

* Real cached tensor flows through unchanged (``inputs["context"]``
  is the supplied embedding, not a freshly randomized one).
* Wrapper sees the cached embedding when batch size > 1.
* All-or-nothing per batch — mixing samples-with and samples-without
  ``pre_encoded_text`` raises a clear ``ValueError``.
* Inconsistent L across the batch raises (padded variant is deferred).

A minimal in-file fake VAE encodes the PIL frames into a shape-correct
latents tensor; the test focus is the text plumbing, not the VAE.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from openwam.model.video_backbone.cosmos25 import Cosmos25VideoBackbone, CosmosFlowSchedulerAdapter
from openwam.model.video_backbone.cosmos25.pipeline_wrapper import Cosmos25PipelineWrapper


class _ParamOnlyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))


class _FakeVAE:
    """Tiny stand-in for ``Wan2pt1VAEInterface`` — encodes PIL-derived
    ``(B, 3, T, H, W)`` to a shape-correct latents tensor."""

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = video.shape
        assert C == 3
        T_lat = 1 + (T - 1) // 4
        return torch.randn(B, 16, T_lat, H // 8, W // 8, dtype=video.dtype, device=video.device)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        B, C, T_lat, H_lat, W_lat = latents.shape
        return torch.randn(B, 3, (T_lat - 1) * 4 + 1, H_lat * 8, W_lat * 8, dtype=latents.dtype, device=latents.device)


def _build_cosmos_backbone() -> Cosmos25VideoBackbone:
    pipe = Cosmos25PipelineWrapper(
        net=_ParamOnlyNet(),
        vae=_FakeVAE(),
        text_encoder=None,
        dim=2048,
        num_layers=28,
        num_heads=16,
        head_dim=128,
        context_dim=1024,
        flow_shift=5.0,
    )
    return Cosmos25VideoBackbone(
        pipeline=pipe,
        dim=2048,
        num_layers=28,
        num_heads=16,
        head_dim=128,
        context_dim=1024,
        scheduler=CosmosFlowSchedulerAdapter(flow_shift=5.0),
        freeze=True,
    )


def _build_arch_with_cosmos():
    from openwam.model.architectures.dual_system import DualSystemCrossAttnArchitecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": 20,
        "dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 16,
        "attn_head_dim": 64,
        "use_proprioception": True,
        "state_dim": 20,
        "video_dim": 2048,
        "text_dim": 1024,
        "bridge_interval": 1,
        "num_dit_layers": 28,
    }
    arch = DualSystemCrossAttnArchitecture(cfg=cfg)
    arch.video_backbone = _build_cosmos_backbone()
    # BaseWAMArchitecture defaults to (bfloat16, cuda); pin to CPU/float32 so
    # this smoke test runs on a CUDA-less CI runner.
    arch.set_dtype_device(torch.float32, torch.device("cpu"))
    return arch


def _make_sample(*, with_text: bool = False, L: int = 8, D: int = 1024):
    """RoboTwin-shape stub with real PIL frames so the fake VAE has a
    well-formed ``(B, 3, T, H, W)`` to encode. Optionally carries
    ``pre_encoded_text``."""
    import PIL.Image

    sample = {
        "video": [PIL.Image.fromarray((np.zeros((32, 32, 3), dtype=np.uint8))) for _ in range(5)],
        "prompt": "smoke",
        "action": np.zeros((12, 20), dtype=np.float32),
        "proprio": np.zeros((20,), dtype=np.float32),
        "action_mask": np.ones((12,), dtype=bool),
        "video_mask": np.ones((5,), dtype=bool),
    }
    if with_text:
        # (L, D) — natural shape of a per-prompt safetensors cache file.
        sample["pre_encoded_text"] = torch.randn(L, D)
    return sample


def _cast_to_arch(arch, t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=arch.dtype, device=arch.device)


def test_pre_encoded_text_threads_through_to_context():
    arch = _build_arch_with_cosmos()
    sample = _make_sample(with_text=True, L=11, D=1024)
    expected = _cast_to_arch(arch, sample["pre_encoded_text"].clone())

    inputs = arch.prepare_inputs([sample])

    assert inputs["context"].shape == (1, 11, 1024), "L from the cached embedding must be preserved end-to-end."
    # Values must match — wrapper passed the supplied tensor through to context.
    assert torch.equal(inputs["context"][0], expected)


def test_pre_encoded_text_batched_across_samples():
    arch = _build_arch_with_cosmos()
    s0 = _make_sample(with_text=True, L=8, D=1024)
    s1 = _make_sample(with_text=True, L=8, D=1024)
    e0 = _cast_to_arch(arch, s0["pre_encoded_text"].clone())
    e1 = _cast_to_arch(arch, s1["pre_encoded_text"].clone())

    inputs = arch.prepare_inputs([s0, s1])

    assert inputs["context"].shape == (2, 8, 1024)
    assert torch.equal(inputs["context"][0], e0)
    assert torch.equal(inputs["context"][1], e1)


def test_pre_encoded_text_accepts_1_L_D_shape():
    """safetensors writers sometimes emit (1, L, D) — be defensive and squeeze."""
    arch = _build_arch_with_cosmos()
    sample = _make_sample(with_text=False)
    sample["pre_encoded_text"] = torch.randn(1, 7, 1024)
    expected = _cast_to_arch(arch, sample["pre_encoded_text"][0].clone())

    inputs = arch.prepare_inputs([sample])

    assert inputs["context"].shape == (1, 7, 1024)
    assert torch.equal(inputs["context"][0], expected)


def test_missing_pre_encoded_text_raises():
    """With no cache, no live encoder, and no synthetic fallback, the wrapper
    must raise a clear ValueError instead of silently producing garbage."""
    arch = _build_arch_with_cosmos()
    with pytest.raises(ValueError, match="pre_encoded_text"):
        arch.prepare_inputs([_make_sample(with_text=False)])


def test_mixed_pre_encoded_text_in_batch_raises():
    arch = _build_arch_with_cosmos()
    with pytest.raises(ValueError, match="Mixed pre_encoded_text"):
        arch.prepare_inputs([_make_sample(with_text=True), _make_sample(with_text=False)])


def test_inconsistent_L_in_pre_encoded_text_raises():
    arch = _build_arch_with_cosmos()
    s0 = _make_sample(with_text=True, L=8)
    s1 = _make_sample(with_text=True, L=12)
    with pytest.raises(ValueError, match="Inconsistent sequence length"):
        arch.prepare_inputs([s0, s1])


def test_pre_encoded_text_numpy_array_accepted():
    """Dataset transforms may produce numpy arrays before tensor conversion."""
    arch = _build_arch_with_cosmos()
    sample = _make_sample(with_text=False)
    sample["pre_encoded_text"] = np.random.randn(9, 1024).astype(np.float32)
    expected = _cast_to_arch(arch, torch.from_numpy(sample["pre_encoded_text"]))

    inputs = arch.prepare_inputs([sample])

    assert inputs["context"].shape == (1, 9, 1024)
    assert torch.equal(inputs["context"][0], expected)
