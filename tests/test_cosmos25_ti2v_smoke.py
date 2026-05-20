"""CPU smoke for the Cosmos25 TI2V (text + first-frame → video) path.

Mirrors Wan TI2V semantics on the Cosmos side without loading
``cosmos_predict2``:

- Wrapper ``preprocess_input`` consumes ``ref_images``, VAE-encodes one
  reference frame per sample, and emits ``first_frame_latents`` +
  ``condition_mask`` + ``num_clean_prefix_frames=1``.
- Wrapper ``prepare_block_loop`` zeros out frame-0 timestep when
  ``condition_mask`` marks a clean prefix (LVG-native equivalent of Wan's
  ``fuse_vae_embedding_in_latents`` per-token AdaLN).
- Adapter ``prepare_inputs_for_inference`` no longer rejects
  ``first_frame_image``; it encodes the reference and overwrites
  ``latents[:, :, 0:1]`` so the denoise loop starts from the right state
  (mirrors ``wan/pipeline.py:387-388``).

All fakes here mimic the minimum upstream surface; no GPU, no real Cosmos
weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from openwam.model.inference_inputs import InferenceInputs
from openwam.model.video_backbone.cosmos25 import (
    Cosmos25VideoBackbone,
    CosmosFlowSchedulerAdapter,
)
from openwam.model.video_backbone.cosmos25.pipeline_wrapper import Cosmos25PipelineWrapper

# ----------------------------------------------------------------------
# Fakes — minimal upstream-DiT and VAE surfaces
# ----------------------------------------------------------------------


class _RecordingTEmbedder(nn.Module):
    """Captures the timestep tensor passed by ``prepare_block_loop``.

    The recorded ``last_input`` lets the per-token timestep test assert
    that frame-0 was zeroed under TI2V conditioning.
    """

    def __init__(self, *, dim: int, lora_dim: int) -> None:
        super().__init__()
        self.t_proj = nn.Linear(1, dim, bias=False)
        self.adaln_proj = nn.Linear(1, lora_dim, bias=False)
        self.last_input: torch.Tensor | None = None

    def forward(self, timesteps_B_T):
        self.last_input = timesteps_B_T.detach().clone()
        t_f = timesteps_B_T.to(torch.float32).unsqueeze(-1)
        return self.t_proj(t_f), self.adaln_proj(t_f)


class _FakeBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x_B_T_H_W_D,
        t_embedding_B_T_D,
        crossattn_emb,
        *,
        rope_emb_L_1_1_D=None,
        adaln_lora_B_T_3D=None,
        extra_per_block_pos_emb=None,
    ):
        ctx_mean = crossattn_emb.mean(dim=1)
        return self.proj(x_B_T_H_W_D) + 0.0 * ctx_mean.sum()


class _FakeFinalLayer(nn.Module):
    def __init__(self, dim: int, out_per_patch: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, out_per_patch, bias=False)

    def forward(self, x_B_T_H_W_D, t_embedding_B_T_D, *, adaln_lora_B_T_3D=None):
        return self.linear(x_B_T_H_W_D)


class _FakeMiniDIT(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 32,
        num_blocks: int = 2,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        out_channels: int = 16,
        ctx_dim_post: int = 24,
        ctx_dim_pre: int = 64,
    ) -> None:
        super().__init__()
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.timestep_scale = 1.0
        self.concat_padding_mask = True
        self.use_crossattn_projection = True
        self.crossattn_proj_in_channels = ctx_dim_pre
        self.crossattn_proj = nn.Sequential(nn.Linear(ctx_dim_pre, ctx_dim_post, bias=True))
        self.t_embedder = _RecordingTEmbedder(dim=dim, lora_dim=dim * 3)
        self.t_embedding_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([_FakeBlock(dim) for _ in range(num_blocks)])
        self.final_layer = _FakeFinalLayer(
            dim=dim, out_per_patch=patch_spatial * patch_spatial * patch_temporal * out_channels
        )
        self._patched_dim = dim
        self._out_channels = out_channels

    def prepare_embedded_sequence(self, x_B_C_T_H_W, *, fps=None, padding_mask=None):
        if self.concat_padding_mask:
            assert padding_mask is not None
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)],
                dim=1,
            )
        B, C, T, H, W = x_B_C_T_H_W.shape
        f = T // self.patch_temporal
        h = H // self.patch_spatial
        w = W // self.patch_spatial
        x_B_T_H_W_D = torch.zeros(B, f, h, w, self._patched_dim, dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
        rope = torch.zeros(f * h * w, 1, 1, self._patched_dim, dtype=torch.float32, device=x_B_C_T_H_W.device)
        return x_B_T_H_W_D, rope, None

    def unpatchify(self, x_B_T_H_W_O):
        B, f, h, w, O = x_B_T_H_W_O.shape
        out = x_B_T_H_W_O.reshape(
            B, f, h, w, self._out_channels, self.patch_spatial, self.patch_spatial, self.patch_temporal
        )
        out = out.permute(0, 4, 1, 7, 2, 5, 3, 6).contiguous()
        out = out.reshape(
            B, self._out_channels, f * self.patch_temporal, h * self.patch_spatial, w * self.patch_spatial
        )
        return out


class _FakeWanVAEModel:
    def __init__(self) -> None:
        self.model: nn.Module = nn.Linear(1, 1)
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16


class _DeterministicVAE:
    """Stride-4×8 VAE stub that returns latents derived from the input mean.

    Determinism lets the inference test assert ``latents[:, :, 0:1]`` was
    overwritten by the *encoded reference*, not by random noise.
    """

    def __init__(self) -> None:
        self.model = _FakeWanVAEModel()
        self.encode_calls = 0

    def encode(self, video):
        self.encode_calls += 1
        B, C, T, H, W = video.shape
        assert C == 3
        T_lat = 1 + (T - 1) // 4
        mean_per_sample = video.mean(dim=(1, 2, 3, 4)).view(B, 1, 1, 1, 1)
        base = torch.full(
            (B, 16, T_lat, H // 8, W // 8),
            0.0,
            dtype=video.dtype,
            device=video.device,
        )
        return base + mean_per_sample


def _make_pil_frame(value: int, *, H: int = 32, W: int = 32) -> Image.Image:
    return Image.fromarray((np.ones((H, W, 3), dtype=np.uint8) * (value % 256)))


def _build_wrapper(*, with_vae: bool = True) -> Cosmos25PipelineWrapper:
    torch.manual_seed(0)
    net = _FakeMiniDIT(dim=32, num_blocks=2, patch_spatial=2, patch_temporal=1, out_channels=16)
    return Cosmos25PipelineWrapper(
        net=net,
        vae=_DeterministicVAE() if with_vae else None,
        text_encoder=None,
        dim=32,
        num_layers=2,
        num_heads=4,
        head_dim=8,
        context_dim=24,
        flow_shift=5.0,
    )


# ----------------------------------------------------------------------
# A1 — preprocess_input emits TI2V keys when ref_images is present
# ----------------------------------------------------------------------


def test_preprocess_input_emits_ti2v_keys_with_ref_images():
    wrapper = _build_wrapper(with_vae=True)
    B, T_lat, H_lat, W_lat = 1, 2, 4, 4
    input_latents = torch.randn(B, 16, T_lat, H_lat, W_lat)
    pre_text = torch.randn(B, 8, 24)
    ref = _make_pil_frame(value=42, H=H_lat * 8, W=W_lat * 8)

    out = wrapper.preprocess_input(
        input_latents=input_latents,
        pre_encoded_text=pre_text,
        ref_images=[ref],
    )

    assert "first_frame_latents" in out, "TI2V key must be emitted when ref_images is present."
    assert "condition_mask" in out
    assert out["num_clean_prefix_frames"] == 1
    ffl = out["first_frame_latents"]
    assert ffl.shape == (B, 16, 1, H_lat, W_lat), f"got {tuple(ffl.shape)}"
    cm = out["condition_mask"]
    assert cm.shape == (B, 1, T_lat, H_lat, W_lat), f"got {tuple(cm.shape)}"
    assert torch.all(cm[:, :, 0] == 1.0), "condition_mask must mark frame 0 as clean."
    assert torch.all(cm[:, :, 1:] == 0.0), "condition_mask must leave tail frames as 0."


def test_preprocess_input_no_ref_no_ti2v_keys():
    """Pure T2V batches must not pick up any TI2V keys (regression guard)."""
    wrapper = _build_wrapper(with_vae=True)
    input_latents = torch.randn(1, 16, 2, 4, 4)
    pre_text = torch.randn(1, 8, 24)
    out = wrapper.preprocess_input(input_latents=input_latents, pre_encoded_text=pre_text)
    assert "first_frame_latents" not in out
    assert "condition_mask" not in out


def test_preprocess_input_with_ref_requires_vae():
    """``ref_images`` present but no VAE configured ⇒ explicit error."""
    wrapper = _build_wrapper(with_vae=False)
    input_latents = torch.randn(1, 16, 2, 4, 4)
    pre_text = torch.randn(1, 8, 24)
    ref = _make_pil_frame(value=7, H=32, W=32)
    with pytest.raises(RuntimeError, match="VAE"):
        wrapper.preprocess_input(input_latents=input_latents, pre_encoded_text=pre_text, ref_images=[ref])


# ----------------------------------------------------------------------
# A2 — prepare_block_loop forces frame-0 timestep = 0 under TI2V
# ----------------------------------------------------------------------


def test_prepare_block_loop_zeros_frame0_timestep_under_ti2v():
    wrapper = _build_wrapper(with_vae=True)
    B, T_lat, H_lat, W_lat = 1, 3, 4, 4
    latents = torch.randn(B, 16, T_lat, H_lat, W_lat)
    context = torch.randn(B, 8, 24)
    timestep = torch.tensor([777], dtype=torch.long)
    condition_mask = torch.zeros(B, 1, T_lat, H_lat, W_lat)
    condition_mask[:, :, 0] = 1.0

    wrapper.prepare_block_loop(
        input_latents=latents,
        context=context,
        timestep=timestep,
        condition_mask=condition_mask,
    )
    captured = wrapper.net.t_embedder.last_input
    assert captured is not None
    assert captured.shape == (B, T_lat), (
        f"Expected per-token timestep shape (B={B}, T={T_lat}), got {tuple(captured.shape)}."
    )
    assert torch.all(captured[:, 0] == 0), "Frame-0 timestep must be zero under TI2V conditioning."
    assert torch.all(captured[:, 1:] == 777), "Tail-frame timesteps must keep the sampled value."


def test_prepare_block_loop_keeps_legacy_shape_for_t2v():
    """Without TI2V conditioning the wrapper retains the ``(B, 1)`` broadcast.

    Regression guard so the new branch doesn't perturb the T2V code path.
    """
    wrapper = _build_wrapper(with_vae=True)
    B = 1
    latents = torch.randn(B, 16, 2, 4, 4)
    context = torch.randn(B, 8, 24)
    timestep = torch.tensor([314], dtype=torch.long)
    wrapper.prepare_block_loop(input_latents=latents, context=context, timestep=timestep)
    captured = wrapper.net.t_embedder.last_input
    assert captured is not None
    assert captured.shape == (B, 1), (
        f"T2V must keep legacy (B, 1) timestep broadcast; got {tuple(captured.shape)}."
    )
    assert int(captured[0, 0]) == 314


# ----------------------------------------------------------------------
# B2 — adapter prepare_inputs_for_inference accepts first_frame_image
# ----------------------------------------------------------------------


def _build_backbone_with_wrapper() -> Cosmos25VideoBackbone:
    wrapper = _build_wrapper(with_vae=True)
    return Cosmos25VideoBackbone(
        pipeline=wrapper,
        dim=wrapper.dim,
        num_layers=wrapper.num_layers,
        num_heads=wrapper.num_heads,
        head_dim=wrapper.head_dim,
        context_dim=wrapper.context_dim,
        scheduler=CosmosFlowSchedulerAdapter(flow_shift=5.0),
        freeze=False,
    )


def test_prepare_inputs_for_inference_accepts_first_frame_image():
    bb = _build_backbone_with_wrapper()
    H, W, num_frames = 32, 32, 5  # 5 frames ⇒ T_lat = 1 + (5-1)//4 = 2
    img = _make_pil_frame(value=128, H=H, W=W)
    pre_text = torch.randn(1, 8, 24)

    out = bb.prepare_inputs_for_inference(
        InferenceInputs(
            prompt="ignored",
            first_frame_image=img,
            pre_encoded_text=pre_text,
            num_frames=num_frames,
            height=H,
            width=W,
        )
    )

    # TI2V keys propagate
    assert out["first_frame_latents"] is not None
    assert out["num_clean_prefix_frames"] == 1
    assert "condition_mask" in out
    cm = out["condition_mask"]
    assert cm.shape == (1, 1, 2, H // 8, W // 8)
    assert torch.all(cm[:, :, 0] == 1.0)
    assert torch.all(cm[:, :, 1:] == 0.0)

    # The denoise initial state has the encoded reference written into frame 0.
    latents = out["latents"]
    ffl = out["first_frame_latents"]
    assert latents.shape == (1, 16, 2, H // 8, W // 8)
    assert torch.allclose(latents[:, :, 0:1], ffl, atol=0.0), (
        "Adapter must overwrite latents[:, :, 0:1] with the encoded first-frame latent."
    )


def test_prepare_inputs_for_inference_no_first_frame_is_pure_t2v():
    """Without ``first_frame_image`` the adapter must not introduce TI2V keys."""
    bb = _build_backbone_with_wrapper()
    pre_text = torch.randn(1, 8, 24)
    out = bb.prepare_inputs_for_inference(
        InferenceInputs(
            prompt="ignored",
            first_frame_image=None,
            pre_encoded_text=pre_text,
            num_frames=5,
            height=32,
            width=32,
        )
    )
    assert out["first_frame_latents"] is None
    assert out["num_clean_prefix_frames"] == 0
    # condition_mask should be absent (so the wrapper synthesizes a zero mask).
    assert "condition_mask" not in out
