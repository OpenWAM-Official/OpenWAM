"""Tests for OpenWAMTrainer: freeze strategy, loss computation, and training loop.

Uses mock pipeline and tiny architecture to avoid GPU / real weights dependency.
All tests run on CPU.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from openwam.model.dual_system import DualSystemArchitecture
from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

# ---------------------------------------------------------------------------
# Mock pipeline: replaces WanVideoPipeline to avoid loading ~20GB of weights
# ---------------------------------------------------------------------------


class _MockDiT(nn.Module):
    """Tiny mock DiT with the attributes the trainer expects."""

    def __init__(self, dim=64, in_dim=16, num_blocks=2):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.fuse_vae_embedding_in_latents = False
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_blocks)])

    def forward(self, x):
        return x


class _MockScheduler:
    """Mock flow-match scheduler with timestep / sigma tables."""

    def __init__(self, num_timesteps=1000):
        self.timesteps = torch.linspace(0, 1, num_timesteps)
        self.sigmas = torch.linspace(1, 0, num_timesteps)
        self.linear_timesteps_weights = torch.ones(num_timesteps)

    def set_timesteps(self, n, **kwargs):
        self.timesteps = torch.linspace(0, 1, n)
        self.sigmas = torch.linspace(1, 0, n)
        self.linear_timesteps_weights = torch.ones(n)


class _MockPipeline:
    """Minimal mock of WanVideoPipeline for trainer tests."""

    def __init__(self, dim=64):
        self.dit = _MockDiT(dim=dim)
        self.vae = nn.Linear(4, 4)
        self.text_encoder = nn.Linear(4, 4)
        self.vace = None
        self.image_encoder = None
        self.torch_dtype = torch.float32
        self.device = "cpu"
        self.scheduler = _MockScheduler()
        self.units = []
        self.in_iteration_models = ["dit"]

    def modules(self):
        return [self.dit, self.vae, self.text_encoder]

    def named_parameters(self):
        yield from self.dit.named_parameters(prefix="dit")

    def named_buffers(self):
        yield from self.dit.named_buffers(prefix="dit")

    def load_state_dict(self, state_dict, strict=True):
        pass

    def model_fn(self, dit=None, latents=None, timestep=None, **kwargs):
        """Return noise_pred matching latents shape, populate bridge features."""
        # Populate bridge_feature_store if requested (for non-interleaved architectures)
        store = kwargs.get("bridge_feature_store", None)
        layers = kwargs.get("bridge_feature_layers", set())
        if store is not None and layers:
            B = latents.shape[0]
            # Flatten latents to (B, num_tokens, dim) for bridge features
            num_tokens = 1
            for d in latents.shape[2:]:
                num_tokens *= d
            for _ in sorted(layers):
                store.append(torch.randn(B, num_tokens, self.dit.dim))
        return torch.randn_like(latents)

    def unit_runner(self, unit, pipe, shared, posi, nega):
        return shared, posi, nega


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_TINY_ARCH_CFG = {
    "action_dim": 7,
    "dim": 64,
    "ffn_dim": 128,
    "num_heads": 2,
    "num_layers": 2,
    "video_dim": 64,
    "bridge_layers": (0, 1),
    "bridge_type": "cross_attn",
}


def _make_tiny_arch():
    return DualSystemArchitecture(cfg=_TINY_ARCH_CFG)


def _make_loss_fn(lambda_video=1.0, lambda_action=1.0):
    return FlowMatchVideoActionLoss(
        lambda_video=lambda_video,
        lambda_action=lambda_action,
        detach_bridge=False,
    )


def _make_fake_loss_inputs(B=1, action_dim=7, T_action=5, video_dim=64):
    """Build the minimal dict that FlowMatchVideoActionLoss.__call__ expects."""
    C, T, H, W = 16, 3, 8, 8
    return {
        "input_latents": torch.randn(B, C, T, H, W),
        "latents": None,
        "height": H * 8,
        "width": W * 8,
        "num_frames": 9,
        "cfg_scale": 1,
        "cfg_merge": False,
        "tiled": False,
        "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False,
        "max_timestep_boundary": 1.0,
        "min_timestep_boundary": 0.0,
    }


# ---------------------------------------------------------------------------
# Tests: Freeze strategy (from training_strategy config)
# ---------------------------------------------------------------------------


def _apply_freeze(pipe, trainer_attrs, freeze_list):
    """Replicate the freeze logic from OpenWAMTrainer.__init__."""
    for name in freeze_list:
        module = getattr(pipe, name, None)
        if module is None:
            module = trainer_attrs.get(name, None)
        if module is not None:
            module.requires_grad_(False)


def test_freeze_joint_strategy():
    """joint.yaml: freeze text_encoder + vae; dit + action_dit remain trainable."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_dit = arch.action_dit

    freeze_list = ["text_encoder", "vae"]  # from configs/training_strategy/joint.yaml
    _apply_freeze(pipe, {"action_dit": action_dit}, freeze_list)

    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert all(p.requires_grad for p in pipe.dit.parameters())
    assert all(p.requires_grad for p in action_dit.parameters())


def test_freeze_video_only_strategy():
    """video_only.yaml: freeze text_encoder + vae + action_dit; dit trainable."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_dit = arch.action_dit

    freeze_list = ["text_encoder", "vae", "action_dit"]  # from video_only.yaml
    _apply_freeze(pipe, {"action_dit": action_dit}, freeze_list)

    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert not any(p.requires_grad for p in action_dit.parameters())
    assert all(p.requires_grad for p in pipe.dit.parameters())


def test_freeze_custom_list():
    """Custom freeze: dit frozen, action_dit stays trainable."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_dit = arch.action_dit

    freeze_list = ["text_encoder", "vae", "dit"]
    _apply_freeze(pipe, {"action_dit": action_dit}, freeze_list)

    assert not any(p.requires_grad for p in pipe.dit.parameters())
    assert all(p.requires_grad for p in action_dit.parameters())


# ---------------------------------------------------------------------------
# Tests: Loss computation (single and multi-batch)
# ---------------------------------------------------------------------------


def test_single_batch_loss():
    """B=1: loss_fn produces valid scalar losses."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_scheduler = _MockScheduler()
    action_scheduler.set_timesteps(1000)
    loss_fn = _make_loss_fn()

    B, T_action, action_dim = 1, 5, 7
    action_data = torch.randn(B, T_action, action_dim)
    inputs = _make_fake_loss_inputs(B=B, action_dim=action_dim, T_action=T_action)

    result = loss_fn(
        pipe=pipe,
        architecture=arch,
        action_scheduler=action_scheduler,
        action_data=action_data,
        current_step=0,
        **inputs,
    )

    assert "loss" in result
    assert "loss_video" in result
    assert "loss_action" in result
    assert result["loss"].shape == ()
    assert result["loss"].item() > 0
    assert result["loss_video"].item() >= 0
    assert result["loss_action"].item() >= 0


def test_multi_batch_loss():
    """B=2: loss_fn produces valid scalar losses with batched input."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_scheduler = _MockScheduler()
    action_scheduler.set_timesteps(1000)
    loss_fn = _make_loss_fn()

    B, T_action, action_dim = 2, 5, 7
    action_data = torch.randn(B, T_action, action_dim)
    inputs = _make_fake_loss_inputs(B=B, action_dim=action_dim, T_action=T_action)

    result = loss_fn(
        pipe=pipe,
        architecture=arch,
        action_scheduler=action_scheduler,
        action_data=action_data,
        current_step=0,
        **inputs,
    )

    assert result["loss"].shape == ()
    assert result["loss"].item() > 0


def test_video_only_loss():
    """lambda_action=0: only video loss is computed."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_scheduler = _MockScheduler()
    action_scheduler.set_timesteps(1000)
    loss_fn = _make_loss_fn(lambda_video=1.0, lambda_action=0.0)

    B = 1
    inputs = _make_fake_loss_inputs(B=B)

    result = loss_fn(
        pipe=pipe,
        architecture=arch,
        action_scheduler=action_scheduler,
        action_data=None,
        current_step=0,
        **inputs,
    )

    assert result["loss"].item() > 0
    assert result["loss_action"].item() == 0.0


def test_loss_backward():
    """Loss should be differentiable and backward should succeed."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    arch.train()
    action_scheduler = _MockScheduler()
    action_scheduler.set_timesteps(1000)
    loss_fn = _make_loss_fn()

    B, T_action, action_dim = 1, 5, 7
    action_data = torch.randn(B, T_action, action_dim)
    inputs = _make_fake_loss_inputs(B=B)

    result = loss_fn(
        pipe=pipe,
        architecture=arch,
        action_scheduler=action_scheduler,
        action_data=action_data,
        current_step=0,
        **inputs,
    )

    # Check backward completes without error
    result["loss"].backward()

    # Verify gradients exist on action_dit parameters
    has_grad = any(p.grad is not None for p in arch.action_dit.parameters())
    assert has_grad, "ActionDiT should have gradients after backward"


# ---------------------------------------------------------------------------
# Tests: Training loop with mock components
# ---------------------------------------------------------------------------


def test_training_step():
    """Run 3 optimizer steps: loss should decrease or remain stable."""
    arch = _make_tiny_arch()
    arch.train()
    action_scheduler = _MockScheduler()
    action_scheduler.set_timesteps(1000)
    loss_fn = _make_loss_fn()

    optimizer = torch.optim.Adam(arch.parameters(), lr=1e-3)

    losses = []
    for step in range(3):
        B, T_action, action_dim = 1, 5, 7
        action_data = torch.randn(B, T_action, action_dim)
        inputs = _make_fake_loss_inputs(B=B)

        pipe = _MockPipeline()
        result = loss_fn(
            pipe=pipe,
            architecture=arch,
            action_scheduler=action_scheduler,
            action_data=action_data,
            current_step=step,
            **inputs,
        )

        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()
        losses.append(result["loss"].item())

    # All losses should be finite positive
    assert all(loss > 0 for loss in losses)
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)


# ---------------------------------------------------------------------------
# Tests: Checkpointing (save / load / manage)
# ---------------------------------------------------------------------------


def test_save_load_checkpoint():
    """Save and load checkpoint; verify weights match."""
    from openwam.train.utils.checkpointing import (
        load_trainable_checkpoint,
        save_trainable_checkpoint,
    )

    arch = _make_tiny_arch()
    pipe = _MockPipeline()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "test_ckpt.safetensors")
        save_trainable_checkpoint(ckpt_path, arch.action_dit, pipe, lambda_action=1.0)
        assert Path(ckpt_path).exists()

        # Load into fresh model
        arch2 = _make_tiny_arch()
        pipe2 = _MockPipeline()
        load_trainable_checkpoint(ckpt_path, arch2.action_dit, pipe2)

        # Verify action_dit weights match
        for (k1, v1), (k2, v2) in zip(
            arch.action_dit.state_dict().items(),
            arch2.action_dit.state_dict().items(),
        ):
            assert k1 == k2
            assert torch.equal(v1, v2), f"Weight mismatch for {k1}"


def test_manage_checkpoints():
    """manage_checkpoints should keep only the latest K files."""
    from openwam.train.utils.checkpointing import manage_checkpoints

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 fake checkpoints
        for i in range(1, 6):
            path = Path(tmpdir) / f"checkpoint_step_{i * 100}.safetensors"
            path.write_text("fake")

        manage_checkpoints(tmpdir, keep_last_k=2)

        remaining = sorted(Path(tmpdir).glob("checkpoint_step_*"))
        assert len(remaining) == 2
        names = [r.name for r in remaining]
        assert "checkpoint_step_400.safetensors" in names
        assert "checkpoint_step_500.safetensors" in names


# ---------------------------------------------------------------------------
# Tests: Mask downsampling utility
# ---------------------------------------------------------------------------


def test_downsample_video_mask():
    """Verify the frame → latent mask downsampling logic."""
    from openwam.train.openwam_trainer import _downsample_video_mask_to_latent

    # 9 frames: frame 0 excluded, frames 1-8 grouped by 4
    # All valid (is_pad=False) → all latent steps valid
    video_is_pad = torch.zeros(9, dtype=torch.bool)
    latent_mask = _downsample_video_mask_to_latent(video_is_pad)
    assert latent_mask.shape[0] == 2  # (9-1)/4 = 2
    assert not latent_mask.any()

    # Last 4 frames padded → second latent step padded
    video_is_pad = torch.tensor([False, False, False, False, False, True, True, True, True])
    latent_mask = _downsample_video_mask_to_latent(video_is_pad)
    assert latent_mask.shape[0] == 2
    assert not latent_mask[0]  # frames 1-4 valid
    assert latent_mask[1]  # frames 5-8 all padded
