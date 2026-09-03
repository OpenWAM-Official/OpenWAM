"""Tests for OpenWAMTrainer: freeze strategy, loss computation, and training loop.

Uses mock pipeline and tiny architecture to avoid GPU / real weights dependency.
All tests run on CPU.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from openwam.model.architectures.dual_system import DualSystemCrossAttnArchitecture
from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone

# ---------------------------------------------------------------------------
# Mock pipeline: replaces WanVideoPipeline to avoid loading ~20GB of weights
# ---------------------------------------------------------------------------


class _MockDiT(nn.Module):
    """Tiny mock DiT with the attributes the trainer + adapter expect."""

    def __init__(self, dim=64, in_dim=16, num_blocks=2):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.fuse_vae_embedding_in_latents = False
        self.seperated_timestep = False
        self.freq_dim = dim
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_blocks)])

    def forward(self, x):
        return x


class _MockScheduler:
    """Mock flow-match scheduler with timestep / sigma tables."""

    num_train_timesteps = 1000

    def __init__(self, num_timesteps=1000):
        self.timesteps = torch.linspace(0, 1, num_timesteps)
        self.sigmas = torch.linspace(1, 0, num_timesteps)
        self.linear_timesteps_weights = torch.ones(num_timesteps)

    def set_timesteps(self, n, **kwargs):
        self.timesteps = torch.linspace(0, 1, n)
        self.sigmas = torch.linspace(1, 0, n)
        self.linear_timesteps_weights = torch.ones(n)

    def add_noise(self, original, noise, sigma):
        return (1 - sigma) * original + sigma * noise

    def training_target(self, original, noise):
        return noise - original

    def training_weight(self, timestep_ids):
        return self.linear_timesteps_weights[timestep_ids]

    def flow_step(self, pred, sigma, sigma_next, sample):
        return sample + pred * (sigma_next - sigma)


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
        """Return noise_pred matching latents shape; invoke architecture
        callbacks so bridge features get populated for cross_attn / interleaved
        paths. Mirrors how the real model_fn_wan_video calls the hooks.
        """
        # Compute the (B, num_tokens, dim) layout the real model_fn produces
        # after patchify + flatten.
        B = latents.shape[0]
        num_tokens = 1
        for d in latents.shape[2:]:
            num_tokens *= d
        x = torch.randn(B, num_tokens, self.dit.dim)

        on_before_blocks = kwargs.get("on_before_blocks")
        on_after_block = kwargs.get("on_after_block")
        on_after_blocks = kwargs.get("on_after_blocks")
        if on_before_blocks is not None:
            x, _, _ = on_before_blocks(x, torch.zeros(B, 6, self.dit.dim), None)
        for block_id, _ in enumerate(self.dit.blocks):
            if on_after_block is not None:
                x = on_after_block(block_id, x)
        if on_after_blocks is not None:
            x = on_after_blocks(x)
        return torch.randn_like(latents)

    def unit_runner(self, unit, pipe, shared, posi, nega):
        return shared, posi, nega


class _MockVideoBackbone(VideoBackbone):
    """Minimal VideoBackbone implementing the ABC for tests."""

    def __init__(self, dim=64, num_layers=2, num_heads=4):
        super().__init__()
        self._dim = dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._scheduler = _MockScheduler()
        self._dit = nn.Linear(dim, dim)
        self._vae = nn.Linear(4, 4)
        self._text_encoder = nn.Linear(4, 4)
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        # ABC-property backings (mirror Wan native so legacy mask code stays
        # bit-for-bit identical).
        self._dit_patch_size = (1, 2, 2)
        self._temporal_compression, self._causal_temporal = 4, True
        self.last_injected = None
        self.last_extracted = None

    @classmethod
    def from_pretrained(cls, source, **kw):
        return cls()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._dim // self._num_heads

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def submodule_names(self) -> list:
        return ["dit", "vae", "text_encoder"]

    @property
    def video_attention_mask_mode(self) -> str:
        return getattr(self, "_video_attention_mask_mode", "bidirectional")

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode: str) -> None:
        self._video_attention_mask_mode = mode

    def build_video_to_video_mask(self, video_seq_len: int, video_tokens_per_frame: int, device: torch.device):
        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        if self.video_attention_mask_mode == "first_frame_causal":
            mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            mask[:first_frame_tokens, first_frame_tokens:] = False
            return mask
        raise ValueError(f"Unsupported mock video_attention_mask_mode '{self.video_attention_mask_mode}'")

    def prepare(self, **inputs) -> BlockLoopState:
        latents = inputs["latents"]
        B = latents.shape[0]
        # Derive (f, h, w) from latent dims so 3D RoPE can index correctly.
        f, h, w = (latents.shape[2], latents.shape[3], latents.shape[4]) if latents.ndim >= 5 else (1, 1, 1)
        num_tokens = f * h * w
        x = torch.randn(B, num_tokens, self._dim)
        # Per-token t_mod (B, T, 6, dim) mirrors Wan2.2-TI2V-5B's
        # ``seperated_timestep=True + fuse_vae_embedding_in_latents=True`` mode,
        # which is what production runs and what SingleSystem's forward
        # fail-fast enforces.
        t_mod = torch.zeros(B, num_tokens, 6, self._dim)
        freq_dim = self._dim // 2
        freqs = torch.polar(torch.ones(num_tokens, 1, freq_dim), torch.zeros(num_tokens, 1, freq_dim))
        context = torch.randn(B, 4, self._dim)
        context_mask = torch.ones(B, 4, dtype=torch.bool)
        return BlockLoopState(
            hidden_states=x,
            time_mod=t_mod,
            rope_freqs=freqs,
            context=context,
            context_mask=context_mask,
            grid_frames=f,
            grid_height=h,
            grid_width=w,
            extras={},
        )

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        return state

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState):
        """Tiny pre-attn split for split-attention path testing.

        Returns x as q/k/v (no projection, no RoPE) so the MoT driver can
        exercise concat/split shape handling end-to-end. ``post_state`` carries
        the residual and a no-op modulation that ``post_attn_at_layer`` consumes.
        """
        residual = state.hidden_states
        post_state = {"residual": residual}
        return residual, residual, residual, post_state

    def post_attn_at_layer(self, layer_id: int, state: BlockLoopState, attn_out, post_state: dict) -> BlockLoopState:
        """Trivial post-attn: residual + attn_out so the driver round-trips.

        No cross-attn, no FFN, no VACE residuals — those are the real
        backbone's responsibility and aren't exercised by the trainer mock.
        """
        state.hidden_states = post_state["residual"] + attn_out
        return state

    def finalize(self, state: BlockLoopState):
        B = state.hidden_states.shape[0]
        return torch.randn(B, 16, 3, 8, 8)

    def inject_action_tokens(self, state, action_tokens, n_action, *, timestep=None):
        state.hidden_states = torch.cat([state.hidden_states, action_tokens.to(state.hidden_states.dtype)], dim=1)
        return state

    def inject_shared_tokens(
        self,
        state,
        action_tokens,
        n_action,
        *,
        state_tokens=None,
        n_state=0,
        timestep=None,
    ):
        self.last_injected = {"n_action": int(n_action), "n_state": int(n_state)}
        pieces = []
        if n_action:
            pieces.append(action_tokens.to(state.hidden_states.dtype))
        if n_state:
            pieces.append(state_tokens.to(state.hidden_states.dtype))
        state.hidden_states = torch.cat([state.hidden_states, *pieces], dim=1)
        return state

    def extract_action_tokens(self, state, n_action):
        n_video = state.hidden_states.shape[1] - n_action
        action_tokens = state.hidden_states[:, n_video:, :]
        state.hidden_states = state.hidden_states[:, :n_video, :]
        return state, action_tokens

    def extract_shared_tokens(self, state, n_action, *, n_state=0):
        self.last_extracted = {"n_action": int(n_action), "n_state": int(n_state)}
        n_tail = n_action + n_state
        n_video = state.hidden_states.shape[1] - n_tail
        action_tokens = state.hidden_states[:, n_video : n_video + n_action, :]
        state.hidden_states = state.hidden_states[:, :n_video, :]
        return state, action_tokens

    def preprocess_input_for_train(self, *, frames=None, text=None, **kw):
        # ``first_frame_latents`` puts the test on the Wan TI2V path
        # (latent[0] = conditioning, loss + mask both skip it).
        return {
            "input_latents": torch.randn(1, 16, 3, 8, 8),
            "context": torch.randn(1, 4, self._dim),
            "context_mask": torch.ones(1, 4, dtype=torch.bool),
            "seq_lens": torch.ones(1, dtype=torch.long),
            "first_frame_latents": torch.zeros(1, 16, 1, 8, 8),
        }

    def get_submodule(self, name):
        return getattr(self, f"_{name}", None)

    def set_submodule(self, name, module):
        setattr(self, f"_{name}", module)

    def decode_video(self, latents, *, tiled=True):
        return []

    def set_dtype_device(self, dtype, device):
        pass


class _MockScheduler:
    """Minimal scheduler mock for loss tests."""

    num_train_timesteps = 1000

    def __init__(self, n=1000):
        self.timesteps = torch.linspace(1, 0, n)
        self.sigmas = torch.linspace(1, 0, n)
        self.linear_timesteps_weights = torch.ones(n)

    def set_timesteps(self, n, **kwargs):
        self.timesteps = torch.linspace(1, 0, n)
        self.sigmas = torch.linspace(1, 0, n)
        self.linear_timesteps_weights = torch.ones(n)

    def add_noise(self, original, noise, sigma):
        return (1 - sigma) * original + sigma * noise

    def training_target(self, original, noise):
        return noise - original

    def training_weight(self, timestep_ids):
        return self.linear_timesteps_weights[timestep_ids]

    def flow_step(self, pred, sigma, sigma_next, sample):
        return sample + pred * (sigma_next - sigma)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_TINY_ARCH_CFG = {
    "framework": "dual_system",
    "variant": "joint_cross_attn",
    "detach_bridge": False,
    "action_dim": 7,
    "dim": 64,
    "ffn_dim": 128,
    "num_heads": 2,
    "num_layers": 2,
    "video_dim": 64,
    "text_dim": 64,
    "bridge_layers": (0, 1),
}


def _make_tiny_arch():
    arch = DualSystemCrossAttnArchitecture(cfg=_TINY_ARCH_CFG)
    arch.video_backbone = _MockVideoBackbone(dim=64, num_layers=2)
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def _make_fake_loss_inputs(B=1, action_dim=7, T_action=5, video_dim=64):
    """Build the minimal dict that architecture.compute_loss expects."""
    C, T, H, W = 16, 3, 8, 8
    return {
        "input_latents": torch.randn(B, C, T, H, W),
        "latents": None,
        "height": H * 8,
        "width": W * 8,
        "num_frames": 9,
        "tiled": False,
        "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False,
        "max_timestep_boundary": 1.0,
        "min_timestep_boundary": 0.0,
    }


def test_resolve_architecture_config_merges_action_backbone_fields():
    from types import SimpleNamespace

    from openwam.model import resolve_architecture_config

    model_cfg = SimpleNamespace(
        architecture={
            "framework": "dual_system",
            "variant": "joint_cross_attn",
            "detach_bridge": False,
            "action_dim": 7,
        },
        action_backbone={"dim": 64, "ffn_dim": 128, "num_heads": 2},
    )
    resolved = resolve_architecture_config(model_cfg, video_dim=64, num_dit_layers=12)

    assert resolved.registry_name == "dual_system_cross_attn"
    assert resolved.params["dim"] == 64
    assert resolved.params["ffn_dim"] == 128
    assert resolved.params["num_heads"] == 2
    assert resolved.params["video_dim"] == 64


def _apply_freeze(pipe, trainer_attrs, freeze_list):
    """Replicate the freeze logic from OpenWAMTrainer.__init__."""
    for name in freeze_list:
        module = getattr(pipe, name, None)
        if module is None:
            module = trainer_attrs.get(name, None)
        if module is not None:
            module.requires_grad_(False)


def test_freeze_joint_strategy():
    """Joint strategy (dual_system.yaml): freeze text_encoder + vae; dit + action_dit remain trainable."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_dit = arch.action_backbone

    freeze_list = ["text_encoder", "vae"]  # from configs/model/dual_system.yaml
    _apply_freeze(pipe, {"action_dit": action_dit}, freeze_list)

    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert all(p.requires_grad for p in pipe.dit.parameters())
    assert all(p.requires_grad for p in action_dit.parameters())


def test_freeze_video_only_strategy():
    """Video-only freeze strategy: freeze text_encoder + vae + action_dit; dit stays trainable."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_dit = arch.action_backbone

    # Freeze the action backbone too → video-only training (no shipped yaml uses this;
    # exercises the freeze mechanism for a hypothetical lambda_action=0 setup).
    freeze_list = ["text_encoder", "vae", "action_dit"]
    _apply_freeze(pipe, {"action_dit": action_dit}, freeze_list)

    assert not any(p.requires_grad for p in pipe.text_encoder.parameters())
    assert not any(p.requires_grad for p in pipe.vae.parameters())
    assert not any(p.requires_grad for p in action_dit.parameters())
    assert all(p.requires_grad for p in pipe.dit.parameters())


def test_freeze_custom_list():
    """Custom freeze: dit frozen, action_dit stays trainable."""
    pipe = _MockPipeline()
    arch = _make_tiny_arch()
    action_dit = arch.action_backbone

    freeze_list = ["text_encoder", "vae", "dit"]
    _apply_freeze(pipe, {"action_dit": action_dit}, freeze_list)

    assert not any(p.requires_grad for p in pipe.dit.parameters())
    assert all(p.requires_grad for p in action_dit.parameters())


# ---------------------------------------------------------------------------
# Tests: Loss computation (single and multi-batch)
# ---------------------------------------------------------------------------


def test_single_batch_loss():
    """B=1: architecture.compute_loss produces valid scalar losses."""
    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _MockScheduler()

    B, T_action, action_dim = 1, 5, 7
    action_data = torch.randn(B, T_action, action_dim)
    inputs = _make_fake_loss_inputs(B=B, action_dim=action_dim, T_action=T_action)

    result = arch.compute_loss(
        **inputs,
        actions=action_data,
    )

    assert "loss" in result
    assert "loss_video" in result
    assert "loss_action" in result
    assert result["loss"].shape == ()
    assert result["loss"].item() > 0
    assert result["loss_video"].item() >= 0
    assert result["loss_action"].item() >= 0


def test_multi_batch_loss():
    """B=2: architecture.compute_loss produces valid scalar losses with batched input."""
    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _MockScheduler()

    B, T_action, action_dim = 2, 5, 7
    action_data = torch.randn(B, T_action, action_dim)
    inputs = _make_fake_loss_inputs(B=B, action_dim=action_dim, T_action=T_action)

    result = arch.compute_loss(
        **inputs,
        actions=action_data,
    )

    assert result["loss"].shape == ()
    assert result["loss"].item() > 0


def test_video_only_loss():
    """lambda_action=0: only video loss is computed."""
    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _MockScheduler()

    B = 1
    inputs = _make_fake_loss_inputs(B=B)

    result = arch.compute_loss(
        **inputs,
        actions=None,
        lambda_video=1.0,
        lambda_action=0.0,
    )

    assert result["loss"].item() > 0
    assert result["loss_action"].item() == 0.0


def test_loss_backward():
    """Loss should be differentiable and backward should succeed."""
    arch = _make_tiny_arch()
    arch.train()
    arch.action_backbone.scheduler = _MockScheduler()

    B, T_action, action_dim = 1, 5, 7
    action_data = torch.randn(B, T_action, action_dim)
    inputs = _make_fake_loss_inputs(B=B)

    result = arch.compute_loss(
        **inputs,
        actions=action_data,
    )

    result["loss"].backward()

    has_grad = any(p.grad is not None for p in arch.action_backbone.parameters())
    assert has_grad, "ActionDiT should have gradients after backward"


# ---------------------------------------------------------------------------
# Tests: Training loop with mock components
# ---------------------------------------------------------------------------


def test_training_step():
    """Run 3 optimizer steps: loss should decrease or remain stable."""
    arch = _make_tiny_arch()
    arch.train()
    arch.action_backbone.scheduler = _MockScheduler()

    optimizer = torch.optim.Adam(arch.parameters(), lr=1e-3)

    losses = []
    for step in range(3):
        B, T_action, action_dim = 1, 5, 7
        action_data = torch.randn(B, T_action, action_dim)
        inputs = _make_fake_loss_inputs(B=B)

        result = arch.compute_loss(
            **inputs,
            actions=action_data,
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
    """Save and load checkpoint via architecture; verify weights match."""
    arch = _make_tiny_arch()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "test_ckpt.safetensors")
        arch.save_checkpoint(ckpt_path)
        assert Path(ckpt_path).exists()

        # Load into fresh model
        arch2 = _make_tiny_arch()
        arch2.load_checkpoint(ckpt_path)

        # Verify action_dit weights match
        for (k1, v1), (k2, v2) in zip(
            arch.action_backbone.state_dict().items(),
            arch2.action_backbone.state_dict().items(),
        ):
            assert k1 == k2
            assert torch.equal(v1, v2), f"Weight mismatch for {k1}"


def test_save_checkpoint_excludes_vlm_backbone():
    """Architecture save_checkpoint must exclude vlm_backbone params (saved separately)."""
    from openwam.model.architectures.base import BaseWAMArchitecture

    class _VLMArch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self.action_head = nn.Linear(16, 8)
            self.vlm_backbone = nn.Linear(16, 32)

        def forward(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    arch = _VLMArch()
    with torch.no_grad():
        arch.action_head.weight.copy_(torch.randn_like(arch.action_head.weight))

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "vlm_ckpt.safetensors")
        arch.save_checkpoint(ckpt_path)

        from safetensors.torch import load_file

        saved = load_file(ckpt_path)
        assert not any(k.startswith("vlm_backbone.") for k in saved), "vlm_backbone params should be excluded"
        assert any(k.startswith("action_head.") for k in saved), "non-VLM params should be saved"

        reloaded = _VLMArch()
        reloaded.load_checkpoint(ckpt_path)
        assert torch.equal(reloaded.action_head.weight, arch.action_head.weight)


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
    from openwam.model.architectures.utils.common import (
        downsample_video_mask_to_latent as _downsample_video_mask_to_latent,
    )

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


def test_architecture_variant_strings_for_sampler_guard():
    """The guard keys on canonical.variant == 'joint_self_attn'; pin the variant
    string for every architecture so the guard cannot silently drift."""
    import openwam.model.architectures  # noqa: F401  (run every @register_architecture)
    from openwam.model.architectures.registry import normalize_architecture_spec

    assert normalize_architecture_spec("dual_system_self_attn").variant == "joint_self_attn"
    assert normalize_architecture_spec("tri_system_joint_self_attn").variant == "joint_self_attn"
    for name in (
        "dual_system_cross_attn",
        "dual_system_idm",
        "single_system_vanilla",
        "single_system_moe",
    ):
        assert normalize_architecture_spec(name).variant != "joint_self_attn"
