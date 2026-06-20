"""End-to-end routing of per-sample proprio_mask through the model layer.

Asserts that a mixed [EgoDex-style mask=False, AgiBot-style mask=True] batch
gets a per-row proprio_mask collected by ``prepare_inputs`` and consumed by
``_append_proprio_context_token`` via ``pipeline_inputs['_proprio_sample_mask']``.

We avoid spinning up a real video backbone by stubbing ``preprocess`` and the
sample-pipeline transform, and by driving the proprio path directly via
``_append_proprio_context_token``. Backbones aren't needed for the per-sample
gating contract: prepare_inputs collects, compute_loss routes, the helper
consumes. Mid-pipeline behavior (video backbone forward) is out of scope.
"""

from __future__ import annotations

import torch

from openwam.model.architectures.dual_system.joint_self_attn import (
    DualSystemSelfAttnArchitecture,
)


class _IdentityTransform:
    def apply(self, sample):
        return sample


class _StubArch(DualSystemSelfAttnArchitecture):
    """Architecture stub: enough state for prepare_inputs + proprio gating."""

    def __init__(self, *, state_dim: int = 20, text_dim: int = 16):
        super().__init__(None)
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._init_proprio_context(
            {
                "use_proprioception": True,
                "state_dim": state_dim,
                "text_dim": text_dim,
            },
            text_dim=text_dim,
        )
        self._use_gradient_checkpointing = False
        self._use_gradient_checkpointing_offload = False
        self._max_timestep_boundary = 1.0
        self._min_timestep_boundary = 0.0
        self._pipeline_transform_instance = _IdentityTransform()

    def preprocess(self, **kwargs) -> dict:
        return {}


def _egodex_like_sample(state_dim: int = 20):
    """Mirror the EgoDexDataset output contract for proprio fields."""
    return {
        "video": [],
        "prompt": "",
        "action": torch.zeros(2, state_dim, dtype=torch.float32),
        "action_mask": None,
        "video_mask": None,
        "proprio": torch.zeros(1, state_dim, dtype=torch.float32),
        "proprio_mask": torch.zeros(1, dtype=torch.bool),
    }


def _agibot_like_sample(state_dim: int = 20):
    """Mirror the canonical proprio fields contract (real-valued proprio + mask)."""
    torch.manual_seed(7)
    return {
        "video": [],
        "prompt": "",
        "action": torch.randn(2, state_dim),
        "action_mask": None,
        "video_mask": None,
        "proprio": torch.randn(1, state_dim),
        "proprio_mask": torch.ones(1, dtype=torch.bool),
    }


def _run_through_proprio_helper(arch: _StubArch, inputs: dict, text_len: int = 4):
    """Helper: synthesize a context tensor + drive _append_proprio_context_token."""
    B = inputs["proprio"].shape[0]
    pipeline_inputs = {
        "context": torch.randn(B, text_len, arch.context_dim),
        "seq_lens": torch.tensor([text_len] * B),
        "_proprio_sample_mask": inputs["proprio_mask"],
    }
    return arch._append_proprio_context_token(pipeline_inputs, inputs["proprio"])


# ---------------------------------------------------------------------------
# 1) End-to-end: prepare_inputs -> _append_proprio_context_token
# ---------------------------------------------------------------------------


def test_egodex_agibot_mixed_batch_end_to_end():
    """Mixed [EgoDex, AgiBot] batch flows through prepare_inputs without error."""
    arch = _StubArch(state_dim=20, text_dim=16)
    batch = [_egodex_like_sample(), _agibot_like_sample()]
    inputs = arch.prepare_inputs(batch)

    # prepare_inputs squeezes (1, D) → (D,) per sample, so stacked → (B, D)
    assert inputs["proprio"].shape == (2, 20)
    assert inputs["proprio_mask"].shape == (2, 1)
    assert inputs["proprio_mask"].squeeze(-1).tolist() == [False, True]

    out = _run_through_proprio_helper(arch, inputs)
    assert out["context"].shape == (2, 5, 16)
    assert out["context_mask"].shape == (2, 5)


def test_mixed_batch_context_mask_last_two_match():
    """Last column of context_mask reflects per-sample proprio_mask."""
    arch = _StubArch(state_dim=20, text_dim=16)
    batch = [_egodex_like_sample(), _agibot_like_sample()]
    inputs = arch.prepare_inputs(batch)
    out = _run_through_proprio_helper(arch, inputs)

    assert out["context_mask"][:, -1].tolist() == [False, True]
    # EgoDex row: appended proprio token must be all-zero
    assert out["context"][0, -1, :].abs().sum().item() == 0.0
    # AgiBot row: appended proprio token must be non-zero
    assert out["context"][1, -1, :].abs().sum().item() > 0


# ---------------------------------------------------------------------------
# 2) Gradient isolation across the full prepare_inputs path
# ---------------------------------------------------------------------------


def test_mixed_batch_egodex_no_proprio_encoder_grad():
    """proprio_encoder.weight grad in mixed batch == grad from AgiBot-only batch."""
    # Mixed batch (EgoDex masked, AgiBot live)
    arch_mixed = _StubArch(state_dim=20, text_dim=16)
    arch_mixed.proprio_encoder.zero_grad(set_to_none=True)
    inputs_mixed = arch_mixed.prepare_inputs([_egodex_like_sample(), _agibot_like_sample()])
    torch.manual_seed(0)
    pipeline_inputs_mixed = {
        "context": torch.randn(2, 4, arch_mixed.context_dim),
        "seq_lens": torch.tensor([4, 4]),
        "_proprio_sample_mask": inputs_mixed["proprio_mask"],
    }
    out_mixed = arch_mixed._append_proprio_context_token(pipeline_inputs_mixed, inputs_mixed["proprio"])
    out_mixed["context"].sum().backward()
    g_mixed = arch_mixed.proprio_encoder.weight.grad.clone()

    # AgiBot-only reference, identical encoder init
    arch_solo = _StubArch(state_dim=20, text_dim=16)
    arch_solo.proprio_encoder.load_state_dict(arch_mixed.proprio_encoder.state_dict())
    arch_solo.proprio_encoder.zero_grad(set_to_none=True)
    inputs_solo = arch_solo.prepare_inputs([_agibot_like_sample()])
    torch.manual_seed(0)
    ctx_full = torch.randn(2, 4, arch_solo.context_dim)
    pipeline_inputs_solo = {
        "context": ctx_full[1:2].detach().clone(),
        "seq_lens": torch.tensor([4]),
        "_proprio_sample_mask": inputs_solo["proprio_mask"],
    }
    out_solo = arch_solo._append_proprio_context_token(pipeline_inputs_solo, inputs_solo["proprio"])
    out_solo["context"].sum().backward()
    g_solo = arch_solo.proprio_encoder.weight.grad.clone()

    assert torch.allclose(g_mixed, g_solo, atol=1e-6, rtol=1e-5)
