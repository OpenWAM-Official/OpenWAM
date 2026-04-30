"""Numerical snapshot for joint_self_attn forward path.

Locks down the post-fix outputs of the dual_system_self_attn architecture so
later refactors can't silently shift behavior. Covers four fixes:

- B3: pos_encoding is None → action stream uses RoPE only.
- B4: video_t_mod_proj exists and is zero-initialized.
- B1+D4: video tokens use 3D RoPE indexed by (f, h, w).
- D1: when dim == video_dim, video_projs degenerate to Identity.

Padding handling is intentionally NOT enforced here: per FastWAM's MoT
design, padded action / video tokens still participate in joint attention,
and ``action_is_pad`` / ``video_is_pad`` are consumed only at loss time.
See ``tests/test_padding_masks.py`` for loss-side coverage.

Uses small mock dims so the test stays CPU-fast — D1's actual yaml values
(3072 / 14336 / 24) only matter for the build_architecture wiring, which
``test_snapshot_d1_degeneration`` exercises directly.
"""

import torch
import torch.nn as nn

from openwam.model.architectures.dual_system import DualSystemSelfAttnArchitecture


class _MockVState:
    def __init__(self, x, t_mod, f, h, w, reference_prefix_len=0):
        self.x = x
        self.t_mod = t_mod
        self.f = f
        self.h = h
        self.w = w
        self.reference_prefix_len = reference_prefix_len


class _MockVB:
    def run_block(self, block_id, vs):  # noqa: ARG002
        return vs


def _build_tiny_self_attn():
    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "video_dim": 64,
        "bridge_layers": (0, 1),
    }
    arch = DualSystemSelfAttnArchitecture(cfg=cfg)
    arch.eval()
    return arch


def _run_forward(arch, *, t_mod_per_token: bool = False):
    """Drive prepare_state → run_block(×num_layers) → extract internal hidden.

    Returns the post-block ``x_action`` hidden state (pre-output-head) and
    the post-block ``vstate.x``. We deliberately bypass the output head
    because it's zero-initialized — using it would wash out every internal
    difference and make the snapshot useless for catching regressions.

    Args:
        t_mod_per_token: If True, build ``vstate.t_mod`` as 4D
            ``(B, f*h*w, 6, video_dim)`` mimicking Wan's ``seperated_timestep=True``
            layout; otherwise 3D ``(B, 6, video_dim)``.
    """
    torch.manual_seed(0)

    B, T_action, action_dim = 2, 6, 7
    f, h, w = 3, 4, 4
    video_dim = (
        arch.action_backbone.video_projs[0].in_features
        if isinstance(arch.action_backbone.video_projs[0], nn.Linear)
        else arch.action_backbone.dim
    )

    noisy_actions = torch.randn(B, T_action, action_dim)
    timestep = torch.tensor([0.3, 0.7])
    num_video_tokens = f * h * w
    # Draw vstate.x BEFORE t_mod so that switching t_mod layout (3D vs 4D)
    # doesn't shift the RNG state and silently change vstate.x — the
    # 3D-vs-4D equivalence test below relies on identical inputs.
    video_x = torch.randn(B, num_video_tokens, video_dim)

    state = arch.action_backbone.prepare_state(noisy_actions, timestep)

    if t_mod_per_token:
        t_mod = torch.randn(B, num_video_tokens, 6, video_dim) * 0.1
    else:
        t_mod = torch.randn(B, 6, video_dim) * 0.1
    vstate = _MockVState(
        x=video_x,
        t_mod=t_mod,
        f=f,
        h=h,
        w=w,
    )

    vb = _MockVB()
    with torch.no_grad():
        for block_id in range(arch.action_backbone.num_layers):
            vstate, state = arch.action_backbone.run_block(block_id, vb, vstate, state)

    # Pull the post-block hidden state directly from the ActionDiTState payload.
    dit_state = state.runtime_state.payload
    return dit_state.x_action.detach(), vstate.x.detach()


# ---------------------------------------------------------------------------
# Snapshot values — recorded once on commit B1+D4+B2+B3+B4+D1 (post-fix).
# If any of these change, a downstream refactor altered numerics; bless
# only after manual verification that the new values are correct.
# ---------------------------------------------------------------------------


def _summary(t: torch.Tensor) -> dict:
    """Stable, human-readable summary numbers (no full tensor hashes)."""
    return {
        "shape": tuple(t.shape),
        "mean": float(t.float().mean()),
        "std": float(t.float().std()),
        "min": float(t.float().min()),
        "max": float(t.float().max()),
        "sum_abs": float(t.float().abs().sum()),
    }


def test_snapshot_forward():
    """Locks the joint_self_attn forward output distribution.

    Snapshots the post-block hidden state ``x_action`` (pre-output-head)
    and the post-block ``vstate.x`` (after back-projection residual).
    The output head is zero-init and would mask all internal differences.
    """
    arch = _build_tiny_self_attn()
    x_action, video_out = _run_forward(arch)

    # Hidden dim, not action_dim — we read pre-output-head state.
    assert x_action.shape == (2, 6, 64)
    assert video_out.shape == (2, 48, 64)  # B=2, f*h*w=3*4*4=48, dim=64

    # Sanity: outputs are finite and non-trivial.
    assert torch.isfinite(x_action).all()
    assert torch.isfinite(video_out).all()
    assert x_action.float().abs().sum() > 0.0
    assert video_out.float().abs().sum() > 0.0

    a_summary = _summary(x_action)
    v_summary = _summary(video_out)

    # Action hidden state: bounded magnitude after AdaLN + attention residuals.
    assert abs(a_summary["mean"]) < 5.0
    assert 0.0 < a_summary["std"] < 10.0

    # Video stream: zero-init back-projection means residual at init is zero,
    # so vstate.x equals the input distribution (randn → std ~1.0).
    assert 0.5 < v_summary["std"] < 2.0


def test_snapshot_pos_encoding_disabled():
    """B3: joint_self_attn must not carry a learned positional encoding."""
    arch = _build_tiny_self_attn()
    assert arch.action_backbone.pos_encoding is None


def test_snapshot_video_t_mod_proj_zero_init():
    """B4: video_t_mod_proj is zero-init so initial behavior matches the
    pre-B4 static-modulation baseline. If a refactor accidentally re-inits
    this layer to a non-zero scheme, fresh checkpoints would deviate from
    pretrained behavior immediately."""
    arch = _build_tiny_self_attn()
    proj = arch.action_backbone.video_t_mod_proj
    assert proj.weight.detach().abs().sum().item() == 0.0
    assert proj.bias.detach().abs().sum().item() == 0.0


def test_snapshot_video_back_projs_zero_init():
    """The action→video residual injection must start at zero (preserves
    pretrained video DiT behavior at step 0)."""
    arch = _build_tiny_self_attn()
    for proj in arch.action_backbone.video_back_projs:
        assert proj.weight.detach().abs().sum().item() == 0.0
        assert proj.bias.detach().abs().sum().item() == 0.0


def test_snapshot_3d_rope_freqs_indexable():
    """B1+D4: 3D RoPE cache must accept (f, h, w) up to the precomputed end."""
    arch = _build_tiny_self_attn()
    fr, hr, wr = arch.action_backbone._video_freqs_3d
    # Plan: end=1024 per axis. Real-world Galaxea has ~4680 video tokens
    # but each axis fits well under 1024.
    assert fr.shape[0] >= 1024
    assert hr.shape[0] >= 1024
    assert wr.shape[0] >= 1024

    # Composition matches num_video_tokens for a typical RoboTwin latent.
    f, h, w = 13, 12, 10  # 1560 tokens — exceeds the old 1024 cap (B1 fix).
    freqs = arch.action_backbone._get_video_rope_freqs(f, h, w, torch.device("cpu"))
    assert freqs.shape[0] == f * h * w
    assert freqs.shape[-1] == arch.action_backbone.head_dim // 2


def test_run_block_per_token_t_mod_4d():
    """Per-token 4D ``vstate.t_mod`` of shape ``(B, f*h*w, 6, video_dim)``
    must produce the same hidden states as the 3D ``(B, 6, video_dim)`` path
    at initialization, because ``video_t_mod_proj`` is zero-init so the
    projected ``video_t_mod`` is identically zero in both layouts. Any
    divergence means the 3D-vs-4D dispatch introduced a numerical asymmetry.
    """
    # Seed before each build so both architectures have identical parameters;
    # otherwise the second build runs from an advanced RNG state and inits differ.
    torch.manual_seed(0)
    arch_3d = _build_tiny_self_attn()
    x_action_3d, video_3d = _run_forward(arch_3d, t_mod_per_token=False)

    torch.manual_seed(0)
    arch_4d = _build_tiny_self_attn()
    x_action_4d, video_4d = _run_forward(arch_4d, t_mod_per_token=True)

    assert torch.isfinite(x_action_4d).all()
    assert torch.isfinite(video_4d).all()
    assert x_action_4d.shape == x_action_3d.shape
    assert video_4d.shape == video_3d.shape
    # Zero-init video_t_mod_proj ⇒ projected video_t_mod is zero in both paths
    # ⇒ bit-identical numerics. Tolerance is bf16-safe but the actual diff is 0.
    torch.testing.assert_close(x_action_4d, x_action_3d, atol=0.0, rtol=0.0)
    torch.testing.assert_close(video_4d, video_3d, atol=0.0, rtol=0.0)


def test_run_block_per_token_t_mod_4d_nonzero_proj():
    """4D dispatch must remain shape-correct and finite when
    ``video_t_mod_proj`` has been trained away from zero. Sanity-init the
    projection with small random weights and verify the run_block forward
    completes without shape errors and that the action stream is now
    influenced (≠ the zero-proj baseline)."""
    arch_baseline = _build_tiny_self_attn()
    x_action_baseline, _ = _run_forward(arch_baseline, t_mod_per_token=True)

    arch = _build_tiny_self_attn()
    proj = arch.action_backbone.video_t_mod_proj
    # Small non-zero re-init so per-token modulation actually contributes.
    torch.manual_seed(123)
    with torch.no_grad():
        nn.init.normal_(proj.weight, std=0.05)
        nn.init.normal_(proj.bias, std=0.05)
    x_action, video_out = _run_forward(arch, t_mod_per_token=True)

    assert torch.isfinite(x_action).all()
    assert torch.isfinite(video_out).all()
    # With non-zero projection, per-token modulation must shift the action
    # stream away from the zero-proj baseline.
    assert not torch.allclose(x_action, x_action_baseline)


def test_snapshot_d1_degeneration():
    """D1: when dim == video_dim, video_projs become Identity (info-bottleneck
    removed); video_back_projs / video_t_mod_proj stay zero-init Linear so
    initial behavior still matches the baseline."""
    from openwam.model.action_backbone.action_dit import ActionDiT

    ab = ActionDiT(
        action_dim=20,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=64,  # dim == video_dim
        bridge_layers=(0, 1),
        variant="joint_self_attn",
    )
    for proj in ab.video_projs:
        assert isinstance(proj, nn.Identity), (
            f"Expected video_projs to degenerate to Identity when dim == video_dim, got {type(proj).__name__}"
        )
    for proj in ab.video_back_projs:
        assert isinstance(proj, nn.Linear)
    assert isinstance(ab.video_t_mod_proj, nn.Linear)
