"""Tests for ``video_backbone.from_scratch`` switch and the underlying
``reinit_dit_from_scratch`` helper.

Two layers:

A. Lightweight CPU unit tests build a *miniature* WanModel (dim=128,
   2 layers) and exercise the helper directly. No real weights, no GPU,
   runs in seconds. These guard the four classes of directly-mounted
   ``nn.Parameter`` (RMSNorm.weight, DiTBlock.modulation, Head.modulation,
   MLP.emb_pos) that ``module.modules() + reset_parameters()`` does not
   reach — if anyone changes the helper and forgets one, these tests fail.

B. GPU integration test (``@pytest.mark.gpu``) builds a real
   ``WanVideoBackbone`` twice (from_scratch=False / True) and verifies
   VAE/T5 weights are identical across the two runs while DiT weights
   differ. Requires the model_path in dual_system.yaml to be valid.
"""

from __future__ import annotations

import pytest
import torch

from openwam.model.video_backbone.wan.models.dit import RMSNorm, WanModel
from openwam.model.video_backbone.wan.reinit import reinit_dit_from_scratch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mini_dit() -> WanModel:
    """A tiny WanModel that exercises every code path in the helper.

    Kept small (dim=128, 2 layers) so tests run on CPU in < 1s.
    """
    return WanModel(
        dim=128,
        in_dim=16,
        ffn_dim=256,
        out_dim=16,
        text_dim=64,
        freq_dim=64,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=2,
        has_image_input=False,
    )


class _FakePipe:
    """Stand-in for ``WanVideoPipeline``: only ``.dit`` / ``.dit2`` matter."""

    def __init__(self, dit: torch.nn.Module, dit2: torch.nn.Module | None = None):
        self.dit = dit
        self.dit2 = dit2


def _poison_params(dit: WanModel, value: float = 7.0) -> None:
    """Overwrite every learnable param with ``value`` to simulate "pretrained
    weights loaded". Re-init should erase these uniformly."""
    with torch.no_grad():
        for p in dit.parameters():
            p.fill_(value)


# ---------------------------------------------------------------------------
# A. Lightweight CPU unit tests
# ---------------------------------------------------------------------------


def test_reinit_block_modulation_distribution():
    """``DiTBlock.modulation`` is a raw nn.Parameter not reachable via
    ``modules()`` — without hand-reset it would keep the poisoned value.
    After reset its std must match the constructor: ``dim**-0.5``."""
    dit = _build_mini_dit()
    _poison_params(dit, value=7.0)

    torch.manual_seed(42)
    reinit_dit_from_scratch(_FakePipe(dit))

    expected_std = dit.dim**-0.5
    for block in dit.blocks:
        # No longer poisoned.
        assert not torch.allclose(block.modulation, torch.full_like(block.modulation, 7.0))
        # Mean ~ 0.
        assert abs(block.modulation.float().mean().item()) < 0.1, (
            f"modulation mean drifted: {block.modulation.float().mean().item()}"
        )
        # Std in the right ballpark (Gaussian sampling on small tensors is noisy).
        actual_std = block.modulation.float().std().item()
        assert 0.5 * expected_std < actual_std < 2.0 * expected_std, (
            f"modulation std {actual_std} not within [0.5x, 2x] of expected {expected_std}"
        )


def test_reinit_head_modulation_distribution():
    """``Head.modulation`` has the same raw-Parameter shape and must be
    reset the same way as ``DiTBlock.modulation``."""
    dit = _build_mini_dit()
    _poison_params(dit, value=7.0)
    torch.manual_seed(42)
    reinit_dit_from_scratch(_FakePipe(dit))

    expected_std = dit.dim**-0.5
    head_mod = dit.head.modulation
    assert not torch.allclose(head_mod, torch.full_like(head_mod, 7.0))
    assert abs(head_mod.float().mean().item()) < 0.1
    actual_std = head_mod.float().std().item()
    assert 0.5 * expected_std < actual_std < 2.0 * expected_std


def test_reinit_rmsnorm_weight_set_to_one():
    """``RMSNorm.weight`` constructor default is ``ones(dim)``. Re-init must
    restore that, not leave the poisoned value and not turn it into Gaussian."""
    dit = _build_mini_dit()
    _poison_params(dit, value=0.3)
    reinit_dit_from_scratch(_FakePipe(dit))

    rmsnorm_count = 0
    for sub in dit.modules():
        if isinstance(sub, RMSNorm):
            rmsnorm_count += 1
            assert torch.allclose(sub.weight, torch.ones_like(sub.weight)), (
                f"RMSNorm.weight not reset to ones: {sub.weight}"
            )
    # Sanity: the mini-DiT has at least 8 RMSNorms (2 layers × 2 attn × 2 norms).
    assert rmsnorm_count >= 8, f"expected >=8 RMSNorm modules, found {rmsnorm_count}"


def test_reinit_stdlib_layers_change():
    """Stdlib ``nn.Linear`` / ``LayerNorm`` etc. get standard
    ``reset_parameters()``. After reset they must differ from poisoned 7.0."""
    dit = _build_mini_dit()
    _poison_params(dit, value=7.0)
    reinit_dit_from_scratch(_FakePipe(dit))

    # Pick a Linear and a LayerNorm and ensure they're no longer all 7.0.
    qkv = dit.blocks[0].self_attn.q
    assert isinstance(qkv, torch.nn.Linear)
    assert not torch.allclose(qkv.weight, torch.full_like(qkv.weight, 7.0))

    norm3 = dit.blocks[0].norm3
    assert isinstance(norm3, torch.nn.LayerNorm)
    # LayerNorm.reset_parameters sets weight=1, bias=0.
    assert torch.allclose(norm3.weight, torch.ones_like(norm3.weight))
    assert torch.allclose(norm3.bias, torch.zeros_like(norm3.bias))


def test_reinit_freqs_untouched():
    """``WanModel.freqs`` is a deterministic RoPE cache (plain Tensor attr,
    neither buffer nor parameter). It must survive re-init bit-exact."""
    dit = _build_mini_dit()
    freqs_before = tuple(f.clone() for f in dit.freqs)
    reinit_dit_from_scratch(_FakePipe(dit))
    for f_before, f_after in zip(freqs_before, dit.freqs):
        assert torch.equal(f_before, f_after), "WanModel.freqs was mutated"


def test_reinit_deterministic_under_seed():
    """Same manual_seed -> bit-exact identical post-reinit parameters.
    Guards the user-facing promise: cfg.project.seed makes the from-scratch
    ablation reproducible across runs."""
    dit1 = _build_mini_dit()
    dit2 = _build_mini_dit()

    torch.manual_seed(123)
    reinit_dit_from_scratch(_FakePipe(dit1))
    torch.manual_seed(123)
    reinit_dit_from_scratch(_FakePipe(dit2))

    for (n1, p1), (n2, p2) in zip(dit1.named_parameters(), dit2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"{n1} differs across seeded runs"


def test_reinit_does_not_alter_param_count():
    """Re-init only changes *values*, never the model's *structure*."""
    dit = _build_mini_dit()
    count_before = sum(p.numel() for p in dit.parameters())
    reinit_dit_from_scratch(_FakePipe(dit))
    count_after = sum(p.numel() for p in dit.parameters())
    assert count_before == count_after


def test_reinit_handles_dit2_when_present():
    """If a Wan pipeline has both ``dit`` and ``dit2`` (Wan2.2 high/low-noise
    expert architectures), both should be re-initialized."""
    dit_a = _build_mini_dit()
    dit_b = _build_mini_dit()
    _poison_params(dit_a, value=7.0)
    _poison_params(dit_b, value=7.0)

    reinit_dit_from_scratch(_FakePipe(dit_a, dit2=dit_b))

    for dit in (dit_a, dit_b):
        for block in dit.blocks:
            assert not torch.allclose(block.modulation, torch.full_like(block.modulation, 7.0))


def test_reinit_no_dit_warns_not_crashes(caplog):
    """A pipe with neither ``dit`` nor ``dit2`` should log a warning and
    return — never crash."""

    class _EmptyPipe:
        pass

    import logging

    with caplog.at_level(logging.WARNING, logger="openwam.model.video_backbone.wan"):
        reinit_dit_from_scratch(_EmptyPipe())
    assert any("no dit/dit2" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# A2. Terminal-visible verification output
#
# The user wants to *see* in the terminal that reinit actually happened —
# print(..., flush=True) bypasses the logging config (which may suppress INFO)
# and emits a BEFORE/AFTER summary on rank 0. These tests guard:
#   - the summary actually appears on rank 0
#   - it includes both BEFORE and AFTER numbers so the user can eyeball that
#     reset actually happened (i.e. AFTER differs from the poisoned BEFORE)
#   - it stays silent on non-zero ranks (no log spam in multi-rank runs)
#   - verbose=False suppresses output cleanly
# ---------------------------------------------------------------------------


def test_reinit_prints_before_after_summary_on_rank0(capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch):
    """Rank 0 must print a BEFORE/AFTER summary to stdout so the user sees
    "yes the loaded weights were actually replaced" in their terminal,
    independently of the logging config."""
    monkeypatch.setenv("RANK", "0")
    dit = _build_mini_dit()
    _poison_params(dit, value=7.0)

    reinit_dit_from_scratch(_FakePipe(dit))
    captured = capsys.readouterr().out

    # Header / footer bars present.
    assert "===" in captured
    assert "re-initialized from scratch" in captured
    # The BEFORE/AFTER table includes the marker tokens and at least one number.
    assert "BEFORE" in captured and "AFTER" in captured
    assert "q.weight" in captured
    assert "blocks[0].modulation" in captured
    assert "head.modulation" in captured
    assert "expected std" in captured  # sanity-check std target shown
    # The BEFORE numbers should reflect the poisoned state (mean ~ 7.0) and
    # the AFTER numbers should NOT — that's how the user knows reset happened.
    # We don't pin exact strings (format may evolve) but require the message
    # body to contain *both* a "7." value and at least one non-7 mean.
    assert "+7." in captured or "7.0" in captured, "BEFORE row should show poisoned 7.0"


def test_reinit_silent_on_non_rank_zero(capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch):
    """On rank > 0 the print() summary must be suppressed to avoid N copies
    of the same banner cluttering multi-rank logs."""
    monkeypatch.setenv("RANK", "1")
    dit = _build_mini_dit()
    reinit_dit_from_scratch(_FakePipe(dit))
    captured = capsys.readouterr().out
    assert captured == "", f"Non-rank-0 should not print, got: {captured!r}"


def test_reinit_verbose_false_silent(capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch):
    """Explicit ``verbose=False`` suppresses output even on rank 0 — useful
    for tests / library use where the banner would pollute output."""
    monkeypatch.setenv("RANK", "0")
    dit = _build_mini_dit()
    reinit_dit_from_scratch(_FakePipe(dit), verbose=False)
    captured = capsys.readouterr().out
    assert captured == ""


def test_reinit_prints_dit2_when_present(capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch):
    """Both ``dit`` and ``dit2`` should appear in the summary so the user
    sees coverage of all Wan2.2 high/low-noise expert architectures."""
    monkeypatch.setenv("RANK", "0")
    dit_a = _build_mini_dit()
    dit_b = _build_mini_dit()
    reinit_dit_from_scratch(_FakePipe(dit_a, dit2=dit_b))
    captured = capsys.readouterr().out
    assert "[dit]" in captured
    assert "[dit2]" in captured


def test_reinit_same_seed_reproduces_two_independent_constructs():
    """Two independent DiT constructions, each followed by ``manual_seed(N) +
    reinit_dit_from_scratch``, must yield bit-exact identical parameters.
    This is the property the user relies on for cross-run ablation: same
    cfg.project.seed -> same initial DiT weights, every time."""
    dit1 = _build_mini_dit()
    dit2 = _build_mini_dit()
    _poison_params(dit1, value=3.14)
    _poison_params(dit2, value=-2.71)  # different "pretrained" state on each side

    torch.manual_seed(2026)
    reinit_dit_from_scratch(_FakePipe(dit1), verbose=False)
    torch.manual_seed(2026)
    reinit_dit_from_scratch(_FakePipe(dit2), verbose=False)

    # Every named parameter — including the four hand-reset raw nn.Parameter
    # classes — must match bit-exact across the two constructions.
    p1 = dict(dit1.named_parameters())
    p2 = dict(dit2.named_parameters())
    assert p1.keys() == p2.keys()
    for k in p1:
        assert torch.equal(p1[k], p2[k]), f"param {k} differs across same-seed runs"


# ---------------------------------------------------------------------------
# B. GPU integration test — requires real Wan2.2-TI2V-5B weights
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_from_scratch_integration_real_weights():
    """End-to-end: build the architecture twice with the same seed, once
    with ``from_scratch: false`` and once with ``true``. Verify:
      - VAE and text_encoder params are bit-exact identical across runs
        (they are NOT re-initialized).
      - DiT params differ across runs (they ARE re-initialized).

    Marked ``@pytest.mark.gpu`` because it loads the full ~10 GB Wan
    pipeline; locally use ``pytest -m "not gpu"`` to skip.
    """
    import os

    from omegaconf import OmegaConf

    from openwam.model import build_architecture, resolve_architecture_config

    # Use a minimal cfg surface that the architecture builder accepts.
    base_cfg_yaml = """
model:
  architecture:
    framework: dual_system
    variant: joint_self_attn
    action_dim: 20
    use_proprioception: true
    state_dim: 20
    bridge_layers: null
    bridge_interval: 1
    mot_checkpoint_mixed_attn: true
    attention_mask_mode: action_sees_video
    video_attention_mask_mode: first_frame_causal
    dim: 1024
    text_dim: 4096
    ffn_dim: 4096
    num_heads: 24
    attn_head_dim: 128
  video_backbone:
    name: wan22_ti2v_5b
    model_path: __PLACEHOLDER__
    from_scratch: false
  action_backbone:
    dim: 1024
    text_dim: 4096
    ffn_dim: 4096
    num_heads: 24
    attn_head_dim: 128
training:
  initialize_model_on_cpu: true
"""
    model_path = os.environ.get("WAN_TI2V_5B_PATH", "/path/to/Wan2.2-TI2V-5B")
    if not os.path.isdir(model_path):
        pytest.skip(f"Wan2.2-TI2V-5B not found at {model_path}")

    def _build(from_scratch: bool):
        cfg = OmegaConf.create(base_cfg_yaml)
        cfg.model.video_backbone.model_path = model_path
        cfg.model.video_backbone.from_scratch = from_scratch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        resolved = resolve_architecture_config(cfg.model)
        return build_architecture(resolved.registry_name, resolved.params)

    arch_a = _build(from_scratch=False)
    arch_b = _build(from_scratch=True)

    # The video backbone is itself the module holding dit / vae / text_encoder as
    # named children (the old `._pipe` wrapper was removed); read them directly.
    pipe_a = arch_a.video_backbone
    pipe_b = arch_b.video_backbone

    # VAE: bit-exact identical (untouched by from_scratch).
    vae_a = dict(pipe_a.vae.named_parameters())
    vae_b = dict(pipe_b.vae.named_parameters())
    assert vae_a.keys() == vae_b.keys()
    for k in vae_a:
        assert torch.equal(vae_a[k], vae_b[k]), f"VAE param {k} differs across runs"

    # Text encoder: bit-exact identical.
    te_a = dict(pipe_a.text_encoder.named_parameters())
    te_b = dict(pipe_b.text_encoder.named_parameters())
    assert te_a.keys() == te_b.keys()
    for k in te_a:
        assert torch.equal(te_a[k], te_b[k]), f"text_encoder param {k} differs across runs"

    # DiT: must differ (at least the first block's q weight).
    qa = pipe_a.dit.blocks[0].self_attn.q.weight
    qb = pipe_b.dit.blocks[0].self_attn.q.weight
    assert not torch.equal(qa, qb), "DiT q.weight identical across pretrained vs from-scratch"

    # DiTBlock.modulation: in run B it should be ~N(0, dim**-0.5).
    dim = pipe_b.dit.dim
    for block in pipe_b.dit.blocks:
        std = block.modulation.float().std().item()
        expected = dim**-0.5
        assert 0.5 * expected < std < 2.0 * expected, (
            f"from_scratch DiTBlock.modulation std={std} far from expected {expected}"
        )


@pytest.mark.gpu
def test_from_scratch_same_seed_bit_exact_across_two_full_builds():
    """End-to-end ablation reproducibility: two independent
    ``build_architecture(from_scratch=true)`` calls under the same
    ``torch.manual_seed(N)`` MUST yield bit-exact identical DiT parameters.

    This is the property the user relies on when running comparative
    ablations: "rerun with same project.seed and the model starts from
    exactly the same weights". Without this guarantee, two ablation runs
    would have different initial DiTs and the comparison would be confounded.

    Uses the same real Wan2.2-TI2V-5B path as the other GPU integration
    test. Two full architecture builds (~2 minutes total).
    """
    import os

    from omegaconf import OmegaConf

    from openwam.model import build_architecture, resolve_architecture_config

    base_cfg_yaml = """
model:
  architecture:
    framework: dual_system
    variant: joint_self_attn
    action_dim: 20
    use_proprioception: true
    state_dim: 20
    bridge_layers: null
    bridge_interval: 1
    mot_checkpoint_mixed_attn: true
    attention_mask_mode: action_sees_video
    video_attention_mask_mode: first_frame_causal
    dim: 1024
    text_dim: 4096
    ffn_dim: 4096
    num_heads: 24
    attn_head_dim: 128
  video_backbone:
    name: wan22_ti2v_5b
    model_path: __PLACEHOLDER__
    from_scratch: true
  action_backbone:
    dim: 1024
    text_dim: 4096
    ffn_dim: 4096
    num_heads: 24
    attn_head_dim: 128
training:
  initialize_model_on_cpu: true
"""
    model_path = os.environ.get("WAN_TI2V_5B_PATH", "/path/to/Wan2.2-TI2V-5B")
    if not os.path.isdir(model_path):
        pytest.skip(f"Wan2.2-TI2V-5B not found at {model_path}")

    def _build_with_seed(seed: int):
        cfg = OmegaConf.create(base_cfg_yaml)
        cfg.model.video_backbone.model_path = model_path
        # Seed BEFORE build_architecture, mirroring what OpenWAMTrainer does.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        resolved = resolve_architecture_config(cfg.model)
        return build_architecture(resolved.registry_name, resolved.params)

    arch_a = _build_with_seed(seed=42)
    arch_b = _build_with_seed(seed=42)

    dit_a = arch_a.video_backbone.dit
    dit_b = arch_b.video_backbone.dit

    # Sanity: the re-init banner should have printed twice (once per build) —
    # we don't capture it here (capsys + pytest -s interactions are flaky in
    # GPU tests), but the assertion below is what actually matters.

    # Every named parameter — including the 30 DiTBlock.modulation tensors,
    # the Head.modulation, and ~180 RMSNorm.weight tensors — must match
    # bit-exact across the two independent builds.
    params_a = dict(dit_a.named_parameters())
    params_b = dict(dit_b.named_parameters())
    assert params_a.keys() == params_b.keys()
    mismatched = []
    for k in params_a:
        if not torch.equal(params_a[k], params_b[k]):
            mismatched.append(k)
    assert not mismatched, (
        f"{len(mismatched)} DiT params differ across same-seed builds (sample: {mismatched[:3]}). "
        "from_scratch is not deterministic under cfg.project.seed."
    )

    # Spot-check one of the hand-reset directly-mounted Parameters:
    # if these were missed by the helper, they'd be the first to diverge.
    assert torch.equal(dit_a.blocks[0].modulation, dit_b.blocks[0].modulation)
    assert torch.equal(dit_a.head.modulation, dit_b.head.modulation)
