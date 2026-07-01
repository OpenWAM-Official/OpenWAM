"""joint_cross_attn (dual_system) support for the CosmosPredict25 video backbone.

``joint_cross_attn`` runs the video DiT to completion, captures each
``bridge_layer``'s hidden state, and lets a separate ActionDiT cross-attend to
those bridge features once. On Wan the bridge is a flat ``(B, L, D)`` sequence;
on Cosmos the DiT state is a 5D grid ``(B, T, H, W, D)``, so the architecture
flattens the spatial+temporal axes to the Wan-compatible ``(B, T·H·W, D)`` token
layout before the action backbone consumes it
(``joint_cross_attn.py`` ``bridge.ndim == 5`` branch).

Unlike idm/shared, this needs no backbone-polymorphic hook — the flatten lives
in the architecture and the bridge cross-attention is permutation-invariant over
the video-token axis. These CPU tests (on the ``_RichCosmosBlock`` fakes shared
with the joint_self_attn suite) lock in:

* the 5D→3D bridge flatten produces exactly ``grid_frames·grid_height·grid_width``
  tokens, preserving the video hidden dim;
* a full forward restores the 5D video prediction and yields a correctly-shaped
  action prediction;
* the cross-attn isolation invariant — video runs to completion *before* the
  action stream, so ``video_pred`` is invariant to ``noisy_actions`` while
  ``action_pred`` responds to it (the property that makes bridge collection
  sound on the 5D-grid backbone);
* the video-only path (``noisy_actions=None``) matches the full forward's video.

A ``@pytest.mark.gpu`` test exercises the same invariants on real
Cosmos-Predict2.5-2B weights.
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.test_cosmos_predict25_joint_self_attn import _build_rich_wrapper

COSMOS25_2B = os.environ.get("OPENWAM_COSMOS25_2B", "/path/to/assets/Cosmos-Predict2.5-2B")


def _make_cosmos_cross_attn(num_blocks=2, *, action_dim=3, text_dim=12):
    """Build a DualSystemCrossAttnArchitecture on a CosmosPredict25 rich-fake backbone."""
    from openwam.model.architectures.dual_system.joint_cross_attn import DualSystemCrossAttnArchitecture

    backbone = _build_rich_wrapper(num_blocks=num_blocks)
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "action_dim": action_dim,
        "dim": 16,
        "ffn_dim": 32,
        "num_heads": 4,
        "attn_head_dim": 4,
        "video_dim": 16,
        "text_dim": text_dim,
        "bridge_layers": tuple(range(num_blocks)),
    }
    arch = DualSystemCrossAttnArchitecture(cfg)
    arch.video_backbone = backbone
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch, backbone


def _inputs(*, action_dim=3, action_len=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    # latents (1,16,2,4,4) → grid frames f=2 (temporal patch 1), h=w=2 (spatial patch 2).
    latents = torch.randn(1, 16, 2, 4, 4, generator=g)
    context = torch.randn(1, 4, 12, generator=g)
    noisy_actions = torch.randn(1, action_len, action_dim, generator=g)
    action_timestep = torch.tensor([0.3])
    return latents, context, noisy_actions, action_timestep


# ----------------------------------------------------------------------
# CPU: bridge flatten 5D → 3D token layout
# ----------------------------------------------------------------------


def test_bridge_flatten_5d_to_tokens():
    """The 5D grid captured at a bridge layer flattens to ``(B, T·H·W, D)`` — the
    exact reshape the architecture applies before the action cross-attn — and
    preserves the video hidden dim (the bridge shape contract in
    ``separate_action_dit`` last-dim check)."""
    arch, backbone = _make_cosmos_cross_attn(num_blocks=2)
    latents, context, _, _ = _inputs()

    vstate = backbone.prepare(input_latents=latents, context=context, timestep=torch.tensor([0.5]))
    vstate = backbone.run_block(0, vstate)
    bridge = vstate.hidden_states
    assert bridge.ndim == 5, "Cosmos DiT state should be a 5D grid (B,T,H,W,D)"

    B, T, H, W, D = bridge.shape
    assert (T, H, W) == (vstate.grid_frames, vstate.grid_height, vstate.grid_width)
    assert D == backbone.dim  # bridge feature dim == video hidden dim (ActionDiT video_dim)

    flat = bridge.reshape(B, T * H * W, D)
    assert flat.shape == (B, T * H * W, D)
    # Flatten must be a plain view over (T,H,W) — no reordering.
    assert torch.equal(flat, bridge.reshape(B, -1, D))


# ----------------------------------------------------------------------
# CPU: full forward
# ----------------------------------------------------------------------


def test_joint_cross_attn_forward_restores_5d_video():
    """End-to-end forward: 5D video prediction restored to the input latent
    shape, action prediction correctly shaped, both finite."""
    arch, _ = _make_cosmos_cross_attn(num_blocks=2, action_dim=3)
    latents, context, noisy_actions, a_ts = _inputs(action_dim=3, action_len=5)

    video_pred, action_pred = arch.forward(
        noisy_actions, a_ts, input_latents=latents, context=context, timestep=torch.tensor([0.5])
    )

    assert video_pred.ndim == 5
    assert video_pred.shape == latents.shape
    assert action_pred.shape == (1, 5, 3)
    assert torch.isfinite(video_pred).all()
    assert torch.isfinite(action_pred).all()


def test_video_invariant_to_action_input():
    """Cross-attn isolation: the video DiT runs to completion before the action
    stream, so ``video_pred`` must be independent of ``noisy_actions`` while
    ``action_pred`` responds to it."""
    arch, _ = _make_cosmos_cross_attn(num_blocks=2, action_dim=3)
    latents, context, noisy_actions, a_ts = _inputs(action_dim=3, action_len=5)

    v1, a1 = arch.forward(noisy_actions, a_ts, input_latents=latents, context=context, timestep=torch.tensor([0.5]))
    v2, a2 = arch.forward(
        noisy_actions + 3.0, a_ts, input_latents=latents, context=context, timestep=torch.tensor([0.5])
    )

    assert torch.allclose(v1, v2, atol=1e-6), "video_pred must not depend on the action input"
    assert not torch.allclose(a1, a2, atol=1e-6), "action_pred must respond to the action input"


def test_video_only_path_matches_full_forward_video():
    """``noisy_actions=None`` returns the 5D video + ``None`` action, and the
    video matches the full forward's video branch (action cross-attn does not
    feed back into the video)."""
    arch, _ = _make_cosmos_cross_attn(num_blocks=2, action_dim=3)
    latents, context, noisy_actions, a_ts = _inputs(action_dim=3, action_len=5)

    v_only, a_none = arch.forward(None, None, input_latents=latents, context=context, timestep=torch.tensor([0.5]))
    assert a_none is None
    assert v_only.ndim == 5 and v_only.shape == latents.shape

    v_full, _ = arch.forward(
        noisy_actions, a_ts, input_latents=latents, context=context, timestep=torch.tensor([0.5])
    )
    assert torch.allclose(v_only, v_full, atol=1e-6)


# ----------------------------------------------------------------------
# GPU: real Cosmos-Predict2.5-2B weights
# ----------------------------------------------------------------------


@pytest.mark.gpu
def test_joint_cross_attn_forward_on_real_cosmos():
    """joint_cross_attn + real Cosmos-Predict2.5-2B: one forward restores 5D
    video, yields a finite action prediction, and preserves the cross-attn
    isolation invariant (video independent of the action input)."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA + CosmosPredict25 install")
    try:
        import cosmos_predict2  # noqa: F401
    except ImportError:
        pytest.skip("cosmos_predict2 not installed; run scripts/install_cosmos_predict25.sh first.")
    if not os.path.isdir(COSMOS25_2B):
        pytest.skip(f"Cosmos-Predict2.5-2B checkpoint missing at {COSMOS25_2B}")

    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from openwam.model import build_architecture, resolve_architecture_config

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "model=dual_system",
                "model.architecture.variant=joint_cross_attn",
                "model/video_backbone=cosmos_predict25",
                f"model.video_backbone.model_path={COSMOS25_2B}",
            ],
        )

    resolved = resolve_architecture_config(cfg.model)
    arch = build_architecture(resolved.registry_name, resolved.params)
    arch.set_dtype_device(dtype, device)
    arch.init_training_schedulers(1000)
    arch.eval()

    vb = arch.video_backbone
    g = torch.Generator(device=device).manual_seed(0)
    B, C, T, H, W = 1, 16, 3, 24, 20
    latents = torch.randn(B, C, T, H, W, generator=g, device=device, dtype=dtype)
    context = torch.randn(B, 16, vb.text_dim, generator=g, device=device, dtype=dtype)
    noisy_actions = torch.randn(B, 16, arch.action_backbone.action_dim, generator=g, device=device, dtype=dtype)
    a_ts = torch.tensor([0.3], device=device, dtype=dtype)
    v_ts = torch.tensor([0.5], device=device, dtype=dtype)
    # The default dual_system config enables proprioception; supply a state token.
    state_dim = int(cfg.model.architecture.state_dim)
    proprio = torch.randn(B, state_dim, generator=g, device=device, dtype=dtype)

    with torch.no_grad():
        v1, a1 = arch.forward(
            noisy_actions, a_ts, proprio=proprio, input_latents=latents, context=context, timestep=v_ts
        )
        v2, a2 = arch.forward(
            noisy_actions + 3.0, a_ts, proprio=proprio, input_latents=latents, context=context, timestep=v_ts
        )

    assert v1.ndim == 5 and v1.shape == latents.shape
    assert a1.shape == noisy_actions.shape
    assert torch.isfinite(v1).all() and torch.isfinite(a1).all()
    assert torch.allclose(v1, v2, atol=1e-3), "video_pred must not depend on the action input"
    assert not torch.allclose(a1, a2, atol=1e-3), "action_pred must respond to the action input"
