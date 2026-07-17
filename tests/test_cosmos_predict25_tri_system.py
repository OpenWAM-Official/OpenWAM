"""tri_system support for the CosmosPredict25 video backbone.

tri_system is a trimodal MoT: the video DiT, a separate ActionDiT, and an
Understanding Expert (fed by a VLM) run one mixed self-attention per layer.
Cross-modal visibility (``configs/model/tri_system.yaml``): video↔action per the
mask mode, and **understanding is a read-only tail — everyone attends to it, it
attends only to itself**.

The trimodal driver (``TriSystemMoTDriver``) is backbone-agnostic: it drives the
video stream through ``vb.pre_attn_at_layer`` / ``vb.post_attn_at_layer`` (which
flatten Cosmos's 5D grid ``(B,T,H,W,D)`` to the ``(B, S, H·D)`` MoT contract and
back) and counts video tokens as ``grid_frames · grid_height · grid_width`` via
``compute_video_tokens_per_frame`` — NOT ``hidden_states.shape[1]`` (which is
``T`` for the 5D-grid Cosmos state). This is the same path dual_system
joint_self_attn already uses on Cosmos, so tri_system needs no Cosmos-specific
code; these tests lock that in.

CPU tests (on the ``_RichCosmosBlock`` fakes shared with the joint_self_attn
suite) validate:

* the driver counts video tokens from the grid, not ``shape[1]==T``;
* a full trimodal forward restores the 5D video prediction + a correctly-shaped
  action prediction, both finite;
* joint self-attention couples the streams — ``video_pred`` responds to the
  action input;
* the read-only-tail isolation — video/action are invariant to the *values* of
  padding-masked understanding tokens, yet respond to *valid* understanding
  content.

A ``@pytest.mark.gpu`` test drives the real Cosmos-Predict2.5-2B DiT through the
trimodal loop (random action/understanding backbones + a stubbed VLM feeding
``vlm_hidden`` directly, so no Qwen3-VL dependency).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from tests.test_cosmos_predict25_joint_self_attn import _build_rich_wrapper

COSMOS25_2B = os.environ.get("OPENWAM_COSMOS25_2B", "/path/to/assets/Cosmos-Predict2.5-2B")


def _make_tri_arch(vb, *, action_dim=7, action_res_dim=24, und_dim=16, vlm_input_dim=20, device=None):
    """Wire a tri_system arch onto ``vb`` with random action + understanding
    backbones matched to the video backbone's head geometry / layer count.

    Works for both the CPU rich-Cosmos fake and the real Cosmos-2B backbone —
    all head/layer dims are derived from ``vb`` so the driver's parity checks
    pass either way. No VLM backbone; ``vlm_hidden`` is supplied at forward.
    """
    from openwam.model.action_backbone.separate_action_dit import ActionDiT
    from openwam.model.architectures.tri_system.joint_self_attn import TriSystemJointSelfAttnArchitecture
    from openwam.model.architectures.tri_system.mot_driver import TriSystemMoTDriver
    from openwam.model.architectures.tri_system.und_expert import UnderstandingExpert, UnderstandingExpertConfig

    n_layers = vb.num_layers
    ab = ActionDiT(
        action_dim=action_dim,
        dim=action_res_dim,
        ffn_dim=action_res_dim * 2,
        num_heads=vb.num_heads,
        num_layers=n_layers,
        video_dim=vb.dim,
        bridge_layers=tuple(range(n_layers)),
        variant="joint_self_attn",
        attn_head_dim=vb.head_dim,
        text_dim=vb.text_dim,
        eps=1.0e-6,
    )
    ub = UnderstandingExpert(
        UnderstandingExpertConfig(
            dim=und_dim,
            ffn_dim=und_dim * 2,
            num_layers=n_layers,
            vlm_input_dim=vlm_input_dim,
            vlm_projector_type="linear",
        ),
        wan_dim=vb.dim,
        wan_num_heads=vb.num_heads,
    )

    _dev = device if device is not None else torch.device("cpu")

    class _Arch(TriSystemJointSelfAttnArchitecture):
        # ``device`` is a class attribute (nn.Module rejects instance assignment).
        device = _dev

        def __init__(self):
            nn.Module.__init__(self)
            self.video_backbone = vb
            self.action_backbone = ab
            self.understanding_expert = ub
            self.vlm_backbone = None
            self._mot_driver = TriSystemMoTDriver(vb, ab, ub, mot_checkpoint_mixed_attn=False)

    arch = _Arch()
    return arch, ab, ub


def _make_tri_cosmos_fake(num_blocks=2, **kw):
    vb = _build_rich_wrapper(num_blocks=num_blocks)  # dim=16, heads=4, head_dim=4, text_dim(context)=12
    arch, ab, ub = _make_tri_arch(vb, **kw)
    return arch, vb, ab, ub


def _fwd_kwargs(vb, ab, ub, *, batch=2, action_len=4, und_len=5, und_mask=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Asymmetric grid (…,2,4,6) → frames=2, H=2, W=3 (distinct; h·w=6 ≠ h+w=5 ≠ frames).
    latents = torch.randn(1, 16, 2, 4, 6, generator=g).repeat(batch, 1, 1, 1, 1)
    context = torch.randn(batch, 4, vb.text_dim, generator=g)
    if und_mask is None:
        und_mask = torch.ones(batch, und_len, dtype=torch.bool)
    return dict(
        noisy_actions=torch.randn(batch, action_len, ab.action_dim, generator=g),
        action_timestep=torch.full((batch,), 0.4),
        context=context,
        context_mask=torch.ones(batch, 4, dtype=torch.bool),
        vlm_hidden=torch.randn(batch, und_len, ub.cfg.vlm_input_dim, generator=g),
        vlm_attention_mask=und_mask,
        input_latents=latents,
        timestep=torch.full((batch,), 0.5),
    )


# ----------------------------------------------------------------------
# CPU: video-token count from the grid, not shape[1]==T
# ----------------------------------------------------------------------


def test_driver_counts_video_tokens_from_grid_not_frames():
    """The trimodal driver must size the video stream as
    ``grid_frames · grid_height · grid_width`` (the flattened token count), not
    ``hidden_states.shape[1]`` — which is ``T`` for the 5D-grid Cosmos state.

    Uses an asymmetric grid (frames=2, H=2, W=3) so the three plausible wrong
    formulas are all distinguishable from the right one: ``h·w=6`` ≠ ``h+w=5`` ≠
    ``frames=2`` ≠ ``shape[1]=T=2``.
    """
    arch, vb, _, _ = _make_tri_cosmos_fake(num_blocks=2)
    latents = torch.randn(2, 16, 2, 4, 6)
    vstate = vb.prepare(input_latents=latents, context=torch.randn(2, 4, vb.text_dim), timestep=torch.zeros(2))

    tpf = arch._mot_driver._video_tokens_per_frame(vstate)
    s_video = int(vstate.grid_frames) * tpf

    assert vstate.hidden_states.ndim == 5
    assert (vstate.grid_frames, vstate.grid_height, vstate.grid_width) == (2, 2, 3)
    assert tpf == vstate.grid_height * vstate.grid_width == 6  # not h+w=5
    assert s_video == 12
    # The bug this guards: shape[1] is T (== grid_frames == 2), not the 12 tokens.
    assert s_video != vstate.hidden_states.shape[1]


# ----------------------------------------------------------------------
# CPU: full trimodal forward
# ----------------------------------------------------------------------


def test_tri_system_forward_restores_5d_video():
    arch, vb, ab, ub = _make_tri_cosmos_fake(num_blocks=2, action_dim=7)
    kw = _fwd_kwargs(vb, ab, ub)

    video_pred, action_pred = arch.forward(**kw)

    assert video_pred.ndim == 5
    assert video_pred.shape == kw["input_latents"].shape
    assert action_pred.shape == (2, 4, 7)
    assert torch.isfinite(video_pred).all()
    assert torch.isfinite(action_pred).all()


def test_action_sees_video_cross_modal_coupling():
    """Under the default ``action_sees_video`` mask, the trimodal mixed attention
    lets the action stream read the video stream but not vice-versa. On the 5D
    Cosmos grid: ``action_pred`` responds to the video latents, while
    ``video_pred`` is invariant to the action input."""
    arch, vb, ab, ub = _make_tri_cosmos_fake(num_blocks=2, action_dim=7)
    kw = _fwd_kwargs(vb, ab, ub, seed=1)

    v1, a1 = arch.forward(**kw)

    # Different action input, same video → video unchanged (video can't see action).
    v_act, a_act = arch.forward(**dict(kw, noisy_actions=kw["noisy_actions"] + 5.0))
    assert torch.allclose(v1, v_act, atol=1e-6), "video must not see the action stream"
    assert not torch.allclose(a1, a_act, atol=1e-6), "action_pred must depend on its own noised input"

    # Different video latents → action changes (action sees video).
    v_vid, a_vid = arch.forward(**dict(kw, input_latents=kw["input_latents"] + 0.5))
    assert not torch.allclose(a1, a_vid, atol=1e-6), "action_pred must couple to the video stream"


def test_understanding_read_only_tail_isolation():
    """Understanding is a read-only tail: video/action attend to *valid*
    understanding tokens (so their outputs depend on that content) but padding-
    masked understanding tokens are blocked (outputs invariant to their values).
    Validates the cross-modal mask on the 5D-grid Cosmos geometry."""
    arch, vb, ab, ub = _make_tri_cosmos_fake(num_blocks=2, action_dim=7)
    und_mask = torch.tensor([[True, True, True, False, False], [True, True, True, False, False]])
    kw = _fwd_kwargs(vb, ab, ub, und_mask=und_mask, seed=2)

    v1, a1 = arch.forward(**kw)

    # Perturb ONLY padding-masked understanding positions → outputs must not move.
    vlm_pad = kw["vlm_hidden"].clone()
    vlm_pad[:, 3:, :] += 100.0
    v_pad, a_pad = arch.forward(**dict(kw, vlm_hidden=vlm_pad))
    assert torch.allclose(v1, v_pad, atol=1e-5), "video leaked from padding-masked understanding tokens"
    assert torch.allclose(a1, a_pad, atol=1e-5), "action leaked from padding-masked understanding tokens"

    # Perturb a VALID understanding position → outputs must respond (tail is read).
    vlm_valid = kw["vlm_hidden"].clone()
    vlm_valid[:, 0, :] += 100.0
    v_valid, _ = arch.forward(**dict(kw, vlm_hidden=vlm_valid))
    assert not torch.allclose(v1, v_valid, atol=1e-5), "video should read valid understanding content"


# ----------------------------------------------------------------------
# GPU: real Cosmos-Predict2.5-2B DiT in the trimodal loop
# ----------------------------------------------------------------------


@pytest.mark.gpu
def test_tri_system_forward_on_real_cosmos(stub_reason1):
    """Drive the real Cosmos-Predict2.5-2B DiT through the trimodal MoT loop with
    random action/understanding backbones (VLM stubbed via ``vlm_hidden``): the
    forward restores 5D video, yields a finite action prediction, and preserves
    the read-only-tail understanding isolation on real weights."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA + CosmosPredict25 install")
    try:
        import cosmos_predict2  # noqa: F401
    except ImportError:
        pytest.skip("cosmos_predict2 not installed; run scripts/install_cosmos_predict25.sh first.")
    if not os.path.isdir(COSMOS25_2B):
        pytest.skip(f"Cosmos-Predict2.5-2B checkpoint missing at {COSMOS25_2B}")

    from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    vb = CosmosPredict25VideoBackbone.from_pretrained(
        {
            "video_backbone": {
                "model_path": COSMOS25_2B,
                "text_encoder_path": "/stub",
                "vae": "none",
                "shift_video": 5.0,
            }
        },
        device=device,
    )
    arch, ab, ub = _make_tri_arch(vb, action_dim=20, action_res_dim=256, und_dim=256, vlm_input_dim=1024, device=device)
    arch.to(device=device, dtype=dtype)

    B = 1
    g = torch.Generator(device=device).manual_seed(0)
    latents = torch.randn(B, 16, 3, 24, 20, generator=g, device=device, dtype=dtype)
    context = torch.randn(B, 16, vb.text_dim, generator=g, device=device, dtype=dtype)
    und_mask = torch.tensor([[True, True, True, True, True, True, False, False]], device=device)
    kw = dict(
        noisy_actions=torch.randn(B, 16, ab.action_dim, generator=g, device=device, dtype=dtype),
        action_timestep=torch.full((B,), 0.4, device=device, dtype=dtype),
        context=context,
        context_mask=torch.ones(B, 16, dtype=torch.bool, device=device),
        vlm_hidden=torch.randn(B, 8, ub.cfg.vlm_input_dim, generator=g, device=device, dtype=dtype),
        vlm_attention_mask=und_mask,
        input_latents=latents,
        timestep=torch.full((B,), 0.5, device=device, dtype=dtype),
    )

    with torch.no_grad():
        v1, a1 = arch.forward(**kw)
        vlm_pad = kw["vlm_hidden"].clone()
        vlm_pad[:, 6:, :] += 100.0
        v_pad, a_pad = arch.forward(**dict(kw, vlm_hidden=vlm_pad))

    assert v1.ndim == 5 and v1.shape == latents.shape
    assert a1.shape == kw["noisy_actions"].shape
    assert torch.isfinite(v1).all() and torch.isfinite(a1).all()
    assert torch.allclose(v1, v_pad, atol=1e-3), "video leaked from padding-masked understanding tokens"
    assert torch.allclose(a1, a_pad, atol=1e-3), "action leaked from padding-masked understanding tokens"
