"""Train ↔ deploy numerical consistency tests for tri_system and dual_system_idm.

Both architectures take different shapes between training and deployment:

- ``dual_system_idm`` trains a 3-branch teacher-forcing forward
  ([noisy_video, cond_video, action]) and deploys a two-stage path
  (Stage 1 video-only → Stage 2 action with frozen video KV cache).
  The teacher-forcing mask blocks ``noisy_video → {cond_video, action}``,
  so the training action prediction must equal the Stage-2 action
  prediction when ``cond_video`` is identical. Any mask drift or KV
  layout regression breaks this contract silently.

- ``tri_system`` runs the same trimodal MoT forward at train and infer
  time; the only delta is that deploy computes ``vlm_hidden`` once
  outside ``forward()`` and passes it in as a cached tensor, while
  train re-extracts it from ``vlm_inputs`` inside ``forward()``. The
  contract: identical outputs in ``eval()`` mode regardless of which
  call style is used.

Existing coverage (``test_idm_video_cache_matches_joint_loop``) compares
``run_joint_loop`` vs ``prefill_video_cache + run_action_with_video_cache``
— that catches deploy-internal regressions but does not exercise the
*training-side* 3-branch path. These tests close that gap.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from tests.test_dual_system_idm import _CapturePrepareVideoBackbone, _make_idm_with_video

# ---------------------------------------------------------------------------
# IDM: train (3-branch teacher-forcing) ↔ deploy (Stage-2 video KV cache)
# ---------------------------------------------------------------------------


def _make_action_inputs(arch, *, B=1, seq=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    action_latents = torch.randn(B, seq, arch.action_backbone.action_dim, generator=g)
    a_timestep = torch.tensor([0.5])
    context = torch.randn(B, 2, arch.action_backbone.text_dim, generator=g)
    context_mask = torch.ones(B, 2, dtype=torch.bool)
    return action_latents, a_timestep, context, context_mask


def test_idm_train_action_matches_deploy_stage2():
    """3-branch train action_pred must equal Stage-2 deploy action_pred.

    Train: ``run_idm_training_loop([noisy_v, cond_v, action])`` with a
    teacher-forcing mask that blocks ``noisy_v → action``.
    Deploy: ``prefill_video_cache(cond_v) → run_action_with_video_cache(action)``.

    With identical ``cond_v`` (and any noisy_v, since it's masked off),
    both paths must produce the same action prediction. This is the
    FastWAM-IDM core invariant; if the teacher-forcing mask ever leaks
    ``noisy → action`` or the prefill K/V layout drifts from the
    training cond branch's K/V, this test fails numerically.
    """
    torch.manual_seed(0)
    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.eval()
    driver = arch._mot_driver

    B = 1
    cond_latents = torch.randn(B, 1, 1, 1, 1)
    noisy_latents = torch.randn(B, 1, 1, 1, 1)

    vstate_cond_train = vb.prepare(latents=cond_latents, timestep=torch.zeros(B))
    vstate_noisy_train = vb.prepare(latents=noisy_latents, timestep=torch.tensor([0.7]))
    vstate_cond_deploy = vb.prepare(latents=cond_latents, timestep=torch.zeros(B))

    action_latents, a_timestep, context, context_mask = _make_action_inputs(arch, B=B)
    astate_train = arch.action_backbone.prepare_state(
        action_latents, a_timestep, context=context, context_mask=context_mask
    )
    astate_deploy = arch.action_backbone.prepare_state(
        action_latents, a_timestep, context=context, context_mask=context_mask
    )

    _, _, astate_train = driver.run_idm_training_loop(vstate_noisy_train, vstate_cond_train, astate_train)
    pred_train = arch.action_backbone.extract_prediction(astate_train)

    kv_cache, _, _ = driver.prefill_video_cache(vstate_cond_deploy)
    astate_deploy = driver.run_action_with_video_cache(
        astate_deploy,
        video_kv_cache=kv_cache,
        video_seq_len=int(vstate_cond_deploy.hidden_states.shape[1]),
    )
    pred_deploy = arch.action_backbone.extract_prediction(astate_deploy)

    assert torch.allclose(pred_train, pred_deploy, atol=1e-5, rtol=1e-5), (
        "IDM train↔deploy action_pred diverged. "
        "Either the teacher-forcing mask leaks noisy→action, or the "
        "prefill_video_cache K/V layout differs from the training cond branch.\n"
        f"  train: {pred_train.flatten()[:8]}\n"
        f"  deploy: {pred_deploy.flatten()[:8]}"
    )


def test_idm_train_action_invariant_under_noisy_video_perturbation():
    """Stronger version: action_pred must be bit-identical under any noisy_video.

    The teacher-forcing mask blocks both ``noisy → action`` and
    ``noisy → cond`` (which would otherwise leak via action ← cond).
    Perturbing only the noisy_video branch should leave the action
    output unchanged. A leak anywhere in this chain breaks the
    train↔deploy alignment.
    """
    torch.manual_seed(0)
    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.eval()
    driver = arch._mot_driver

    B = 1
    cond_latents = torch.randn(B, 1, 1, 1, 1)
    action_latents, a_timestep, context, context_mask = _make_action_inputs(arch, B=B)

    def _run(noisy_seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(noisy_seed)
        noisy_latents = torch.randn(B, 1, 1, 1, 1, generator=g)
        noisy_t = torch.tensor([0.3 + 0.1 * noisy_seed])
        vstate_n = vb.prepare(latents=noisy_latents, timestep=noisy_t)
        vstate_c = vb.prepare(latents=cond_latents, timestep=torch.zeros(B))
        astate = arch.action_backbone.prepare_state(
            action_latents, a_timestep, context=context, context_mask=context_mask
        )
        _, _, astate = driver.run_idm_training_loop(vstate_n, vstate_c, astate)
        return arch.action_backbone.extract_prediction(astate)

    out_a = _run(1)
    out_b = _run(2)

    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6), (
        "Action prediction changed when only noisy_video changed — "
        "teacher-forcing mask leaks noisy → {cond, action}.\n"
        f"  seed 1: {out_a.flatten()[:8]}\n"
        f"  seed 2: {out_b.flatten()[:8]}"
    )


# ---------------------------------------------------------------------------
# tri_system: train (lazy VLM in forward) ↔ deploy (cached vlm_hidden)
# ---------------------------------------------------------------------------


class _TriStubVideoBackbone(nn.Module):
    """Video backbone stub compatible with tri_system arch.forward.

    Provides the minimal Wan-shaped surface ``forward`` and the MoT driver
    touch: ``prepare`` / ``run_block`` / ``pre_attn_at_layer`` /
    ``post_attn_at_layer`` / ``build_video_to_video_mask`` / ``finalize``.
    Mirrors the dual_system test's ``_CapturePrepareVideoBackbone`` but
    matches tri_system's larger head-dim contract.
    """

    num_layers = 1

    def __init__(self, dim=24, num_heads=4):
        super().__init__()
        from tests.test_openwam_trainer import _MockScheduler

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scheduler = _MockScheduler()
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self.submodule_names = []

    @property
    def video_attention_mask_mode(self):
        return "bidirectional"

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode):
        del mode

    def set_dtype_device(self, dtype, device):
        self._dtype = dtype
        self._device = device

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        del video_tokens_per_frame
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    def prepare(self, **kw):
        from openwam.model.video_backbone.base import BlockLoopState

        latents = kw["latents"]
        timestep = kw["timestep"]
        B, _, f, h, w = latents.shape
        s = f * h * w
        x = latents[:, :1].reshape(B, s, 1).expand(B, s, self.dim).contiguous()
        freqs = torch.polar(
            torch.ones(s, 1, self.head_dim // 2),
            torch.zeros(s, 1, self.head_dim // 2),
        )
        t_mod = timestep.view(B, 1, 1, 1).expand(B, s, 6, self.dim).contiguous()
        context = torch.zeros(B, 2, self.dim)
        context_mask = torch.ones(B, 2, dtype=torch.bool)
        return BlockLoopState(
            hidden_states=x,
            time_mod=t_mod,
            rope_freqs=freqs,
            context=context,
            context_mask=context_mask,
            grid_frames=f,
            grid_height=h,
            grid_width=w,
        )

    def pre_attn_at_layer(self, layer_id, state):
        del layer_id
        return state.hidden_states, state.hidden_states, state.hidden_states, {"residual": state.hidden_states}

    def post_attn_at_layer(self, layer_id, state, attn_out, post_state):
        del layer_id
        state.hidden_states = post_state["residual"] + attn_out
        return state

    def run_block(self, block_id, state):
        del block_id
        state.hidden_states = state.hidden_states + 1
        return state

    def finalize(self, state):
        B = state.hidden_states.shape[0]
        return (
            state.hidden_states[:, :, :1]
            .transpose(1, 2)
            .reshape(B, 1, state.grid_frames, state.grid_height, state.grid_width)
        )

    def decode_video(self, latents, *, tiled=True):
        del latents, tiled
        return None


def _make_stub_tri_arch_for_forward(monkeypatch, num_video_layers=1, vlm_input_dim=12, vlm_seq=4):
    """Build a tri_system arch wired to a controllable stub VB + deterministic VLM.

    The VLM stub is a tiny ``nn.Linear`` so ``extract_features`` is
    deterministic in eval mode but non-trivial (output depends on input).
    """
    from openwam.model.architectures.tri_system import joint_self_attn as tri_mod
    from openwam.model.architectures.tri_system.joint_self_attn import (
        TriSystemJointSelfAttnArchitecture,
    )

    class _StubVLM(nn.Module):
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            super().__init__()
            self._proj = nn.Linear(8, vlm_input_dim)

        @property
        def hidden_size(self):
            return vlm_input_dim

        def prepare_vlm_inputs(self, prompts, images):  # noqa: ARG002
            return {"input_ids": torch.ones(1, vlm_seq, 8), "attention_mask": torch.ones(1, vlm_seq, dtype=torch.bool)}

        def extract_features(self, vlm_inputs):
            return self._proj(vlm_inputs["input_ids"].float())

    class _Arch(TriSystemJointSelfAttnArchitecture):
        def _init_video_backbone(self, cfg):  # noqa: ARG002
            self.video_backbone = _TriStubVideoBackbone()

    monkeypatch.setattr(tri_mod, "build_vlm_backbone", lambda *a, **k: _StubVLM())

    cfg = {
        "vlm_backbone": {"checkpoint_path": "", "load_pretrained": False},
        "understanding_expert": {
            "dim": 16,
            "ffn_dim": 32,
            "vlm_projector_type": "linear",
        },
        "action_dim": 5,
        "dim": 24,
        "ffn_dim": 48,
        "num_heads": 4,
        "attn_head_dim": 6,
        "text_dim": 16,
        "mot_checkpoint_mixed_attn": False,
        "bridge_interval": 1,
    }
    arch = _Arch(cfg)
    # BaseWAMArchitecture defaults _device to cuda; force CPU without invoking
    # set_dtype_device (our stub VLM doesn't implement that contract).
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    arch.eval()
    return arch


def test_tri_system_forward_vlm_inputs_matches_vlm_hidden(monkeypatch):
    """Train (vlm_inputs in forward) and deploy (cached vlm_hidden) must agree.

    In training ``arch.forward(..., vlm_inputs=X)`` lazy-computes
    ``vlm_hidden = self.vlm_backbone.extract_features(X)`` inside
    forward. In deploy ``arch.generate()`` pre-computes ``vlm_hidden``
    once and passes it through ``super().generate``, which forwards it
    via ``inputs_shared`` to every per-step ``arch.forward(...,
    vlm_hidden=h)`` call without re-extracting.

    The two paths share the rest of the forward (proprio handling,
    vstate / astate / ustate construction, MoT driver, finalize), so
    they MUST be bit-identical in ``eval()`` mode. Drift here would
    indicate either non-determinism in ``extract_features`` (unexpected
    dropout / RNG) or a code-path divergence between the two call
    styles.
    """
    torch.manual_seed(0)
    arch = _make_stub_tri_arch_for_forward(monkeypatch)
    B = 1
    seq_action = 3
    f, h, w = 1, 2, 3
    latents = torch.randn(B, 1, f, h, w)
    timestep = torch.tensor([10.0])
    noisy_actions = torch.randn(B, seq_action, arch.action_backbone.action_dim)
    action_timestep = torch.tensor([0.5])
    context = torch.randn(B, 2, arch.action_backbone.text_dim)
    context_mask = torch.ones(B, 2, dtype=torch.bool)
    vlm_inputs = arch.vlm_backbone.prepare_vlm_inputs(["task"], [None])

    with torch.no_grad():
        # Train path: forward owns the VLM call.
        video_train, action_train = arch.forward(
            noisy_actions,
            action_timestep,
            latents=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            vlm_inputs=vlm_inputs,
        )

        # Deploy path: VLM precomputed once outside forward.
        vlm_hidden = arch.vlm_backbone.extract_features(vlm_inputs)
        vlm_attention_mask = vlm_inputs["attention_mask"]
        video_deploy, action_deploy = arch.forward(
            noisy_actions,
            action_timestep,
            latents=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            vlm_hidden=vlm_hidden,
            vlm_attention_mask=vlm_attention_mask,
        )

    assert torch.allclose(video_train, video_deploy, atol=1e-6, rtol=1e-6), (
        "tri_system video output diverged between forward(vlm_inputs=...) "
        "and forward(vlm_hidden=cached_h). The two call styles must be "
        "numerically identical in eval mode."
    )
    assert action_train is not None and action_deploy is not None
    assert torch.allclose(action_train, action_deploy, atol=1e-6, rtol=1e-6), (
        "tri_system action output diverged between forward(vlm_inputs=...) and forward(vlm_hidden=cached_h)."
    )


def test_tri_system_forward_action_invariant_under_vlm_attention_mask_padding(monkeypatch):
    """Padding keys in ``vlm_attention_mask`` must not influence action_pred.

    Deploy may pass a ``vlm_attention_mask`` that pads some keys to a
    common length across batch. The trimodal joint attention's
    understanding-key padding mask must block those keys cleanly; if
    they leak, two ``vlm_hidden`` tensors that differ only in their
    padded suffix will produce different action outputs.

    This pins the deploy-time padding contract.
    """
    torch.manual_seed(0)
    arch = _make_stub_tri_arch_for_forward(monkeypatch, vlm_seq=4)
    B = 1
    f, h, w = 1, 2, 3
    latents = torch.randn(B, 1, f, h, w)
    timestep = torch.tensor([10.0])
    noisy_actions = torch.randn(B, 3, arch.action_backbone.action_dim)
    action_timestep = torch.tensor([0.5])
    context = torch.randn(B, 2, arch.action_backbone.text_dim)
    context_mask = torch.ones(B, 2, dtype=torch.bool)

    base_inputs = arch.vlm_backbone.prepare_vlm_inputs(["task"], [None])
    with torch.no_grad():
        h_base = arch.vlm_backbone.extract_features(base_inputs)
    valid_len = 2
    attn_mask = torch.zeros(1, h_base.shape[1], dtype=torch.bool)
    attn_mask[:, :valid_len] = True

    def _run(perturb_pad: bool) -> torch.Tensor:
        h_perturbed = h_base.clone()
        if perturb_pad:
            h_perturbed[:, valid_len:] += 100.0
        with torch.no_grad():
            _, action_out = arch.forward(
                noisy_actions,
                action_timestep,
                latents=latents,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                vlm_hidden=h_perturbed,
                vlm_attention_mask=attn_mask,
            )
        return action_out

    out_a = _run(perturb_pad=False)
    out_b = _run(perturb_pad=True)
    assert torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5), (
        "Perturbing padded keys in vlm_hidden changed the action prediction "
        "— vlm_attention_mask is not being applied. Padded VLM tokens must "
        "not influence trimodal attention output."
    )


# ---------------------------------------------------------------------------
# Optional GPU integration variants (require real backbones; gated by env var)
# ---------------------------------------------------------------------------


_GPU_INTEGRATION_FLAG = "OPENWAM_RUN_TRAIN_DEPLOY_GPU"


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get(_GPU_INTEGRATION_FLAG, "0") != "1" or not torch.cuda.is_available(),
    reason=f"set {_GPU_INTEGRATION_FLAG}=1 and have a CUDA GPU to run the integration variant",
)
def test_idm_train_deploy_consistency_gpu():
    """Real-DiT IDM consistency: same contract as the CPU test, exercised on a real DiT block.

    We can't load Wan2.2-TI2V-5B here without a full checkpoint setup,
    so this re-runs the CPU stub on CUDA to validate device-portability
    of the consistency invariant. If the user wants a Wan-real run,
    add a separate test gated on the model_path env var.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.eval()
    arch.set_dtype_device(torch.float32, device)
    driver = arch._mot_driver

    B = 1
    cond_latents = torch.randn(B, 1, 1, 1, 1, device=device)
    noisy_latents = torch.randn(B, 1, 1, 1, 1, device=device)

    vstate_cond_train = vb.prepare(latents=cond_latents, timestep=torch.zeros(B, device=device))
    vstate_noisy_train = vb.prepare(latents=noisy_latents, timestep=torch.tensor([0.7], device=device))
    vstate_cond_deploy = vb.prepare(latents=cond_latents, timestep=torch.zeros(B, device=device))

    action_latents = torch.randn(B, 3, arch.action_backbone.action_dim, device=device)
    a_timestep = torch.tensor([0.5], device=device)
    context = torch.randn(B, 2, arch.action_backbone.text_dim, device=device)
    context_mask = torch.ones(B, 2, dtype=torch.bool, device=device)

    astate_train = arch.action_backbone.prepare_state(
        action_latents, a_timestep, context=context, context_mask=context_mask
    )
    astate_deploy = arch.action_backbone.prepare_state(
        action_latents, a_timestep, context=context, context_mask=context_mask
    )
    _, _, astate_train = driver.run_idm_training_loop(vstate_noisy_train, vstate_cond_train, astate_train)
    pred_train = arch.action_backbone.extract_prediction(astate_train)
    kv_cache, _, _ = driver.prefill_video_cache(vstate_cond_deploy)
    astate_deploy = driver.run_action_with_video_cache(
        astate_deploy,
        video_kv_cache=kv_cache,
        video_seq_len=int(vstate_cond_deploy.hidden_states.shape[1]),
    )
    pred_deploy = arch.action_backbone.extract_prediction(astate_deploy)
    assert torch.allclose(pred_train, pred_deploy, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get(_GPU_INTEGRATION_FLAG, "0") != "1" or not torch.cuda.is_available(),
    reason=f"set {_GPU_INTEGRATION_FLAG}=1 and have a CUDA GPU to run the integration variant",
)
def test_tri_system_train_deploy_consistency_gpu(monkeypatch):
    """Same tri_system contract as the CPU test, exercised on CUDA."""
    arch = _make_stub_tri_arch_for_forward(monkeypatch)
    device = torch.device("cuda")
    # Stub VLM has no ``set_dtype_device``; move modules directly and
    # update the cached device/dtype state on the architecture.
    arch._device = device
    arch._dtype = torch.float32
    arch.to(device)
    B = 1
    f, h, w = 1, 2, 3
    latents = torch.randn(B, 1, f, h, w, device=device)
    timestep = torch.tensor([10.0], device=device)
    noisy_actions = torch.randn(B, 3, arch.action_backbone.action_dim, device=device)
    action_timestep = torch.tensor([0.5], device=device)
    context = torch.randn(B, 2, arch.action_backbone.text_dim, device=device)
    context_mask = torch.ones(B, 2, dtype=torch.bool, device=device)
    vlm_inputs = arch.vlm_backbone.prepare_vlm_inputs(["task"], [None])
    vlm_inputs = {k: v.to(device) for k, v in vlm_inputs.items()}

    with torch.no_grad():
        _, a_train = arch.forward(
            noisy_actions,
            action_timestep,
            latents=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            vlm_inputs=vlm_inputs,
        )
        vlm_hidden = arch.vlm_backbone.extract_features(vlm_inputs)
        _, a_deploy = arch.forward(
            noisy_actions,
            action_timestep,
            latents=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            vlm_hidden=vlm_hidden,
            vlm_attention_mask=vlm_inputs["attention_mask"],
        )
    assert torch.allclose(a_train, a_deploy, atol=1e-5, rtol=1e-5)
