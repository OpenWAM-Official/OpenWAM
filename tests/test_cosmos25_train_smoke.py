"""GPU smoke for Cosmos-Predict2.5 TI2V training plumbing.

Real-path tests against the 2B post-trained weights. WAM's canonical
conditioning is TI2V (text + first frame → tail video); pure T2V is not
exercised because WAM never trains without the first-frame observation.

1. ``test_cosmos25_train_smoke_ti2v_real_vae_real_cache`` — real Wan2pt1
   VAE encode + ``(512, 1024)`` text cache. Pins the three real-training
   bugs (Bug #1 ``move_frozen_to_device`` over ActionDiT; Bug #2
   ``Cosmos25VideoBackbone.get_submodule('vae') is None``; Bug #3
   ``downsample_video_mask_to_latent`` is called with ``skip_first=True``
   on a TI2V batch). Also re-pins the post-projection 0-hit cache
   invariant for ``net.crossattn_proj``.
2. ``test_cosmos25_train_smoke_ti2v_with_live_text_encoder`` — live
   Reason1-7B encoder produces pre-projection ``(B, 512, 100352)``; the
   wrapper's in-line ``net.crossattn_proj`` fires once per
   ``preprocess_input``.
3. ``test_cosmos25_train_smoke_ti2v_live_encoder_with_dropout`` — live
   encoder + ``text_encoder_dropout=1.0`` still trains end-to-end.
4. ``test_cosmos25_train_smoke_self_attn_ti2v`` — joint self-attn
   (``MoTJointDriver``) end-to-end with first-frame conditioning.

All share ``pytestmark = pytest.mark.gpu`` and skip unless CUDA,
``cosmos_predict2``, and the asset bundle are all present.

TI2V is auto-activated by ``FirstFrameConditioningTransform`` (wrapped
into ``BaseWAMArchitecture.prepare_inputs`` at ``base.py:493-500`` with
``use_first_frame_as_reference=True``): any sample whose ``video`` field
is a list of PIL frames will pick up ``first_frame_image=[video[0]]``
automatically, which flows to the wrapper's ``preprocess_input`` as
``ref_images`` and triggers TI2V key emission.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu


ASSET_PATH = Path(os.environ.get("COSMOS25_ASSET_PATH", "/path/to/assets/Cosmos-Predict2.5-2B"))
REPO_ROOT = Path(__file__).resolve().parents[1]
COSMOS_CFG_PATH = REPO_ROOT / "configs" / "model" / "dual_system_cosmos25.yaml"
COSMOS_SELF_ATTN_CFG_PATH = REPO_ROOT / "configs" / "model" / "dual_system_self_attn_cosmos25.yaml"


def _skip_unless_runnable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if not ASSET_PATH.exists():
        pytest.skip(f"Cosmos asset bundle missing at {ASSET_PATH}.")
    if not COSMOS_CFG_PATH.exists():
        pytest.skip(f"Cosmos config missing at {COSMOS_CFG_PATH}.")
    try:
        import cosmos_predict2  # noqa: F401
    except ImportError:
        pytest.skip("cosmos_predict2 not installed; run scripts/install_cosmos25.sh first.")


def _load_cfg():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(COSMOS_CFG_PATH))
    cfg.video_backbone.model_path = str(ASSET_PATH)
    return cfg


def _make_stub_sample():
    """Synthesize a single dataset sample at the Cosmos2.5 config geometry.

    Uses ``T_video=13`` to match ``num_frames: 13`` in the YAML. With TI2V
    auto-injected by ``FirstFrameConditioningTransform``, frame 0 becomes
    the clean conditioning latent and the tail of 3 latents are predicted
    (``T_lat = 1 + (13-1)//4 = 4`` total, ``skip_first=True`` ⇒ video_is_pad
    covers the 3 tail latents).

    13 PIL frames at ``256x320`` RGB so the real Wan2pt1 VAE encode
    (``_pil_video_to_tensor``) gets the shape it expects.
    """
    import PIL.Image

    return {
        "prompt": "smoke",
        "action": np.zeros((12, 20), dtype=np.float32),
        "proprio": np.zeros((20,), dtype=np.float32),
        "action_mask": np.ones((12,), dtype=bool),
        "video_mask": np.ones((13,), dtype=bool),
        "video": [PIL.Image.fromarray((np.random.rand(256, 320, 3) * 255).astype(np.uint8)) for _ in range(13)],
    }


def test_cosmos25_train_smoke_ti2v_real_vae_real_cache(monkeypatch):
    """TI2V real-path regression: real Wan2pt1 VAE encode +
    post-projection ``(L=512, D=1024) bf16`` text cache + auto-injected
    first frame — drives every plumbing path that only fires during real
    training. Locks in:

    - **Bug #1** (``base.py:417-424``): ``move_frozen_to_device`` survives
      iteration over backbones that lack ``text_encoder`` / ``vae``
      sub-modules (e.g. ActionDiT). The inner ``bb.get_submodule(name)`` is
      wrapped in ``try/except (AttributeError, KeyError)``; without the
      wrap, the call crashes because ``nn.Module.get_submodule`` raises
      ``AttributeError`` on a missing name rather than returning ``None``.
    - **Bug #2** (``adapter.py:318-331``): ``Cosmos25VideoBackbone.get_submodule``
      returns ``None`` when the named attribute is not an ``nn.Module``.
      ``Wan2pt1VAEInterface`` is a plain Python object, so returning it
      from ``get_submodule`` would crash ``move_frozen_to_device`` at
      ``.to(device=device)``.
    - **Bug #3** (``architecture_utils.py:13-67`` +
      ``base.py:prepare_inputs``): ``downsample_video_mask_to_latent``
      receives ``skip_first=True`` because TI2V is active
      (``first_frame_latents`` populated). The returned mask covers the
      tail latents only (``T_latent_tail = ceil((T_video - 1) / k)``),
      matching ``noise_pred[:, :, 1:]`` shape after the
      ``num_clean_prefix_frames=1`` skip in ``_compute_video_loss``.

    The post-projection 0-hit invariant for ``crossattn_proj`` is
    re-asserted here on the real-VAE path: when the caller hands in a
    ``(L, 1024)`` cache tensor, ``preprocess_input`` must NOT call the
    100352→1024 projection again.
    """
    _skip_unless_runnable()

    import openwam.utils as _utils_mod
    from openwam.model import build_architecture, resolve_architecture_config

    cfg = _load_cfg()
    resolved = resolve_architecture_config(cfg)
    arch = build_architecture(resolved.registry_name, resolved.params)

    # — Bug #2 explicit ABC contract —
    assert arch.video_backbone.get_submodule("vae") is None, (
        "Cosmos25VideoBackbone.get_submodule('vae') must return None per "
        "VideoBackbone ABC (Wan2pt1VAEInterface is not an nn.Module). "
        "See docs/cosmos25_backbone.md §12.3.2."
    )

    arch.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
    # — Bug #1 + #2 sequence: must not raise on ActionDiT or on the
    #   plain-Python VAE attribute. —
    arch.move_frozen_to_device(torch.device("cuda:0"))

    arch.init_training_schedulers(1000)

    # — Bug #3 spy: intercept skip_first kwarg via the public re-export
    #   that `base.py:prepare_inputs` resolves at call-time
    #   (`from openwam.utils import downsample_video_mask_to_latent`). —
    skip_first_calls = []
    real_fn = _utils_mod.downsample_video_mask_to_latent

    def _spy(*args, **kw):
        skip_first_calls.append(kw.get("skip_first", "MISSING"))
        return real_fn(*args, **kw)

    monkeypatch.setattr(_utils_mod, "downsample_video_mask_to_latent", _spy)

    sample = _make_stub_sample()
    sample["pre_encoded_text"] = torch.randn(512, 1024)
    # `first_frame_image` is auto-injected by `FirstFrameConditioningTransform`
    # at `base.py:493-500`; no need to set it on the sample explicitly.

    # Re-pin the post-projection 0-hit invariant on the real-VAE path.
    proj = arch.video_backbone._pipe.net.crossattn_proj
    hits = []
    handle = proj.register_forward_hook(lambda *_: hits.append(True))
    try:
        inputs = arch.prepare_inputs([sample])
        assert inputs.get("first_frame_latents") is not None, (
            "TI2V path expected to auto-populate first_frame_latents via "
            "FirstFrameConditioningTransform → wrapper preprocess_input."
        )
        assert inputs.get("num_clean_prefix_frames") == 1, (
            "Cosmos25 TI2V emits num_clean_prefix_frames=1 (one clean "
            "prefix latent = frame 0). Got "
            f"{inputs.get('num_clean_prefix_frames')!r}."
        )
        result = arch.compute_loss(**inputs)
    finally:
        handle.remove()

    loss = result["loss"]
    assert torch.isfinite(loss).item(), f"loss not finite on TI2V real-VAE path: {loss}"
    assert loss.requires_grad
    loss.backward()

    # — Bug #3 explicit assertion: TI2V batch ⇒ skip_first=True —
    assert any(c is True for c in skip_first_calls), (
        f"downsample_video_mask_to_latent never called with skip_first=True; "
        f"Cosmos TI2V path (first_frame_latents populated) is broken. "
        f"Calls observed: {skip_first_calls}. "
        f"See docs/cosmos25_backbone.md §12.3.3."
    )
    assert all(c is not False for c in skip_first_calls), (
        f"downsample_video_mask_to_latent called with skip_first=False in a "
        f"Cosmos TI2V batch; T2V leak. Calls: {skip_first_calls}."
    )

    # — Post-projection invariant still holds on the real-VAE path —
    assert hits == [], (
        f"crossattn_proj fired {len(hits)} time(s) on a (512, 1024) cache; "
        "post-projection cache must NOT re-trigger the 100352→1024 projection."
    )

    ab_has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in arch.action_backbone.parameters())
    assert ab_has_grad, "action_backbone got no grads on the TI2V real-VAE path"

    for n, p in arch.video_backbone.named_parameters():
        assert not p.requires_grad, f"video_backbone.{n} unexpectedly trainable"


# ----------------------------------------------------------------------
# §14 — live Reason1 text encoder, no offline cache.
# ----------------------------------------------------------------------


_REASON1_PATH = Path("/path/to/assets/Cosmos-Reason1-7B")


@pytest.mark.skipif(
    not _REASON1_PATH.exists(),
    reason="needs Cosmos-Reason1-7B weights on disk for live encoder smoke",
)
def test_cosmos25_train_smoke_ti2v_with_live_text_encoder(monkeypatch):
    'Public implementation.'
    _skip_unless_runnable()

    from openwam.model import build_architecture, resolve_architecture_config

    cfg = _load_cfg()
    cfg.video_backbone.text_encoder = "reason1_live"
    cfg.video_backbone.text_encoder_path = str(_REASON1_PATH)
    resolved = resolve_architecture_config(cfg)
    arch = build_architecture(resolved.registry_name, resolved.params)
    arch.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
    arch.move_frozen_to_device(torch.device("cuda:0"))
    arch.init_training_schedulers(1000)

    # Real video frames so the Wan2pt1 VAE encode path runs end-to-end
    # alongside the live text encoder.
    sample = _make_stub_sample()
    # Crucially: NO `pre_encoded_text` — that would short-circuit the live
    # encoder per the documented cache > live precedence.
    sample["prompt"] = "pick up the block"

    # Hook the upstream `crossattn_proj` Sequential — pre-projection (100352)
    # must trigger exactly one projection call per preprocess_input. This is
    # the inverse of the cached-path 0-hit invariant.
    proj = arch.video_backbone._pipe.net.crossattn_proj
    hits = []

    def _spy(_mod, _inp, _out):
        hits.append(True)

    handle = proj.register_forward_hook(_spy)
    try:
        inputs = arch.prepare_inputs([sample])
        # `preprocess_input` applies `net.crossattn_proj` in-line after the
        # live encoder (so `_append_proprio_context_token` in
        # `base.py:_append_proprio_context_token` can concat the proprio
        # token at the matched 1024 dim). The cache test's mirror is
        # `== 1024` for the same reason — both paths leave preprocess_input
        # with a post-projection context.
        assert inputs["context"].shape[-1] == 1024, (
            f"context shape after live encoder + in-line projection: "
            f"{inputs['context'].shape} — expected last dim 1024."
        )
        result = arch.compute_loss(**inputs)
    finally:
        handle.remove()

    assert len(hits) == 1, (
        f"net.crossattn_proj fired {len(hits)} time(s); expected exactly 1. "
        "Live encoder returns pre-projection (100352); `preprocess_input` "
        "applies `net.crossattn_proj` once to drop to 1024 before the "
        "context flows on. 0 = live encoder dispatch missed; >1 = the "
        "`prepare_block_loop` auto-gate (line 174-177) double-projected, "
        "which would mean the in-line projection produced 100352 again."
    )

    loss = result["loss"]
    assert torch.isfinite(loss).item(), f"loss not finite on live encoder path: {loss}"
    assert loss.requires_grad
    loss.backward()

    ab_has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in arch.action_backbone.parameters())
    assert ab_has_grad, "action_backbone got no grads on the live encoder path"

    for n, p in arch.video_backbone.named_parameters():
        assert not p.requires_grad, f"video_backbone.{n} unexpectedly trainable on live encoder path"


@pytest.mark.skipif(
    not _REASON1_PATH.exists(),
    reason="needs Cosmos-Reason1-7B weights on disk for live encoder smoke",
)
def test_cosmos25_train_smoke_ti2v_live_encoder_with_dropout(monkeypatch):
    """§14.7 regression: ``text_encoder=reason1_live`` + ``text_encoder_dropout=1.0``
    must (a) still produce a finite, differentiable loss end-to-end, and
    (b) leave the ``crossattn_proj`` hit count at exactly 1 — dropout swaps
    prompt text for ``""`` before the encoder call, but the encoder still
    returns pre-projection ``(B, 512, 100352)`` so the wrapper's in-line
    ``net.crossattn_proj`` fires once just like the no-dropout path.

    p=1.0 makes the substitution deterministic (no flaky single-sample
    probabilistic assertion). We additionally spy on the live encoder to
    confirm the substitution happened — guards against a regression that
    silently bypasses dropout (e.g. dropping the ``self.training`` gate)."""
    _skip_unless_runnable()

    from openwam.model import build_architecture, resolve_architecture_config

    cfg = _load_cfg()
    cfg.video_backbone.text_encoder = "reason1_live"
    cfg.video_backbone.text_encoder_path = str(_REASON1_PATH)
    cfg.video_backbone.text_encoder_dropout = 1.0
    cfg.video_backbone.text_encoder_dropout_seed = 0
    resolved = resolve_architecture_config(cfg)
    arch = build_architecture(resolved.registry_name, resolved.params)
    arch.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
    arch.move_frozen_to_device(torch.device("cuda:0"))
    arch.init_training_schedulers(1000)

    sample = _make_stub_sample()
    sample["prompt"] = "pick up the block"  # real prompt — dropout must replace it

    wrapper = arch.video_backbone._pipe
    proj = wrapper.net.crossattn_proj
    hits = []

    def _spy(_mod, _inp, _out):
        hits.append(True)

    real_encoder = wrapper.text_encoder
    seen_prompts = []

    class _SpyEncoder:
        """Wraps the real Reason1LiveTextEncoder to record its inputs.

        Patching ``encoder.__call__`` on the instance does not work because
        Python looks up ``__call__`` on the class; we have to replace the
        attribute the wrapper holds with a callable that delegates."""

        def __call__(self, prompts):
            seen_prompts.append(list(prompts) if not isinstance(prompts, str) else [prompts])
            return real_encoder(prompts)

    monkeypatch.setattr(wrapper, "text_encoder", _SpyEncoder())
    handle = proj.register_forward_hook(_spy)
    try:
        inputs = arch.prepare_inputs([sample])
        assert inputs["context"].shape[-1] == 1024, (
            f"context last dim must be 1024 after in-line crossattn_proj; got {inputs['context'].shape}"
        )
        result = arch.compute_loss(**inputs)
    finally:
        handle.remove()

    # p=1.0 ⇒ every prompt substituted with "". Guards the dropout substitution
    # path from being silently bypassed by a future refactor.
    assert seen_prompts == [[""]], (
        f"text_encoder_dropout=1.0 + training must substitute '' for every prompt; encoder saw {seen_prompts}"
    )
    assert len(hits) == 1, (
        f"net.crossattn_proj fired {len(hits)} time(s); expected exactly 1 "
        "(dropout swaps the prompt but the encoder still returns "
        "pre-projection 100352, so the in-line projection still runs)."
    )

    loss = result["loss"]
    assert torch.isfinite(loss).item(), f"loss not finite under CFG dropout: {loss}"
    assert loss.requires_grad
    loss.backward()

    ab_has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in arch.action_backbone.parameters())
    assert ab_has_grad, "action_backbone got no grads with CFG dropout enabled"


# ----------------------------------------------------------------------
# §17 — joint self-attention end-to-end (MoTJointDriver + Cosmos25 split)
# ----------------------------------------------------------------------


def test_cosmos25_train_smoke_self_attn_ti2v(monkeypatch):
    """§17 + §18 regression: ``dual_system_self_attn_cosmos25.yaml`` runs
    one full forward+backward through ``MoTJointDriver`` against the real
    28-block Cosmos25-2B network, with TI2V first-frame conditioning.

    Locks in the contract that:

    - ``Cosmos25VideoBackbone.{pre,post}_attn_at_layer`` are dispatched by the
      driver for every layer (no ``NotImplementedError`` regression).
    - The driver's ``s_video = f * h * w`` derivation works on Cosmos's 5D
      ``state.x`` (was previously ``state.x.shape[1]`` = T — wrong for
      Cosmos).
    - Action backbone receives gradients; Cosmos25 stays frozen.
    - The joint mask uses ``first_frame_causal`` (yaml default) — exercise
      ``build_video_to_video_mask`` end-to-end.
    - TI2V keys (``first_frame_latents`` + ``num_clean_prefix_frames=1``)
      flow through the joint-attn path just like the cross-attn path; the
      loss skips the clean prefix frame; ``downsample_video_mask_to_latent``
      is called with ``skip_first=True``.
    """
    _skip_unless_runnable()
    if not COSMOS_SELF_ATTN_CFG_PATH.exists():
        pytest.skip(f"self-attn config missing at {COSMOS_SELF_ATTN_CFG_PATH}")

    from omegaconf import OmegaConf

    import openwam.utils as _utils_mod
    from openwam.model import build_architecture, resolve_architecture_config

    cfg = OmegaConf.load(str(COSMOS_SELF_ATTN_CFG_PATH))
    cfg.video_backbone.model_path = str(ASSET_PATH)
    resolved = resolve_architecture_config(cfg)
    arch = build_architecture(resolved.registry_name, resolved.params)

    # MoT driver must exist after construction (yaml has both backbones).
    assert arch.mot_driver is not None, "DualSystemSelfAttnArchitecture.mot_driver should be wired in __init__"
    # Per-layer parity is required by the driver — Cosmos25-2B is 28 blocks.
    assert arch.video_backbone.num_layers == arch.action_backbone.num_layers == 28

    arch.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
    arch.move_frozen_to_device(torch.device("cuda:0"))
    arch.init_training_schedulers(1000)

    # TI2V invariant: skip_first=True because first_frame_latents is populated.
    skip_first_calls = []
    real_fn = _utils_mod.downsample_video_mask_to_latent
    monkeypatch.setattr(
        _utils_mod,
        "downsample_video_mask_to_latent",
        lambda *a, **kw: (skip_first_calls.append(kw.get("skip_first", "MISSING")), real_fn(*a, **kw))[1],
    )

    sample = _make_stub_sample()
    sample["pre_encoded_text"] = torch.randn(512, 1024)

    inputs = arch.prepare_inputs([sample])
    assert inputs.get("first_frame_latents") is not None, (
        "Joint self-attn TI2V path expected to auto-populate first_frame_latents "
        "via FirstFrameConditioningTransform → wrapper preprocess_input."
    )
    assert inputs.get("num_clean_prefix_frames") == 1, (
        f"Expected num_clean_prefix_frames=1 on TI2V joint self-attn path, "
        f"got {inputs.get('num_clean_prefix_frames')!r}."
    )
    result = arch.compute_loss(**inputs)

    loss = result["loss"]
    assert torch.isfinite(loss).item(), f"joint self-attn TI2V loss not finite: {loss}"
    assert loss.requires_grad
    loss.backward()

    # ActionDiT must receive gradients through the mixed attention. Cosmos
    # stays frozen.
    ab_has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in arch.action_backbone.parameters())
    assert ab_has_grad, "action_backbone got no grads through MoT joint attention"
    for n, p in arch.video_backbone.named_parameters():
        assert not p.requires_grad, f"video_backbone.{n} unexpectedly trainable on joint self-attn path"

    # TI2V invariant: skip_first=True (first_frame_latents populated).
    assert any(c is True for c in skip_first_calls), (
        f"downsample_video_mask_to_latent never called with skip_first=True on TI2V joint self-attn path; "
        f"calls: {skip_first_calls}."
    )
    assert all(c is not False for c in skip_first_calls), (
        f"skip_first=False leaked into a TI2V batch on joint self-attn path; calls: {skip_first_calls}."
    )
