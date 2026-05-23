"""CPU unit tests for Cosmos25 deploy-artifact contract (Fix #2 v2).

Cosmos25 follows the same convention as dual_system / shared_backbone for
the weights it owns: **everything goes into the unified safetensors**, no
external file copy. Reviewer @d-finite's original complaint
(`tokenizer.pth` unreachable on a deploy host without `/path/to`) is
addressed by registering the upstream `Wan2pt1VAEInterface`'s inner
`WanVAE_` nn.Module as a child of `Cosmos25PipelineWrapper` so its params
flow through `state_dict()`.

These tests pin three guarantees:

1. `Cosmos25PipelineWrapper.__init__` registers `vae.model.model` under
   ``_vae_inner`` whenever a VAE is supplied, so its params join the
   wrapper's `state_dict()`.
2. The architecture's full save → load roundtrip restores those weights
   bit-for-bit even when the second wrapper is built with an empty VAE
   shell (mimicking the deploy path where `tokenizer.pth` is absent).
3. `generate_cosmos25_component_specs` emits a non-empty marker (gating
   ``deploy/model_loader.py:117-122``'s ``_ckpt_dir`` injection) and
   ``copy_cosmos25_artifacts`` copies Reason1 structural JSONs.

The Reason1 text encoder (~16 GB) is also registered under
``_reason1_inner`` so cache-mode and live-mode checkpoints both carry it in
the unified safetensors.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from openwam.model.video_backbone.cosmos25.component_specs import (
    _resolve_text_encoder_path,
    copy_cosmos25_artifacts,
    generate_cosmos25_component_specs,
)
from openwam.model.video_backbone.cosmos25.pipeline_wrapper import Cosmos25PipelineWrapper

# ----------------------------------------------------------------------
# Fake Wan2pt1VAEInterface mirrors the shape we register in __init__.
# Matches the structure used by `tests/test_cosmos25_vae_freeze_and_dtype_plumbing.py`.
# ----------------------------------------------------------------------


class _FakeWanVAE:
    """Mimics ``Wan2pt1VAEInterface.model`` (the inner ``WanVAE`` wrapper)."""

    def __init__(self) -> None:
        # ``model.model`` is the actual nn.Module — what we want in state_dict.
        # Keep it small (a Linear) for fast CPU tests.
        self.model: nn.Module = nn.Linear(4, 4)
        # Six mean/std tensors live on the outer ``WanVAE`` as plain attrs.
        # They are *not* registered in state_dict (constants / placeholders),
        # so we don't need state_dict roundtrip semantics for them — but
        # downstream code (`_move_cosmos_vae`, upstream encode/decode) still
        # reads them, so the fixture has to expose them.
        self.mean = torch.zeros(16)
        self.std = torch.ones(16)
        self.img_mean = torch.zeros(1, 16, 1, 1, 1)
        self.img_std = torch.ones(1, 16, 1, 1, 1)
        self.video_mean = torch.zeros(1, 16, 9, 1, 1)
        self.video_std = torch.ones(1, 16, 9, 1, 1)
        self.device = torch.device("cpu")
        self.dtype = torch.float32


class _FakeWan2pt1Interface:
    def __init__(self) -> None:
        self.model = _FakeWanVAE()


class _ParamNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))


def _make_wrapper(vae) -> Cosmos25PipelineWrapper:
    return Cosmos25PipelineWrapper(
        net=_ParamNet(),
        vae=vae,
        text_encoder=None,
        dim=2048,
        num_layers=28,
        num_heads=16,
        head_dim=128,
        context_dim=1024,
        flow_shift=5.0,
    )


class _FakeReason1:
    def __init__(self) -> None:
        self.model = nn.Linear(3, 3)


# ----------------------------------------------------------------------
# (1) Registration
# ----------------------------------------------------------------------


def test_pipeline_wrapper_registers_vae_inner_module():
    """``__setattr__`` of an nn.Module attribute adds the inner ``WanVAE_``
    under ``_modules['_vae_inner']``, so its params join ``state_dict``."""
    iface = _FakeWan2pt1Interface()
    wrapper = _make_wrapper(iface)

    # Reference identity preserved — upstream call sites that read
    # ``iface.model.model.encode(...)`` keep working bit-for-bit.
    assert wrapper._vae_inner is iface.model.model

    # `_vae_inner` appears in `_modules`, not just `__dict__`.
    assert "_vae_inner" in wrapper._modules
    assert wrapper._modules["_vae_inner"] is iface.model.model

    # ``vae`` itself is still reachable as a plain attribute (upstream
    # interface methods like ``vae.encode`` need it).
    assert wrapper.vae is iface


def test_pipeline_wrapper_state_dict_contains_vae_inner_params():
    """All VAE inner-module params appear in ``state_dict`` under the
    ``_vae_inner.*`` prefix, paving the way for them to flow into the
    unified safetensors that ``BaseWAMArchitecture.save_checkpoint`` writes."""
    iface = _FakeWan2pt1Interface()
    wrapper = _make_wrapper(iface)

    keys = list(wrapper.state_dict().keys())
    vae_keys = [k for k in keys if k.startswith("_vae_inner.")]
    assert vae_keys, (
        f"_vae_inner.* keys missing from state_dict. All keys: {keys}"
    )
    # The fake VAE is `nn.Linear(4, 4)` → weight + bias.
    assert "_vae_inner.weight" in vae_keys
    assert "_vae_inner.bias" in vae_keys


def test_pipeline_wrapper_no_vae_attached_when_none():
    """``vae=None`` (e.g. pre-encoded latents path) leaves ``_vae_inner`` unset
    so ``state_dict`` has no stale VAE keys."""
    wrapper = _make_wrapper(vae=None)
    assert "_vae_inner" not in wrapper._modules
    assert not any(k.startswith("_vae_inner") for k in wrapper.state_dict().keys())


# ----------------------------------------------------------------------
# (2) Round-trip: train-time wrapper → state_dict → deploy-time empty shell
# ----------------------------------------------------------------------


def test_state_dict_roundtrip_loads_vae_weights_into_empty_shell():
    """Pin the deploy contract:

    1. Train-time wrapper is built with real VAE weights.
    2. ``state_dict()`` is captured and used to populate a fresh wrapper
       whose VAE was built with random weights (the cosmos25 equivalent of
       ``vae_pth=None`` empty shell).
    3. The two wrappers produce identical VAE inner outputs and the
       restored ``state_dict()`` matches the original entry for entry.
    """
    # Train wrapper: deterministic weights.
    train_iface = _FakeWan2pt1Interface()
    with torch.no_grad():
        train_iface.model.model.weight.copy_(torch.eye(4))
        train_iface.model.model.bias.copy_(torch.linspace(-1.0, 1.0, 4))
    train_wrapper = _make_wrapper(train_iface)

    # Capture state and the canonical forward output.
    saved_sd = {k: v.clone() for k, v in train_wrapper.state_dict().items()}
    probe = torch.randn(1, 4)
    train_out = train_iface.model.model(probe)

    # Deploy wrapper: empty shell semantics — VAE is freshly initialised
    # with random params, not loaded from any file. Architecture-level
    # ``load_state_dict`` is what should bring it back to the train state.
    deploy_iface = _FakeWan2pt1Interface()  # default init → random nn.Linear weights
    deploy_wrapper = _make_wrapper(deploy_iface)

    missing, unexpected = deploy_wrapper.load_state_dict(saved_sd, strict=True)
    assert not missing and not unexpected

    # The reloaded inner module produces the train-time output.
    deploy_out = deploy_iface.model.model(probe)
    torch.testing.assert_close(deploy_out, train_out)

    # And every saved tensor is byte-for-byte restored.
    restored_sd = deploy_wrapper.state_dict()
    for k, v in saved_sd.items():
        torch.testing.assert_close(restored_sd[k], v)


# ----------------------------------------------------------------------
# (3) component_specs: deploy gate marker + no-op artifact copy
# ----------------------------------------------------------------------


def test_generate_cosmos25_component_specs_emits_marker_when_model_path_valid(tmp_path):
    """A readable ``model_path`` yields a non-None spec — this is the gate
    that makes ``deploy/model_loader.py`` thread ``_ckpt_dir`` into the
    adapter, which in turn flips ``build_cosmos25_pipeline`` into empty-shell
    deploy mode. The spec content documents that the VAE lives in state_dict."""
    spec = generate_cosmos25_component_specs(str(tmp_path))
    assert spec is not None
    assert "components" in spec
    assert spec["components"], "components list must be non-empty to trigger the deploy gate"
    # The marker entry documents the deploy contract.
    vae_entry = next((c for c in spec["components"] if c.get("attr") == "vae"), None)
    assert vae_entry is not None
    assert vae_entry["source"] == "state_dict"
    text_entry = next((c for c in spec["components"] if c.get("attr") == "text_encoder"), None)
    assert text_entry is not None
    assert text_entry["source"] == "state_dict"


def test_pipeline_wrapper_state_dict_contains_reason1_even_when_cache_wins():
    """Cache training still needs Reason1 registered so saves are deploy self-contained."""
    wrapper = Cosmos25PipelineWrapper(
        net=_ParamNet(),
        vae=None,
        text_encoder=_FakeReason1(),
        dim=2048,
        num_layers=28,
        num_heads=16,
        head_dim=128,
        context_dim=1024,
        flow_shift=5.0,
    )

    state_keys = list(wrapper.state_dict().keys())
    assert any(k.startswith("_reason1_inner.") for k in state_keys)


def test_generate_cosmos25_component_specs_returns_none_for_missing_path():
    """An unreadable ``model_path`` bypasses the deploy gate so fake-pipeline
    tests and offline build paths keep the legacy ``_source`` plumbing."""
    assert generate_cosmos25_component_specs("/nonexistent/path/cosmos25") is None
    assert generate_cosmos25_component_specs("") is None
    assert generate_cosmos25_component_specs(None) is None  # type: ignore[arg-type]


def test_resolve_text_encoder_path_keeps_empty_string_missing():
    assert _resolve_text_encoder_path("") is None


def test_copy_cosmos25_artifacts_requires_reason1_path(tmp_path):
    """Reason1 weights are in safetensors, but structural JSONs must be copied."""
    dst = tmp_path / "ckpt"
    dst.mkdir()
    with pytest.raises(RuntimeError, match="Reason1 artifact source"):
        copy_cosmos25_artifacts(str(dst), "/anything/at/all")


def test_copy_cosmos25_artifacts_copies_reason1_structural_files(tmp_path):
    src = tmp_path / "reason1_src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "tokenizer.json").write_text("{}")
    dst = tmp_path / "ckpt"
    dst.mkdir()

    copy_cosmos25_artifacts(str(dst), str(src))

    assert (dst / "reason1" / "config.json").is_file()
    assert (dst / "reason1" / "tokenizer.json").is_file()


# ----------------------------------------------------------------------
# (4) deploy/model_loader.py — components-marker detection survives the
# OmegaConf ListConfig/DictConfig wrapping that a real saved config has.
# ----------------------------------------------------------------------


def test_model_loader_detects_reason1_state_component_through_omegaconf(tmp_path, monkeypatch):
    """Saved ``config.yaml``s come back as ``ListConfig`` of ``DictConfig``;
    iterating with ``isinstance(c, dict)`` against ``DictConfig`` would
    silently fail (``DictConfig`` is not a ``dict`` subclass). This test
    pins that the self-contained Reason1 fallback fires when

      - ``text_encoder`` is **not** ``reason1_live`` in the saved config
        (e.g. cache-mode training left it as ``none``), AND
      - ``components`` contains the ``attr: text_encoder, source: state_dict``
        marker that ``generate_cosmos25_component_specs`` emits, AND
      - ``<ckpt_dir>/reason1/`` artifact dir exists.

    Under those conditions deploy must clear ``text_encoder_path`` and
    flip ``text_encoder=reason1_live`` so the empty-shell deploy path
    picks up the in-state-dict Reason1 weights.
    """
    from unittest.mock import MagicMock, patch

    from omegaconf import OmegaConf

    # Build a real saved-style config (ListConfig of DictConfigs).
    saved_cfg = OmegaConf.create(
        {
            "model": {
                "framework": "wam",
                "variant": "shared_backbone_vanilla",
                "video_backbone": {
                    "name": "cosmos25_5b",
                    "text_encoder": "none",
                    "text_encoder_path": "/path/to/model",
                    "components": [
                        {"attr": "text_encoder", "source": "state_dict"},
                        {"attr": "vae", "source": "state_dict"},
                    ],
                },
            },
        }
    )
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.yaml").write_text(OmegaConf.to_yaml(saved_cfg))
    # The fallback only fires when the reason1 artifact dir exists.
    (ckpt_dir / "reason1").mkdir()
    (ckpt_dir / "checkpoint_step_1.safetensors").write_bytes(b"")

    captured: dict = {}

    def _fake_build_architecture(registry_name, params):
        captured["params"] = params
        arch = MagicMock()
        arch.load_checkpoint = MagicMock()
        arch.set_dtype_device = MagicMock()
        arch.eval = MagicMock()
        arch.attach_action_normalizer = MagicMock()
        return arch

    resolved = MagicMock()
    resolved.registry_name = "shared_backbone_vanilla"
    resolved.canonical.framework = "wam"
    resolved.canonical.variant = "shared_backbone_vanilla"
    resolved.params = {"video_backbone": dict(saved_cfg.model.video_backbone)}

    from openwam.deploy import model_loader

    with (
        patch.object(model_loader, "build_architecture", _fake_build_architecture, create=True),
        patch.object(model_loader, "resolve_architecture_config", lambda _m: resolved, create=True),
        patch.object(model_loader, "_build_action_normalizer", lambda *_a, **_kw: None),
    ):
        # Patch the deferred imports inside load_from_checkpoint_dir.
        import openwam.model as _openwam_model

        monkeypatch.setattr(_openwam_model, "build_architecture", _fake_build_architecture, raising=True)
        monkeypatch.setattr(
            _openwam_model, "resolve_architecture_config", lambda _m: resolved, raising=True
        )
        model_loader.load_from_checkpoint_dir(str(ckpt_dir), device="cpu")

    source = captured["params"]["video_backbone"]["_source"]
    assert isinstance(source, dict), f"Expected plain dict source, got {type(source).__name__}"
    assert source["text_encoder"] == "reason1_live", (
        "Reason1 self-contained marker (components entry with attr=text_encoder) was not "
        "detected — deploy will leave text_encoder=none and the empty-shell branch never fires. "
        "Likely a DictConfig vs dict regression in model_loader.py:132."
    )
    assert source["text_encoder_path"] is None, (
        "External text_encoder_path must be cleared once self-contained Reason1 weights are detected."
    )
