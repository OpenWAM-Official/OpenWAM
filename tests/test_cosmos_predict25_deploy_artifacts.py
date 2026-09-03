"""CPU unit tests for CosmosPredict25 deploy-artifact contract (Fix #2 v2).

CosmosPredict25 follows the same convention as dual_system / single_system for
the weights it owns: **everything goes into the unified safetensors**, no
external file copy. Reviewer @d-finite's original complaint
(`tokenizer.pth` unreachable on a deploy host without `/path/to`) is
addressed by registering the upstream `Wan2pt1VAEInterface`'s inner
`WanVAE_` nn.Module as a child of `CosmosPredict25VideoBackbone` so its params
flow through `state_dict()`.

These tests pin three guarantees:

1. `CosmosPredict25VideoBackbone.__init__` registers `vae.model.model` under
   ``vae`` whenever a VAE is supplied, so its params join the
   wrapper's `state_dict()`.
2. The architecture's full save → load roundtrip restores those weights
   bit-for-bit even when the second wrapper is built with an empty VAE
   shell (mimicking the deploy path where `tokenizer.pth` is absent).
3. `generate_cosmos_predict25_component_specs` emits a non-empty marker (gating
   ``deploy/model_loader.py:117-122``'s ``_ckpt_dir`` injection) and
   ``copy_cosmos_predict25_artifacts`` copies Reason1 structural JSONs.

The Reason1 text encoder (~16 GB) is also registered under ``reason1`` so
checkpoints carry it in the unified safetensors.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from openwam.model.video_backbone.cosmos_predict25.component_specs import (
    _resolve_text_encoder_path,
    copy_cosmos_predict25_artifacts,
    generate_cosmos_predict25_component_specs,
)
from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

# ----------------------------------------------------------------------
# Fake Wan2pt1VAEInterface mirrors the shape we register in __init__.
# Matches the structure used by `tests/test_cosmos_predict25_vae_freeze_and_dtype_plumbing.py`.
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


def _make_wrapper(vae) -> CosmosPredict25VideoBackbone:
    return CosmosPredict25VideoBackbone(
        net=_ParamNet(),
        vae=vae,
        text_encoder=None,
        dim=2048,
        num_layers=28,
        num_heads=16,
        head_dim=128,
        context_dim=1024,
        shift_video=5.0,
    )


class _FakeReason1:
    def __init__(self) -> None:
        self.model = nn.Linear(3, 3)


# ----------------------------------------------------------------------
# (1) Registration
# ----------------------------------------------------------------------


def test_pipeline_wrapper_registersvae_module():
    """``__setattr__`` of an nn.Module attribute adds the inner ``WanVAE_``
    under ``_modules['vae']``, so its params join ``state_dict``."""
    iface = _FakeWan2pt1Interface()
    wrapper = _make_wrapper(iface)

    # Reference identity preserved — upstream call sites that read
    # ``iface.model.model.encode(...)`` keep working bit-for-bit.
    assert wrapper.vae is iface.model.model

    # `vae` appears in `_modules`, not just `__dict__`.
    assert "vae" in wrapper._modules
    assert wrapper._modules["vae"] is iface.model.model

    # The facade itself is still reachable as a plain attribute (upstream
    # interface methods like ``_vae_iface.encode`` need it).
    assert wrapper._vae_iface is iface


def test_pipeline_wrapper_state_dict_containsvae_params():
    """All VAE inner-module params appear in ``state_dict`` under the
    ``vae.*`` prefix, paving the way for them to flow into the
    unified safetensors that ``BaseWAMArchitecture.save_checkpoint`` writes."""
    iface = _FakeWan2pt1Interface()
    wrapper = _make_wrapper(iface)

    keys = list(wrapper.state_dict().keys())
    vae_keys = [k for k in keys if k.startswith("vae.")]
    assert vae_keys, f"vae.* keys missing from state_dict. All keys: {keys}"
    # The fake VAE is `nn.Linear(4, 4)` → weight + bias.
    assert "vae.weight" in vae_keys
    assert "vae.bias" in vae_keys


def test_pipeline_wrapper_no_vae_attached_when_none():
    """``vae=None`` (e.g. pre-encoded latents path) leaves ``vae`` unset
    so ``state_dict`` has no stale VAE keys."""
    wrapper = _make_wrapper(vae=None)
    assert "vae" not in wrapper._modules
    assert not any(k.startswith("vae") for k in wrapper.state_dict().keys())


# ----------------------------------------------------------------------
# (2) Round-trip: train-time wrapper → state_dict → deploy-time empty shell
# ----------------------------------------------------------------------


def test_state_dict_roundtrip_loads_vae_weights_into_empty_shell():
    """Pin the deploy contract:

    1. Train-time wrapper is built with real VAE weights.
    2. ``state_dict()`` is captured and used to populate a fresh wrapper
       whose VAE was built with random weights (the cosmos_predict25 equivalent of
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


def test_generate_cosmos_predict25_component_specs_emits_marker_when_model_path_valid(tmp_path):
    """A readable ``model_path`` yields a non-None spec — this is the gate
    that makes ``deploy/model_loader.py`` thread ``_ckpt_dir`` into the
    adapter, which in turn flips ``build_cosmos_predict25_pipeline`` into empty-shell
    deploy mode. The spec content documents that the VAE lives in state_dict."""
    spec = generate_cosmos_predict25_component_specs(str(tmp_path))
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


def test_pipeline_wrapper_state_dict_contains_reason1():
    """Reason1 must be registered so saves are deploy self-contained."""
    wrapper = CosmosPredict25VideoBackbone(
        net=_ParamNet(),
        vae=None,
        text_encoder=_FakeReason1(),
        dim=2048,
        num_layers=28,
        num_heads=16,
        head_dim=128,
        context_dim=1024,
        shift_video=5.0,
    )

    state_keys = list(wrapper.state_dict().keys())
    assert any(k.startswith("reason1.") for k in state_keys)


def test_generate_cosmos_predict25_component_specs_returns_none_for_missing_path():
    """An unreadable ``model_path`` bypasses the deploy gate so fake-pipeline
    tests and offline build paths keep the legacy ``_source`` plumbing."""
    assert generate_cosmos_predict25_component_specs("/nonexistent/path/cosmos_predict25") is None
    assert generate_cosmos_predict25_component_specs("") is None
    assert generate_cosmos_predict25_component_specs(None) is None  # type: ignore[arg-type]


def test_resolve_text_encoder_path_keeps_empty_string_missing():
    assert _resolve_text_encoder_path("") is None


def test_copy_cosmos_predict25_artifacts_requires_reason1_path(tmp_path):
    """Reason1 weights are in safetensors, but structural JSONs must be copied."""
    dst = tmp_path / "ckpt"
    dst.mkdir()
    with pytest.raises(RuntimeError, match="Reason1 artifact source"):
        copy_cosmos_predict25_artifacts(str(dst), "/anything/at/all")


def test_copy_cosmos_predict25_artifacts_copies_reason1_structural_files(tmp_path):
    src = tmp_path / "reason1_src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "tokenizer.json").write_text("{}")
    dst = tmp_path / "ckpt"
    dst.mkdir()

    copy_cosmos_predict25_artifacts(str(dst), str(src))

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
    pins that the self-contained Reason1 branch fires when ``components``
    contains the ``attr: text_encoder, source: state_dict`` marker that
    ``generate_cosmos_predict25_component_specs`` emits: deploy must clear
    ``text_encoder_path`` so the empty-shell path picks up the in-state-dict
    Reason1 weights.
    """
    from unittest.mock import MagicMock, patch

    from omegaconf import OmegaConf

    # Build a real saved-style config (ListConfig of DictConfigs).
    saved_cfg = OmegaConf.create(
        {
            "model": {
                "framework": "wam",
                "variant": "single_system_vanilla",
                "video_backbone": {
                    "name": "cosmos_predict25_5b",
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
    (ckpt_dir / "reason1").mkdir()
    (ckpt_dir / "checkpoint_step_1.safetensors").write_bytes(b"")

    captured: dict = {}

    def _fake_build_architecture(registry_name, params):
        captured["params"] = params
        arch = MagicMock()
        arch.load_checkpoint = MagicMock()
        arch.set_dtype_device = MagicMock()
        arch.eval = MagicMock()
        arch.attach_normalizer = MagicMock()
        return arch

    resolved = MagicMock()
    resolved.registry_name = "single_system_vanilla"
    resolved.canonical.framework = "wam"
    resolved.canonical.variant = "single_system_vanilla"
    resolved.params = {"video_backbone": dict(saved_cfg.model.video_backbone)}

    from openwam.deploy import model_loader

    with (
        patch.object(model_loader, "build_architecture", _fake_build_architecture, create=True),
        patch.object(model_loader, "resolve_architecture_config", lambda _m: resolved, create=True),
        patch.object(model_loader, "_build_normalizer", lambda *_a, **_kw: None),
    ):
        # Patch the deferred imports inside load_from_checkpoint_dir.
        import openwam.model as _openwam_model

        monkeypatch.setattr(_openwam_model, "build_architecture", _fake_build_architecture, raising=True)
        monkeypatch.setattr(_openwam_model, "resolve_architecture_config", lambda _m: resolved, raising=True)
        model_loader.load_from_checkpoint_dir(str(ckpt_dir), device="cpu")

    source = captured["params"]["video_backbone"]["_source"]
    assert isinstance(source, dict), f"Expected plain dict source, got {type(source).__name__}"
    assert source["text_encoder_path"] is None, (
        "External text_encoder_path must be cleared once self-contained Reason1 weights are detected "
        "(components entry with attr=text_encoder + <ckpt_dir>/reason1/); otherwise the empty-shell "
        "deploy branch never fires. Likely a DictConfig vs dict regression in model_loader.py."
    )


def _backbone_with_reason1(has_reason1: bool, has_vae: bool = True):
    """A fake CosmosPredict25VideoBackbone reporting Reason1 / VAE presence.

    ``save_deploy_assets`` reads ``self.text_encoder`` / ``self.vae`` as the
    ground truth for whether those weights are in the checkpoint. Defaults to a
    configured VAE (the normal ``vae: wan2pt1`` case); pass ``has_vae=False`` to
    model a ``vae: none`` training run.
    """

    from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

    bb = CosmosPredict25VideoBackbone.__new__(CosmosPredict25VideoBackbone)
    bb.text_encoder = object() if has_reason1 else None
    # ``save_deploy_assets`` gates the vae component on the registered ``vae``
    # child (present iff its weights are in the state_dict). Mirror the real
    # class: leave the attr absent for ``vae: none``. Use a plain ``object()``
    # sentinel (not an nn.Module) so the assignment doesn't trip the "assign
    # module before Module.__init__()" guard on this ``__new__``'d shell.
    if has_vae:
        bb.vae = object()  # type: ignore[assignment]
    return bb


def test_save_deploy_assets_merges_components_and_copies_reason1(tmp_path):
    """With a live Reason1 encoder, ``save_deploy_assets`` does both halves:
    merge the (vae + text_encoder) component specs into cfg + copy the JSONs."""
    from omegaconf import OmegaConf

    # Readable model_path (gates the spec emit) + a text_encoder_path with the
    # Reason1 structural files to copy.
    model_path = tmp_path / "cosmos_bundle"
    model_path.mkdir()
    te_path = tmp_path / "reason1_src"
    te_path.mkdir()
    for name in ("config.json", "tokenizer.json"):
        (te_path / name).write_text("{}")
    output_dir = tmp_path / "ckpt_out"
    output_dir.mkdir()

    cfg = OmegaConf.create(
        {
            "model": {
                "video_backbone": {
                    "model_path": str(model_path),
                    "text_encoder_path": str(te_path),
                }
            }
        }
    )

    _backbone_with_reason1(True).save_deploy_assets(str(output_dir), cfg)

    # 1. components merged into cfg with the two state_dict sub_module markers.
    comps = OmegaConf.to_container(cfg.model.video_backbone.components, resolve=True)
    attrs = {c["attr"]: c for c in comps}
    assert attrs["vae"]["sub_module"] == "vae"
    assert attrs["text_encoder"]["sub_module"] == "reason1"

    # 2. Reason1 structural JSONs copied next to the checkpoint.
    assert (output_dir / "reason1" / "config.json").is_file()
    assert (output_dir / "reason1" / "tokenizer.json").is_file()


def test_save_deploy_assets_without_reason1_emits_vae_only_and_no_copy(tmp_path):
    """A backbone without a Reason1 encoder must NOT emit the text_encoder
    marker, must NOT copy artifacts, and must NOT crash. Only the VAE
    component is recorded."""
    from omegaconf import OmegaConf

    model_path = tmp_path / "cosmos_bundle"
    model_path.mkdir()
    output_dir = tmp_path / "ckpt_out"
    output_dir.mkdir()

    cfg = OmegaConf.create(
        {
            "model": {
                "video_backbone": {
                    "model_path": str(model_path),
                    "text_encoder_path": None,
                }
            }
        }
    )

    _backbone_with_reason1(False).save_deploy_assets(str(output_dir), cfg)

    comps = OmegaConf.to_container(cfg.model.video_backbone.components, resolve=True)
    attrs = {c["attr"] for c in comps}
    assert attrs == {"vae"}  # text_encoder marker NOT emitted
    assert not (output_dir / "reason1").exists()  # nothing copied


def test_save_deploy_assets_vae_none_omits_vae_component(tmp_path):
    """Regression for the VAE gate: a ``vae: none`` run (``self.vae is None``,
    so ``vae`` is never registered) must NOT emit the vae component —
    otherwise the saved config references a sub_module absent from the
    state_dict. With no reason1 and no vae, no components remain."""
    from omegaconf import OmegaConf

    model_path = tmp_path / "cosmos_bundle"
    model_path.mkdir()
    output_dir = tmp_path / "ckpt_out"
    output_dir.mkdir()

    cfg = OmegaConf.create({"model": {"video_backbone": {"model_path": str(model_path), "vae": "none"}}})

    _backbone_with_reason1(False, has_vae=False).save_deploy_assets(str(output_dir), cfg)

    comps = OmegaConf.to_container(cfg.model.video_backbone.components, resolve=True)
    assert [c["attr"] for c in comps] == []  # neither vae nor text_encoder emitted


def test_save_deploy_assets_accepts_plain_dict_cfg(tmp_path):
    """Defensive: a plain ``dict`` cfg (not a DictConfig) must work and the
    merged ``components`` must propagate back to the caller's dict."""
    model_path = tmp_path / "cosmos_bundle"
    model_path.mkdir()
    output_dir = tmp_path / "ckpt_out"
    output_dir.mkdir()

    cfg = {"model": {"video_backbone": {"model_path": str(model_path), "text_encoder_path": None}}}
    _backbone_with_reason1(False).save_deploy_assets(str(output_dir), cfg)

    comps = cfg["model"]["video_backbone"]["components"]
    assert [c["attr"] for c in comps] == ["vae"]
    assert not (output_dir / "reason1").exists()


def test_save_deploy_assets_does_not_clobber_explicit_components(tmp_path):
    """An operator-provided ``components`` list is preserved, not overwritten."""
    from omegaconf import OmegaConf

    model_path = tmp_path / "cosmos_bundle"
    model_path.mkdir()
    te_path = tmp_path / "reason1_src"
    te_path.mkdir()
    (te_path / "config.json").write_text("{}")
    output_dir = tmp_path / "ckpt_out"
    output_dir.mkdir()

    explicit = [{"attr": "vae", "source": "custom", "sub_module": "override"}]
    cfg = OmegaConf.create(
        {
            "model": {
                "video_backbone": {
                    "model_path": str(model_path),
                    "text_encoder_path": str(te_path),
                    "components": explicit,
                }
            }
        }
    )

    _backbone_with_reason1(True).save_deploy_assets(str(output_dir), cfg)

    comps = OmegaConf.to_container(cfg.model.video_backbone.components, resolve=True)
    assert comps == explicit  # untouched
