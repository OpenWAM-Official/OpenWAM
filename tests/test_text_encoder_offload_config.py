"""YAML controls frozen Wan text-weight offload independently of checkpointing."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

from openwam.model.architectures.dual_system.idm import DualSystemIDMArchitecture
from openwam.model.architectures.dual_system.joint_cross_attn import DualSystemCrossAttnArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.single_system import SingleSystemMoEArchitecture, SingleSystemVanillaArchitecture
from openwam.model.architectures.tri_system import TriSystemJointSelfAttnArchitecture
from openwam.model.video_backbone.wan import encode as wan_encode
from tests.test_openwam_trainer import _MockVideoBackbone
from tests.test_text_offload_lifecycle import _assert_device, _Encoder, _Tokenizer
from tests.test_vace_context_truncation import _make_backbone, _preprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = [
    SingleSystemVanillaArchitecture,
    SingleSystemMoEArchitecture,
    DualSystemSelfAttnArchitecture,
    DualSystemCrossAttnArchitecture,
    DualSystemIDMArchitecture,
    TriSystemJointSelfAttnArchitecture,
]


@pytest.fixture(autouse=True)
def _isolate_text_optimizations(monkeypatch):
    for name, value in {
        "OPENWAM_TEXT_CACHE": "1",
        "OPENWAM_TEXT_CACHE_MAX_ENTRIES": "8",
        "OPENWAM_TEXT_MISS_FAST": "0",
        "OPENWAM_CONTEXT_TRUNC": "0",
        "OPENWAM_VAE_FIRST": "0",
    }.items():
        monkeypatch.setenv(name, value)


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="train", overrides=list(overrides))


def _make_trainer(monkeypatch, cfg, architecture_type=DualSystemSelfAttnArchitecture, device="cpu"):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    arch = architecture_type()
    bb = _make_backbone("plain")
    # Real Wan preprocessing and small trainable-capable text weights, avoiding
    # foundation checkpoint loading and unrelated action/video transformer work.
    bb.text_encoder = _Encoder().to(dtype=bb.dtype).eval().requires_grad_(False)
    bb._tokenizer = _Tokenizer()
    arch.video_backbone = bb
    arch._device = torch.device(device)
    arch._dtype = bb.dtype
    monkeypatch.setattr(arch, "init_training_schedulers", lambda *args: None)
    resolved = SimpleNamespace(
        registry_name=architecture_type.__name__,
        params={},
        canonical=SimpleNamespace(framework="test", variant="test"),
    )
    monkeypatch.setattr("openwam.model.resolve_architecture_config", lambda _cfg: resolved)
    monkeypatch.setattr("openwam.model.build_architecture", lambda *_args: arch)
    monkeypatch.setattr("openwam.train.openwam_trainer.log_parameter_counts", lambda *args, **kwargs: None)
    trainer = OpenWAMTrainer(cfg, accelerator=None, dataset=None)
    assert trainer.architecture is arch
    return trainer, bb


@pytest.mark.parametrize("override, expected", [(None, False), ("false", False), ("true", True)])
def test_hydra_exposes_boolean_text_offload(override, expected):
    cfg = _compose(*(() if override is None else (f"training.text_encoder_offload={override}",)))
    assert cfg.training.text_encoder_offload is expected


@pytest.mark.parametrize("value", [None, 0, 1, -1, 1.0, "true", "false"])
def test_invalid_config_fails_before_loading_model_weights(monkeypatch, value):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    cfg = _compose()
    cfg.training.text_encoder_offload = value

    def unexpected_build(*args, **kwargs):
        pytest.fail("Invalid text offload configuration reached model loading")

    monkeypatch.setattr("openwam.model.build_architecture", unexpected_build)
    with pytest.raises(ValueError, match="training.text_encoder_offload.*boolean"):
        OpenWAMTrainer(cfg, accelerator=None, dataset=None)


@pytest.mark.parametrize("architecture_type", ARCHITECTURES)
@pytest.mark.parametrize("enabled", [False, True])
def test_yaml_trainer_forwards_offload_for_every_architecture_independently_of_checkpointing(
    monkeypatch, architecture_type, enabled
):
    cfg = _compose(
        f"training.text_encoder_offload={str(enabled).lower()}",
        "training.use_gradient_checkpointing=false",
        "training.use_gradient_checkpointing_offload=false",
    )
    trainer, bb = _make_trainer(monkeypatch, cfg, architecture_type)
    assert bb._text_encoder_offload is enabled
    assert trainer.architecture._use_gradient_checkpointing is False
    assert trainer.architecture._use_gradient_checkpointing_offload is False


def test_older_config_without_new_field_keeps_offload_disabled(monkeypatch):
    cfg = _compose()
    del cfg.training.text_encoder_offload
    _, bb = _make_trainer(monkeypatch, cfg)
    actual = _preprocess(bb, ["old config"])
    assert torch.isfinite(actual["context"]).all()
    assert not getattr(bb.text_encoder, "_openwam_text_offload", None)


@pytest.mark.parametrize("source_key", ["finetune_ckpt_path", "resume_ckpt_path"])
@pytest.mark.parametrize("enabled", [False, True])
def test_checkpoint_loading_uses_current_training_offload_config(monkeypatch, source_key, enabled):
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils import ckpt_model_loader

    trainer, bb = _make_trainer(monkeypatch, _compose())
    cfg = _compose(f"training.text_encoder_offload={str(enabled).lower()}")
    cfg.training[source_key] = "/unused/test-checkpoint"
    checkpoint_cfg = _compose(f"training.text_encoder_offload={str(not enabled).lower()}")
    resolved = SimpleNamespace(
        registry_name="test", params={}, canonical=SimpleNamespace(framework="dual_system", variant="joint_self_attn")
    )
    monkeypatch.setattr(
        ckpt_model_loader,
        "build_architecture_from_ckpt_dir",
        lambda *args, **kwargs: (resolved, trainer.architecture, checkpoint_cfg),
    )
    monkeypatch.setattr(ckpt_model_loader, "propagate_component_specs", lambda *args: None)
    monkeypatch.setattr(ckpt_model_loader, "warn_live_model_cfg_ignored", lambda *args: None)
    restored = OpenWAMTrainer(cfg, accelerator=None, dataset=None)
    assert restored.architecture.video_backbone is bb
    assert bb._text_encoder_offload is enabled


@pytest.mark.parametrize("legacy_value", ["0", "1", "not-a-boolean"])
def test_legacy_environment_cannot_enable_offload_over_yaml_false(monkeypatch, legacy_value):
    monkeypatch.setenv("OPENWAM_TEXT_ENCODER_OFFLOAD", legacy_value)
    _, bb = _make_trainer(monkeypatch, _compose("training.text_encoder_offload=false"))
    reference = wan_encode.encode_text(
        ["same"], tokenizer=bb._tokenizer, text_encoder=bb.text_encoder, device=bb.device
    )[0]
    for _ in range(2):
        actual = _preprocess(bb, ["same"])
        torch.testing.assert_close(actual["context"], reference, rtol=0, atol=0)
    assert bb.text_encoder.calls == 2  # reference + one miss; the second preprocess hits
    assert not getattr(bb.text_encoder, "_openwam_text_offload", None)


def test_offload_without_text_cache_warns_and_runs_normally(monkeypatch, caplog):
    monkeypatch.setenv("OPENWAM_TEXT_CACHE", "0")
    _, bb = _make_trainer(monkeypatch, _compose("training.text_encoder_offload=true"))
    for _ in range(2):
        actual = _preprocess(bb, ["uncached"])
        assert torch.isfinite(actual["context"]).all()
    assert bb.text_encoder.calls == 2
    assert "_train_text_cache" not in bb.__dict__
    assert not getattr(bb.text_encoder, "_openwam_text_offload", None)
    warnings = [record.message for record in caplog.records if "OPENWAM_TEXT_CACHE" in record.message]
    assert len(warnings) == 1
    assert "training.text_encoder_offload" in warnings[0]


@pytest.mark.parametrize("mode", ["train", "trainable"])
def test_runtime_enabled_still_bypasses_offload_for_mutable_encoder(monkeypatch, mode):
    # Do not run Trainer's freeze wrapper: this deliberately exercises the
    # independent train/eval and requires_grad runtime safeguards.
    bb = _make_backbone("plain")
    bb.text_encoder = _Encoder().to(dtype=bb.dtype).eval().requires_grad_(False)
    bb._tokenizer = _Tokenizer()
    bb.set_text_encoder_offload(True)
    if mode == "train":
        bb.text_encoder.train()
    else:
        bb.text_encoder.requires_grad_(True)
    actual = _preprocess(bb, ["mutable"])
    assert torch.isfinite(actual["context"]).all()
    cache = bb.__dict__["_train_text_cache"]
    assert not cache.rows
    manager = getattr(bb.text_encoder, "_openwam_text_offload", None)
    assert manager is None or not manager.enabled
    if mode == "trainable":
        actual["context"].square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bb.text_encoder.parameters())


def test_unsupported_backbone_accepts_default_but_rejects_explicit_offload():
    arch = DualSystemSelfAttnArchitecture()
    arch.video_backbone = _MockVideoBackbone()
    arch.set_training_runtime(text_encoder_offload=False)
    with pytest.raises(NotImplementedError, match="text_encoder_offload"):
        arch.set_training_runtime(text_encoder_offload=True)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("miss_fast", ["0", "1"])
def test_yaml_enabled_offload_preserves_cached_rows_and_disable_restores_gpu_weights(monkeypatch, miss_fast):
    monkeypatch.setenv("OPENWAM_TEXT_ENCODER_OFFLOAD", "0")
    monkeypatch.setenv("OPENWAM_TEXT_MISS_FAST", miss_fast)
    cfg = _compose("training.text_encoder_offload=true", "training.use_gradient_checkpointing=false")
    trainer, bb = _make_trainer(monkeypatch, cfg, device="cuda")
    reference = {
        prompt: wan_encode.encode_text(
            [prompt], tokenizer=bb._tokenizer, text_encoder=bb.text_encoder, device=bb.device
        )[0]
        for prompt in ("first", "new prompt")
    }
    for prompt in ("first", "first", "new prompt"):
        actual = _preprocess(bb, [prompt])
        torch.testing.assert_close(actual["context"], reference[prompt])
        _assert_device(bb.text_encoder, "cpu")
    manager = bb.text_encoder._openwam_text_offload
    assert manager.loads == 1
    trainer.architecture.set_training_runtime(text_encoder_offload=False)
    _assert_device(bb.text_encoder, "cuda")
    assert not manager.enabled and manager.host is None and not manager.slots
    assert not getattr(bb.text_encoder, "_openwam_text_offload", None)
    assert not getattr(bb.__dict__["_train_text_cache"], "offload", None)
    actual = _preprocess(bb, ["first"])
    torch.testing.assert_close(actual["context"], reference["first"])
    _assert_device(bb.text_encoder, "cuda")

    # Disabling also leaves uncached inference usable, and a later explicit
    # enable starts a fresh lifecycle from the current live weights.
    context, _ = wan_encode.encode_text_for_inference(
        "new prompt",
        vace_cache=None,
        prompt_embed_cache=None,
        tokenizer=bb._tokenizer,
        text_encoder=bb.text_encoder,
        device=bb.device,
    )
    torch.testing.assert_close(context, reference["new prompt"])
    trainer.architecture.set_training_runtime(text_encoder_offload=True)
    bb.__dict__["_train_text_cache"].clear()
    actual = _preprocess(bb, ["first"])
    torch.testing.assert_close(actual["context"], reference["first"])
    _assert_device(bb.text_encoder, "cpu")
    assert bb.text_encoder._openwam_text_offload is not manager
