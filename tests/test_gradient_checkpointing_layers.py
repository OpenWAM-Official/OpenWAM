"""YAML layer limits control the existing dual-system joint MoT checkpoint path."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

from openwam.model.architectures.dual_system.idm import DualSystemIDMArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from tests.test_mot_driver import _build_run_inputs, _make_action_dit, _make_states
from tests.test_openwam_trainer import _MockVideoBackbone

REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_LAYERS = 4


@pytest.fixture(autouse=True)
def _disable_independent_attention_checkpoint(monkeypatch):
    # Count only whole-layer recomputations, independently of the mixed-attn
    # checkpoint setting, which has a separate configuration and scope.
    monkeypatch.setenv("OPENWAM_MIXED_ATTN_CKPT", "0")


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="train", overrides=list(overrides))


def _make_driver():
    vb = _MockVideoBackbone(dim=32, num_layers=NUM_LAYERS, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=NUM_LAYERS)
    ab.train()
    return DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ((), None),
        (("training.gradient_checkpointing_layers=0",), 0),
        (("training.gradient_checkpointing_layers=2",), 2),
    ],
)
def test_hydra_exposes_checkpoint_layer_limit(overrides, expected):
    cfg = _compose(*overrides)
    assert cfg.training.gradient_checkpointing_layers == expected
    assert cfg.training.use_gradient_checkpointing is True


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("value", [True, False, -1, 1.0, "2"])
def test_runtime_rejects_invalid_layer_limits_even_when_checkpointing_is_disabled(enabled, value):
    arch = DualSystemSelfAttnArchitecture()
    with pytest.raises(ValueError, match="gradient_checkpointing_layers"):
        arch.set_training_runtime(use_gradient_checkpointing=enabled, gradient_checkpointing_layers=value)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("limit, expected_count", [(None, 4), (0, 0), (2, 2), (99, 4)])
def test_joint_driver_selects_first_n_layers(monkeypatch, enabled, limit, expected_count):
    driver = _make_driver()
    vstate, astate = _make_states(driver.vb, driver.ab, B=1, s_video=3, s_action=2)
    selected = []

    def record_step(layer_id, video_state, action_state, **kwargs):
        selected.append((layer_id, kwargs["use_gradient_checkpointing"]))
        return video_state, action_state

    monkeypatch.setattr(driver, "step", record_step)
    driver.run_joint_loop(
        vstate,
        astate,
        use_gradient_checkpointing=enabled,
        use_gradient_checkpointing_offload=True,
        gradient_checkpointing_layers=limit,
    )
    assert selected == [(i, enabled and i < expected_count) for i in range(NUM_LAYERS)]


@pytest.mark.parametrize("value", [True, -1, 1.5, "2"])
def test_direct_driver_rejects_invalid_limits_before_running_layers(monkeypatch, value):
    driver = _make_driver()
    vstate, astate = _make_states(driver.vb, driver.ab, B=1, s_video=3, s_action=2)

    def unexpected_step(*args, **kwargs):
        pytest.fail("Invalid layer limit reached a transformer layer")

    monkeypatch.setattr(driver, "step", unexpected_step)
    with pytest.raises(ValueError, match="gradient_checkpointing_layers"):
        driver.run_joint_loop(vstate, astate, gradient_checkpointing_layers=value)


@pytest.mark.parametrize("legacy_value", ["0", "99", "not-an-integer"])
def test_removed_environment_variable_cannot_override_yaml_limit(monkeypatch, legacy_value):
    monkeypatch.setenv("OPENWAM_CKPT_LAYERS", legacy_value)
    driver = _make_driver()
    vstate, astate = _make_states(driver.vb, driver.ab, B=1, s_video=3, s_action=2)
    selected = []

    def record_step(layer_id, video_state, action_state, **kwargs):
        selected.append(kwargs["use_gradient_checkpointing"])
        return video_state, action_state

    monkeypatch.setattr(driver, "step", record_step)
    driver.run_joint_loop(vstate, astate, use_gradient_checkpointing=True, gradient_checkpointing_layers=2)
    assert selected == [True, True, False, False]


@pytest.mark.parametrize("architecture_type", [DualSystemSelfAttnArchitecture, DualSystemIDMArchitecture])
@pytest.mark.parametrize("enabled", [False, True])
def test_yaml_trainer_runtime_reaches_architecture_and_joint_driver(monkeypatch, architecture_type, enabled):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    arch = architecture_type()
    driver = _make_driver()
    arch.video_backbone = driver.vb
    arch.action_backbone = driver.ab
    arch._mot_driver = driver
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    monkeypatch.setattr(arch, "set_dtype_device", lambda *args: None)
    monkeypatch.setattr(arch, "init_training_schedulers", lambda *args: None)
    monkeypatch.setattr(arch, "freeze_modules", lambda _names: [])
    resolved = SimpleNamespace(
        registry_name="test_joint_architecture",
        params={},
        canonical=SimpleNamespace(framework="dual_system", variant="joint_self_attn"),
    )
    monkeypatch.setattr("openwam.model.resolve_architecture_config", lambda _cfg: resolved)
    monkeypatch.setattr("openwam.model.build_architecture", lambda *_args: arch)
    monkeypatch.setattr("openwam.train.openwam_trainer.log_parameter_counts", lambda *args, **kwargs: None)
    cfg = _compose(
        "training.gradient_checkpointing_layers=2",
        f"training.use_gradient_checkpointing={str(enabled).lower()}",
    )
    trainer = OpenWAMTrainer(cfg, accelerator=None, dataset=None)
    assert trainer.architecture is arch
    assert arch._gradient_checkpointing_layers == 2
    selected = []
    forwarded = []
    run_joint_loop = driver.run_joint_loop

    def record_loop(*args, **kwargs):
        forwarded.append(kwargs["gradient_checkpointing_layers"])
        return run_joint_loop(*args, **kwargs)

    def record_step(layer_id, video_state, action_state, **kwargs):
        selected.append(kwargs["use_gradient_checkpointing"])
        return video_state, action_state

    monkeypatch.setattr(driver, "run_joint_loop", record_loop)
    monkeypatch.setattr(driver, "step", record_step)
    arch(
        torch.randn(1, 2, driver.ab.action_dim),
        torch.zeros(1),
        latents=torch.randn(1, 16, 1, 1, 3),
        context=torch.randn(1, 3, driver.ab.text_dim),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
        use_gradient_checkpointing=arch._use_gradient_checkpointing,
        use_gradient_checkpointing_offload=arch._use_gradient_checkpointing_offload,
    )
    assert forwarded == [2]
    assert selected == [enabled, enabled, False, False]


@pytest.mark.parametrize(
    "enabled, limit, offload, expected_calls",
    [
        (True, None, False, [2, 2, 2, 2]),
        (True, 2, False, [2, 2, 1, 1]),
        (True, 0, False, [1, 1, 1, 1]),
        (True, 99, False, [2, 2, 2, 2]),
        (True, 2, True, [2, 2, 1, 1]),
        (False, 2, True, [1, 1, 1, 1]),
    ],
)
def test_real_backward_recomputes_selected_layers_and_preserves_outputs_and_gradients(
    monkeypatch, enabled, limit, offload, expected_calls
):
    torch.manual_seed(13)
    driver = _make_driver()
    real_step_impl = driver._step_impl

    def run(checkpointing, layer_limit, cpu_offload):
        calls = [0] * NUM_LAYERS

        def count_step(layer_id, *args, **kwargs):
            calls[layer_id] += 1
            return real_step_impl(layer_id, *args, **kwargs)

        monkeypatch.setattr(driver, "_step_impl", count_step)
        driver.ab.zero_grad(set_to_none=True)
        vstate, astate = _build_run_inputs(driver.vb, driver.ab, B=1, s_video=3, s_action=2, seed=7)
        video_leaf = vstate.hidden_states
        vstate, astate = driver.run_joint_loop(
            vstate,
            astate,
            use_gradient_checkpointing=checkpointing,
            use_gradient_checkpointing_offload=cpu_offload,
            gradient_checkpointing_layers=layer_limit,
        )
        video = vstate.hidden_states
        action = astate.payload.x_action
        (video.square().mean() + action.square().mean()).backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in driver.ab.named_parameters()
            if parameter.grad is not None
        }
        return video.detach().clone(), action.detach().clone(), video_leaf.grad.detach().clone(), gradients, calls

    reference = run(False, None, False)
    result = run(enabled, limit, offload)
    assert reference[-1] == [1, 1, 1, 1]
    assert result[-1] == expected_calls
    for actual, expected in zip(result[:3], reference[:3]):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert result[3].keys() == reference[3].keys()
    assert result[3], "The comparison must exercise trainable action parameters"
    for name in result[3]:
        torch.testing.assert_close(result[3][name], reference[3][name], rtol=1e-4, atol=1e-6, msg=name)
