"""Training presets resolve through Hydra before importing training dependencies."""

import builtins
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def train_module(monkeypatch):
    # Preset loading intentionally changes the process environment. Keep every
    # change, including keys we do not explicitly assert, local to this test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    name = "openwam_train_speed_preset_test"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts/train.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    real_import = builtins.__import__

    def lightweight_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "openwam", "accelerate"}:
            pytest.fail(f"Training dependency imported before preset configuration: {name}")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", lightweight_import)
        spec.loader.exec_module(module)
    return module


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="train", overrides=["dataloader=libero", *overrides])


def _preset_config(path):
    return OmegaConf.create({"training": {"speed_preset": str(path) if path is not None else None}})


@pytest.mark.parametrize(
    ("environment", "overrides", "expected"),
    [
        (None, (), "presets/fast.env"),
        ("presets/fast_80gb.env", (), "presets/fast_80gb.env"),
        ("presets/fast.env", ("training.speed_preset=presets/fast_80gb.env",), "presets/fast_80gb.env"),
        ("presets/fast_80gb.env", ("training.speed_preset=presets/fast.env",), "presets/fast.env"),
        ("presets/fast.env", ("training.speed_preset=null",), None),
    ],
)
def test_hydra_preset_selection(monkeypatch, environment, overrides, expected):
    if environment is None:
        monkeypatch.delenv("OPENWAM_PRESET", raising=False)
    else:
        monkeypatch.setenv("OPENWAM_PRESET", environment)
    cfg = _compose(*overrides)
    assert cfg.training.speed_preset == expected
    assert cfg.dataloader.type == "libero"


def test_default_config_loads_fast_preset_without_environment_prefix(train_module, monkeypatch):
    monkeypatch.delenv("OPENWAM_PRESET", raising=False)
    monkeypatch.setenv("OPENWAM_COMPILE_BLOCKS", "0")

    train_module._apply_speed_preset(_compose())

    assert os.environ["OPENWAM_COMPILE_BLOCKS"] == "1"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_cli_null_does_not_load_legacy_environment_preset(train_module, monkeypatch):
    monkeypatch.setenv("OPENWAM_PRESET", "presets/fast.env")
    monkeypatch.setenv("OPENWAM_COMPILE_BLOCKS", "0")
    before = dict(os.environ)

    train_module._apply_speed_preset(_compose("training.speed_preset=null"))

    assert dict(os.environ) == before


def test_legacy_environment_preset_is_applied(train_module, monkeypatch):
    monkeypatch.setenv("OPENWAM_PRESET", "presets/fast_80gb.env")
    train_module._apply_speed_preset(_compose())
    assert os.environ["OPENWAM_TEXT_ENCODER_OFFLOAD"] == "1"


def test_relative_preset_uses_project_root_instead_of_working_directory(train_module, monkeypatch, tmp_path):
    root = tmp_path / "project"
    (root / "presets").mkdir(parents=True)
    (root / "presets/test.env").write_text("OPENWAM_PRESET_TEST=project\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "presets").mkdir(parents=True)
    (elsewhere / "presets/test.env").write_text("OPENWAM_PRESET_TEST=wrong\n", encoding="utf-8")
    monkeypatch.setattr(train_module, "PROJECT_ROOT", root)
    monkeypatch.chdir(elsewhere)

    train_module._apply_speed_preset(_preset_config("presets/test.env"))

    assert os.environ["OPENWAM_PRESET_TEST"] == "project"


def test_plain_assignments_allow_comments_blank_lines_and_equals_in_values(train_module, tmp_path):
    preset = tmp_path / "custom.env"
    preset.write_text(
        "# A comment\n\nOPENWAM_PRESET_TEST=enabled\nOPENWAM_PRESET_EQUALS=left=right\n",
        encoding="utf-8",
    )
    train_module._apply_speed_preset(_preset_config(preset))
    assert os.environ["OPENWAM_PRESET_TEST"] == "enabled"
    assert os.environ["OPENWAM_PRESET_EQUALS"] == "left=right"


@pytest.mark.parametrize(
    "bad_line",
    ["not an assignment", "9INVALID=value", "INVALID-NAME=value", "export KEY=value", "BAD=with\x00nul"],
)
def test_invalid_file_fails_atomically(train_module, monkeypatch, tmp_path, bad_line):
    preset = tmp_path / "invalid.env"
    preset.write_text(f"OPENWAM_PRESET_TEST=new\n{bad_line}\n", encoding="utf-8")
    monkeypatch.setenv("OPENWAM_PRESET_TEST", "original")
    before = dict(os.environ)

    with pytest.raises(ValueError):
        train_module._apply_speed_preset(_preset_config(preset))

    assert dict(os.environ) == before


def test_missing_file_fails_without_environment_changes(train_module, tmp_path):
    preset = tmp_path / "missing.env"
    before = dict(os.environ)
    with pytest.raises(FileNotFoundError, match="missing.env"):
        train_module._apply_speed_preset(_preset_config(preset))
    assert dict(os.environ) == before


def test_shell_command_lines_are_not_executed(train_module, tmp_path):
    marker = tmp_path / "must_not_exist"
    preset = tmp_path / "shell.env"
    preset.write_text(f"touch {marker}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        train_module._apply_speed_preset(_preset_config(preset))
    assert not marker.exists()


def test_worker_applies_preset_before_training_and_torch_import(train_module, monkeypatch, tmp_path):
    preset = tmp_path / "startup.env"
    preset.write_text("OPENWAM_PRESET_STARTUP=ready\n", encoding="utf-8")
    monkeypatch.delenv("OPENWAM_PRESET_STARTUP", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(train_module, "_install_startup_noise_filter", lambda: None)
    torch = types.ModuleType("torch")
    dist = types.ModuleType("torch.distributed")
    dist.is_available = lambda: False
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    calls = []

    def train(cfg):
        assert os.environ.get("OPENWAM_PRESET_STARTUP") == "ready"
        calls.append(cfg)

    real_import = builtins.__import__

    def import_after_preset(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "openwam", "accelerate"}:
            assert os.environ.get("OPENWAM_PRESET_STARTUP") == "ready", name
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(train_module, "_train", train)
    monkeypatch.setattr(builtins, "__import__", import_after_preset)
    cfg = _preset_config(preset)
    train_module.main.__wrapped__(cfg)
    assert calls == [cfg]


def test_invalid_preset_prevents_training_start(train_module, monkeypatch, tmp_path):
    preset = tmp_path / "invalid.env"
    preset.write_text("not an assignment\n", encoding="utf-8")

    def train(_cfg):
        pytest.fail("Training started despite an invalid preset")

    monkeypatch.setattr(train_module, "_train", train)
    with pytest.raises(ValueError):
        train_module.main.__wrapped__(_preset_config(preset))


def test_shell_launcher_defers_preset_and_automatically_selects_gpu_count(tmp_path):
    """The shell must not source a legacy preset before Hydra can disable it."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nprintf 'GPU 0: fake\nGPU 1: fake\n'\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)
    torchrun = fake_bin / "torchrun"
    torchrun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        'print("CAPTURE=" + json.dumps({"args": sys.argv[1:], "preset": os.environ.get("OPENWAM_PRESET")}))\n',
        encoding="utf-8",
    )
    torchrun.chmod(0o755)
    marker = tmp_path / "shell_must_not_execute"
    preset = tmp_path / "legacy_shell.env"
    preset.write_text(f"touch '{marker}'\n", encoding="utf-8")
    env = dict(os.environ)
    for name in (
        "NPROC_PER_NODE",
        "HOST_GPU_NUM",
        "NNODES",
        "HOST_NUM",
        "WORLD_SIZE",
        "NODE_RANK",
        "RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)
    env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
    env["OPENWAM_PRESET"] = str(preset)

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/train.sh"), "dataloader=libero", "training.speed_preset=null"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )

    assert not marker.exists(), "train.sh sourced a preset before Hydra resolved the null override"
    capture_line = next(line for line in result.stdout.splitlines() if line.startswith("CAPTURE="))
    capture = json.loads(capture_line.removeprefix("CAPTURE="))
    args = capture["args"]
    assert args[args.index("--nproc_per_node") + 1] == "2"
    assert args[args.index("--nnodes") + 1] == "1"
    assert args[-3:] == ["scripts/train.py", "dataloader=libero", "training.speed_preset=null"]
    assert capture["preset"] == str(preset)
