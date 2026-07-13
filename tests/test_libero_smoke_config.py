import importlib.util
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml


def _load_module(repo_root: Path, relative_path: str, name: str):
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_smoke_module(repo_root: Path):
    return _load_module(repo_root, "benchmarks/libero/smoke_libero.py", "libero_smoke")


def _load_interface_module(repo_root: Path):
    return _load_module(repo_root, "benchmarks/libero/openwam2libero_interface.py", "openwam2libero_interface")


def _fake_libero_repo(root: Path):
    package_root = root / "libero" / "libero"
    for name in ["bddl_files", "init_files", "assets"]:
        (package_root / name).mkdir(parents=True, exist_ok=True)
    return package_root


def test_libero_smoke_writes_isolated_configs(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    smoke = _load_smoke_module(repo_root)
    ordinary_repo = tmp_path / "LIBERO"
    ordinary_package_root = _fake_libero_repo(ordinary_repo)
    ordinary_config_root = tmp_path / "ordinary-config"
    monkeypatch.setenv("LIBERO_PATH", str(ordinary_repo))
    monkeypatch.setenv("LIBERO_CONFIG_ROOT", str(ordinary_config_root))

    ordinary = smoke.write_config()

    assert ordinary["benchmark_root"] == str(ordinary_package_root)
    ordinary_yaml = yaml.safe_load((ordinary_config_root / "config.yaml").read_text(encoding="utf-8"))
    assert ordinary_yaml["assets"] == str(ordinary_package_root / "assets")


def test_libero_smoke_rejects_missing_repo_env(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    smoke = _load_smoke_module(repo_root)

    monkeypatch.delenv("LIBERO_PATH", raising=False)
    with pytest.raises(SystemExit, match="LIBERO_PATH is not set"):
        smoke.write_config()


def test_libero_policy_reuses_ws_and_rotates_images(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    iface = _load_interface_module(repo_root)

    instances = []

    class FakeWSPolicyClient:
        def __init__(self, *args, **kwargs):
            self.closed = 0
            self.resets = 0
            self.predictions = []
            instances.append(self)

        def ping(self):
            return {"type": iface.transport.PONG}

        def reset(self):
            self.resets += 1
            return {"type": iface.transport.RESET_ACK}

        def predict(self, payload):
            self.predictions.append(payload)
            return {"action": [0.0] * 7}

        def close(self):
            self.closed += 1

    monkeypatch.setattr(iface, "WSPolicyClient", FakeWSPolicyClient)
    monkeypatch.setattr(iface.client, "encode_numpy_b64", lambda image: f"first={int(image[0, 0, 0])}")

    policy = iface.OpenWAMLiberoPolicy()
    ws = instances[0]
    policy.reset()
    action = policy.act(
        {
            "agentview_image": np.array(
                [
                    [[1, 0, 0], [2, 0, 0]],
                    [[3, 0, 0], [4, 0, 0]],
                ],
                dtype=np.uint8,
            ),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        "pick up the bowl",
    )

    assert action.shape == (7,)
    assert ws.resets == 1
    assert ws.closed == 0
    assert ws.predictions[0]["images"]["head_camera"] == "first=4"
    policy.close()
    assert ws.closed == 1


def test_libero_shell_scripts_are_valid():
    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        repo_root / "benchmarks" / "libero" / "run_smoke.sh",
        repo_root / "benchmarks" / "libero" / "single_eval.sh",
    ]
    result = subprocess.run(
        ["bash", "-n", *map(os.fspath, scripts)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
