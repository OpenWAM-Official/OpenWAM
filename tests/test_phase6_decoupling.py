"""Tests for package-native decoupling in inference/evaluation entrypoints."""

from pathlib import Path


def test_package_native_model_loader_exists():
    """Inference package should expose a package-level model loader."""
    from open_wam.inference import load_wam_models

    assert callable(load_wam_models)


def test_infer_script_no_longer_imports_eval_script():
    """Inference script should not depend on the eval script helper."""
    source = Path("scripts/infer.py").read_text()
    assert "from scripts.eval import _load_models" not in source


def test_policy_server_no_longer_imports_eval_script():
    """Serving path should no longer depend on script-level eval helpers."""
    source = Path("open_wam/serving/policy_server.py").read_text()
    assert "from scripts.eval import _load_models" not in source
