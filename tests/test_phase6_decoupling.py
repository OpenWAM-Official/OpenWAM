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


def test_joint_engine_no_longer_imports_legacy_joint_inference():
    """Joint inference engine should use package-native generation code."""
    source = Path("open_wam/inference/joint_engine.py").read_text()
    assert "from joint_inference import generate_video_and_actions" not in source


def test_schedule_module_no_longer_imports_legacy_joint_inference():
    """Schedule generation should now live in the package."""
    source = Path("open_wam/inference/schedule.py").read_text()
    assert "from joint_inference import" not in source


def test_robotwin_evaluator_no_longer_imports_legacy_eval_robotwin():
    """RoboTwin evaluator should use package-native metrics."""
    source = Path("open_wam/evaluation/robotwin_evaluator.py").read_text()
    assert "from eval_robotwin import compute_video_metrics" not in source


def test_robotwin_policy_no_longer_imports_legacy_eval_robotwin():
    """RoboTwin policy compatibility layer should be package-native."""
    source = Path("open_wam/evaluation/robotwin_policy.py").read_text()
    assert "from eval_robotwin import" not in source
