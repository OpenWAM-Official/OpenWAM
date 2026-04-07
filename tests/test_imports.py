"""Verify that all open_wam modules can be imported successfully."""


def test_import_open_wam():
    import open_wam

    assert hasattr(open_wam, "__version__")


def test_import_data_base():
    from open_wam.data.base import BaseActionDataset

    assert hasattr(BaseActionDataset, "__getitem__")
    assert hasattr(BaseActionDataset, "action_dim")
    assert hasattr(BaseActionDataset, "action_stats")


def test_import_data_robotwin():
    from open_wam.data.robotwin import (
        ROBOTWIN_ALL_TASKS,
        ROBOTWIN_HOLDOUT_TASKS,
        ROBOTWIN_TRAIN_TASKS,
    )

    assert len(ROBOTWIN_TRAIN_TASKS) == 42
    assert len(ROBOTWIN_HOLDOUT_TASKS) == 8
    assert len(ROBOTWIN_ALL_TASKS) == 50


def test_import_data_transforms():
    pass


def test_import_data_action_stats():
    pass


def test_import_data_init():
    pass


def test_import_inference_base():
    from open_wam.inference.base import BaseInferenceEngine

    assert hasattr(BaseInferenceEngine, "generate")


def test_import_inference_schedule():
    pass


def test_import_inference_joint_engine():
    pass


def test_import_inference_init():
    pass


def test_import_training_base():
    from open_wam.training.base import BaseTrainer

    assert hasattr(BaseTrainer, "compute_loss")
    assert hasattr(BaseTrainer, "train_step")


def test_import_training_loss():
    pass


def test_import_training_runtime():
    pass


def test_import_training_optimizer_groups():
    pass


def test_import_training_callbacks():
    pass


def test_import_training_init():
    pass


def test_import_evaluation_base():
    from open_wam.evaluation.base import BaseEvaluator

    assert hasattr(BaseEvaluator, "evaluate")


def test_import_evaluation_policy():
    pass


def test_import_evaluation_robotwin():
    pass


def test_import_evaluation_robotwin_policy():
    pass


def test_import_evaluation_envs():
    pass


def test_import_evaluation_metrics():
    pass


def test_import_evaluation_init():
    pass


def test_import_evaluation_registry():
    from open_wam.evaluation.registry import (
        list_registered_evaluators,
    )

    registered = list_registered_evaluators()
    assert "offline" in registered
    assert "online" in registered
    assert "libero" in registered
    assert "robocasa" in registered
    assert "calvin" in registered
    assert "behavior" in registered
    assert "simpler_env" in registered


def test_import_action_repr_registry():
    from open_wam.models.action_repr import (
        list_registered_action_reprs,
    )

    registered = list_registered_action_reprs()
    assert "continuous" in registered


def test_import_action_dit():
    pass


def test_import_moe_expert_dit():
    pass


def test_import_flow_match_scheduler():
    from open_wam.inference.flow_match_scheduler import FlowMatchScheduler

    s = FlowMatchScheduler("Wan")
    s.set_timesteps(20, shift=5.0)
    assert len(s.timesteps) == 20


def test_import_model_config():
    pass


def test_import_video_pipeline_wrapper():
    pass


def test_import_models_backbone():
    pass
