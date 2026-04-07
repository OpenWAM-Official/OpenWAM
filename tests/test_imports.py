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
        RoboTwinActionDataset,
        MultiTaskRoboTwinActionDataset,
        ROBOTWIN_TRAIN_TASKS,
        ROBOTWIN_HOLDOUT_TASKS,
        ROBOTWIN_ALL_TASKS,
        MULTIVIEW_LAYOUT,
        MULTIVIEW_CAMERAS,
        discover_robotwin_roots,
        assemble_multiview_grid,
        extract_quadrant,
    )
    assert len(ROBOTWIN_TRAIN_TASKS) == 42
    assert len(ROBOTWIN_HOLDOUT_TASKS) == 8
    assert len(ROBOTWIN_ALL_TASKS) == 50


def test_import_data_transforms():
    from open_wam.data.transforms import crop_and_resize, pad_and_resize, resize_frame


def test_import_data_action_stats():
    from open_wam.data.action_stats import (
        compute_action_stats,
        compute_multitask_robotwin_stats,
        parse_tasks_file,
    )


def test_import_data_init():
    from open_wam.data import (
        BaseActionDataset,
        RoboTwinActionDataset,
        MultiTaskRoboTwinActionDataset,
    )


def test_import_inference_base():
    from open_wam.inference.base import BaseInferenceEngine
    assert hasattr(BaseInferenceEngine, "generate")


def test_import_inference_schedule():
    from open_wam.inference.schedule import (
        Schedule,
        make_schedule,
        schedule_sync,
        schedule_video_leading,
        schedule_cascade,
        schedule_action_only,
    )


def test_import_inference_joint_engine():
    from open_wam.inference.joint_engine import JointInferenceEngine


def test_import_inference_init():
    from open_wam.inference import (
        BaseInferenceEngine,
        JointInferenceEngine,
        Schedule,
        make_schedule,
    )


def test_import_training_base():
    from open_wam.training.base import BaseTrainer
    assert hasattr(BaseTrainer, "compute_loss")
    assert hasattr(BaseTrainer, "train_step")


def test_import_training_loss():
    from open_wam.training.loss import FlowMatchVideoActionSFTLoss


def test_import_training_runtime():
    from open_wam.training.runtime import (
        build_training_dataset,
        build_validation_datasets,
        cfg_to_flat_namespace,
    )


def test_import_training_optimizer_groups():
    from open_wam.training.optimizer_groups import (
        attach_optimizer_groups,
        build_trainable_parameters,
    )


def test_import_training_callbacks():
    from open_wam.training.callbacks import (
        TrainingCallback,
        CallbackRunner,
        ValidationLossCallback,
        VideoLogCallback,
        SetupCallback,
    )


def test_import_training_init():
    from open_wam.training import (
        BaseTrainer,
        NativeTrainer,
        FlowMatchVideoActionSFTLoss,
        CallbackRunner,
    )


def test_import_evaluation_base():
    from open_wam.evaluation.base import BaseEvaluator
    assert hasattr(BaseEvaluator, "evaluate")


def test_import_evaluation_policy():
    from open_wam.evaluation.policy import WAMPolicy


def test_import_evaluation_robotwin():
    from open_wam.evaluation.robotwin_evaluator import (
        RoboTwinOfflineEvaluator,
        RoboTwinOnlineEvaluator,
    )


def test_import_evaluation_robotwin_policy():
    from open_wam.evaluation.robotwin_policy import (
        get_model,
        eval_one_step,
        reset_model,
    )


def test_import_evaluation_envs():
    from open_wam.evaluation.envs.base import BaseEnvAdapter
    from open_wam.evaluation.envs.robotwin import RoboTwinEnvAdapter


def test_import_evaluation_metrics():
    from open_wam.evaluation.metrics import compute_video_metrics


def test_import_evaluation_init():
    from open_wam.evaluation import (
        BaseEvaluator,
        WAMPolicy,
        RoboTwinOfflineEvaluator,
        RoboTwinOnlineEvaluator,
    )


def test_import_evaluation_registry():
    from open_wam.evaluation.registry import (
        build_evaluator,
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
        build_action_representation,
        list_registered_action_reprs,
    )
    registered = list_registered_action_reprs()
    assert "continuous" in registered


def test_import_action_dit():
    from open_wam.models.action_dit import (
        ActionDiT,
        ActionDiTState,
        sinusoidal_embedding_1d,
        RMSNorm,
    )


def test_import_moe_expert_dit():
    from open_wam.models.moe_expert_dit import MoEExpertDiT, MoEExpertState


def test_import_flow_match_scheduler():
    from open_wam.inference.flow_match_scheduler import FlowMatchScheduler
    s = FlowMatchScheduler("Wan")
    s.set_timesteps(20, shift=5.0)
    assert len(s.timesteps) == 20


def test_import_model_config():
    from open_wam.inference.model_config import ModelConfig


def test_import_video_pipeline_wrapper():
    from open_wam.inference.video_pipeline import WanVideoPipeline


def test_import_models_backbone():
    from open_wam.models.backbone.base import BaseVideoBackbone
