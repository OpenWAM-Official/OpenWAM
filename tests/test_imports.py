"""Verify that all open_wam modules can be imported successfully."""


def test_import_open_wam():
    import open_wam  # noqa: F401

    assert hasattr(open_wam, "__version__")


def test_import_data_base():
    from open_wam.data.base import BaseActionDataset  # noqa: F401

    assert hasattr(BaseActionDataset, "__getitem__")
    assert hasattr(BaseActionDataset, "action_dim")
    assert hasattr(BaseActionDataset, "action_stats")


def test_import_data_robotwin():
    from open_wam.data.robotwin import (  # noqa: F401
        ROBOTWIN_ALL_TASKS,
        ROBOTWIN_HOLDOUT_TASKS,
        ROBOTWIN_TRAIN_TASKS,
    )

    assert len(ROBOTWIN_TRAIN_TASKS) == 50  # All tasks in training by default
    assert len(ROBOTWIN_HOLDOUT_TASKS) == 0
    assert len(ROBOTWIN_ALL_TASKS) == 50


def test_import_data_transforms():
    from open_wam.data.transforms import crop_and_resize, pad_and_resize, resize_frame  # noqa: F401


def test_import_data_action_stats():
    from open_wam.data.action_stats import (  # noqa: F401
        compute_action_stats,
        compute_multitask_robotwin_stats,
        parse_tasks_file,
    )


def test_import_data_init():
    from open_wam.data import (  # noqa: F401
        BaseActionDataset,
        MultiTaskRoboTwinActionDataset,
        RoboTwinActionDataset,
    )


def test_import_inference_base():
    from open_wam.inference.base import BaseInferenceEngine  # noqa: F401

    assert hasattr(BaseInferenceEngine, "generate")


def test_import_inference_schedule():
    from open_wam.inference.schedule import (  # noqa: F401
        Schedule,
        make_schedule,
        schedule_action_only,
        schedule_cascade,
        schedule_sync,
        schedule_video_leading,
    )


def test_import_inference_joint_engine():
    from open_wam.inference.joint_engine import JointInferenceEngine  # noqa: F401


def test_import_inference_init():
    from open_wam.inference import (  # noqa: F401
        BaseInferenceEngine,
        JointInferenceEngine,
        Schedule,
        make_schedule,
    )


def test_import_training_base():
    from open_wam.training.base import BaseTrainer  # noqa: F401

    assert hasattr(BaseTrainer, "compute_loss")
    assert hasattr(BaseTrainer, "train_step")


def test_import_training_loss():
    from open_wam.training.loss import FlowMatchVideoActionSFTLoss  # noqa: F401


def test_import_training_runtime():
    from open_wam.training.runtime import (  # noqa: F401
        build_training_dataset,
        build_validation_datasets,
        cfg_to_flat_namespace,
    )


def test_import_training_optimizer_groups():
    from open_wam.training.optimizer_groups import (  # noqa: F401
        attach_optimizer_groups,
        build_trainable_parameters,
    )


def test_import_training_callbacks():
    from open_wam.training.callbacks import (  # noqa: F401
        CallbackRunner,
        SetupCallback,
        TrainingCallback,
        ValidationLossCallback,
        VideoLogCallback,
    )


def test_import_training_init():
    from open_wam.training import (  # noqa: F401
        BaseTrainer,
        CallbackRunner,
        FlowMatchVideoActionSFTLoss,
        NativeTrainer,
    )


def test_import_evaluation_base():
    from open_wam.evaluation.base import BaseEvaluator  # noqa: F401

    assert hasattr(BaseEvaluator, "evaluate")


def test_import_evaluation_policy():
    from open_wam.evaluation.policy import WAMPolicy  # noqa: F401


def test_import_evaluation_robotwin():
    from open_wam.evaluation.robotwin_evaluator import (  # noqa: F401
        RoboTwinOfflineEvaluator,
        RoboTwinOnlineEvaluator,
    )


def test_import_evaluation_robotwin_policy():
    from open_wam.evaluation.robotwin_policy import (  # noqa: F401
        eval_one_step,
        get_model,
        reset_model,
    )


def test_import_evaluation_envs():
    from open_wam.evaluation.envs.base import BaseEnvAdapter  # noqa: F401
    from open_wam.evaluation.envs.robotwin import RoboTwinEnvAdapter  # noqa: F401


def test_import_evaluation_metrics():
    from open_wam.evaluation.metrics import compute_video_metrics  # noqa: F401


def test_import_evaluation_init():
    from open_wam.evaluation import (  # noqa: F401
        BaseEvaluator,
        RoboTwinOfflineEvaluator,
        RoboTwinOnlineEvaluator,
        WAMPolicy,
    )


def test_import_evaluation_registry():
    from open_wam.evaluation.registry import list_registered_evaluators

    registered = list_registered_evaluators()
    assert "offline" in registered
    assert "online" in registered
    assert "libero" in registered
    assert "robocasa" in registered
    assert "calvin" in registered
    assert "behavior" in registered
    assert "simpler_env" in registered


def test_import_action_repr_registry():
    from open_wam.models.action_repr import list_registered_action_reprs

    registered = list_registered_action_reprs()
    assert "continuous" in registered


def test_import_action_dit():
    from open_wam.models.action_dit import (  # noqa: F401
        ActionDiT,
        ActionDiTState,
        RMSNorm,
        sinusoidal_embedding_1d,
    )


def test_import_moe_expert_dit():
    from open_wam.models.moe_expert_dit import MoEExpertDiT, MoEExpertState  # noqa: F401


def test_import_flow_match_scheduler():
    from open_wam.inference.flow_match_scheduler import FlowMatchScheduler

    s = FlowMatchScheduler("Wan")
    s.set_timesteps(20, shift=5.0)
    assert len(s.timesteps) == 20


def test_import_model_config():
    from open_wam.inference.model_config import ModelConfig  # noqa: F401


def test_import_video_pipeline_wrapper():
    from open_wam.inference.video_pipeline import WanVideoPipeline  # noqa: F401


def test_import_models_backbone():
    from open_wam.models.backbone.base import BaseVideoBackbone  # noqa: F401


def test_import_architecture_registry():
    from open_wam.models.architectures.registry import (
        build_architecture,
        list_supported_architectures,
    )

    supported = list_supported_architectures()
    assert "dual_system" in supported
    assert "moe_expert" in supported
    assert "shared_backbone" in supported

    # Verify build_architecture works for each supported type
    for arch_name in supported:
        arch = build_architecture(arch_name, {})
        assert arch is not None
