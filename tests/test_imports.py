"""Verify that all open_wam modules can be imported successfully."""


def test_import_openwam():
    import openwam  # noqa: F401

    assert hasattr(openwam, "__version__")


def test_import_data_base():
    from openwam.dataloader.base_dataset import BaseActionDataset  # noqa: F401

    assert hasattr(BaseActionDataset, "__getitem__")
    assert hasattr(BaseActionDataset, "action_dim")
    assert hasattr(BaseActionDataset, "action_stats")


def test_import_data_robotwin():
    from openwam.dataloader.robotwin_dataset import (  # noqa: F401
        ROBOTWIN_ALL_TASKS,
        ROBOTWIN_HOLDOUT_TASKS,
        ROBOTWIN_TRAIN_TASKS,
    )

    assert len(ROBOTWIN_TRAIN_TASKS) == 50  # All tasks in training by default
    assert len(ROBOTWIN_HOLDOUT_TASKS) == 0
    assert len(ROBOTWIN_ALL_TASKS) == 50


def test_import_data_transforms():
    from openwam.dataloader.transforms import RotationTransform, build_transforms  # noqa: F401


def test_import_data_action_stats():
    from openwam.dataloader.robotwin_stats_computation import (  # noqa: F401
        compute_action_stats,
        compute_multitask_robotwin_stats,
        parse_tasks_file,
    )


def test_import_data_init():
    from openwam.dataloader import (  # noqa: F401
        BaseActionDataset,
        MultiTaskRoboTwinDataset,
        RoboTwinDataset,
    )


def test_import_inference_base():
    from openwam.deployment.base import BaseInferenceEngine  # noqa: F401

    assert hasattr(BaseInferenceEngine, "generate")


def test_import_inference_schedule():
    from openwam.deployment.schedule import (  # noqa: F401
        Schedule,
        make_schedule,
        schedule_action_only,
        schedule_cascade,
        schedule_sync,
        schedule_video_leading,
    )


def test_import_inference_joint_engine():
    from openwam.deployment.joint_engine import JointInferenceEngine  # noqa: F401


def test_import_inference_init():
    from openwam.deployment import (  # noqa: F401
        BaseInferenceEngine,
        JointInferenceEngine,
        Schedule,
        make_schedule,
    )


def test_import_training_base():
    from openwam.train.base import BaseTrainer  # noqa: F401

    assert hasattr(BaseTrainer, "compute_loss")
    assert hasattr(BaseTrainer, "train_step")


def test_import_training_loss():
    from openwam.train.loss.sft_loss import FlowMatchVideoActionSFTLoss  # noqa: F401


def test_import_training_runtime():
    from openwam.train.runtime import (  # noqa: F401
        build_training_dataset,
        build_validation_datasets,
        cfg_to_flat_namespace,
    )


def test_import_training_optimizer_groups():
    from openwam.train.optimizer_groups import (  # noqa: F401
        attach_optimizer_groups,
        build_trainable_parameters,
    )


def test_import_training_callbacks():
    from openwam.train.callbacks import (  # noqa: F401
        CallbackRunner,
        SetupCallback,
        TrainingCallback,
        ValidationLossCallback,
        VideoLogCallback,
    )


def test_import_training_init():
    from openwam.train import (  # noqa: F401
        BaseTrainer,
        CallbackRunner,
        FlowMatchVideoActionSFTLoss,
        NativeTrainer,
    )


def test_import_action_repr_registry():
    from openwam.model.action_model.action_repr import list_registered_action_reprs

    registered = list_registered_action_reprs()
    assert "continuous" in registered


def test_import_action_dit():
    from openwam.model.action_model.action_dit import (  # noqa: F401
        ActionDiT,
        ActionDiTState,
        RMSNorm,
        sinusoidal_embedding_1d,
    )


def test_import_moe_expert_dit():
    from openwam.model.action_model.moe_expert_dit import MoEExpertDiT, MoEExpertState  # noqa: F401


def test_import_flow_match_scheduler():
    from openwam.deployment.flow_match_scheduler import FlowMatchScheduler

    s = FlowMatchScheduler("Wan")
    s.set_timesteps(20, shift=5.0)
    assert len(s.timesteps) == 20


def test_import_model_config():
    from openwam.deployment.model_config import ModelConfig  # noqa: F401


def test_import_video_pipeline_wrapper():
    from openwam.model.video_model.video_pipeline import WanVideoPipeline  # noqa: F401


def test_import_models_backbone():
    from openwam.model.backbone.base import BaseVideoBackbone  # noqa: F401


def test_import_architecture_registry():
    from openwam.model.registry import (
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
