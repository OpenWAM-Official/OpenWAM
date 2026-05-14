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
    from openwam.deploy.base import BaseInferenceEngine  # noqa: F401

    assert hasattr(BaseInferenceEngine, "generate")


def test_import_inference_schedule():
    from openwam.deploy.schedule import (  # noqa: F401
        Schedule,
        make_schedule,
        schedule_action_only,
        schedule_cascade,
        schedule_sync,
        schedule_video_leading,
    )


def test_import_inference_joint_engine():
    from openwam.deploy.joint_engine import JointInferenceEngine  # noqa: F401


def test_import_inference_init():
    from openwam.deploy import (  # noqa: F401
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
    from openwam.train.loss.decoupled_loss import DecoupledFlowMatchLoss  # noqa: F401


def test_import_training_optimizer_groups():
    from openwam.train.utils.optimizer_groups import (  # noqa: F401
        attach_optimizer_groups,
        build_trainable_parameters,
    )


def test_import_training_init():
    from openwam.train import (  # noqa: F401
        BaseTrainer,
        DecoupledFlowMatchLoss,
        OpenWAMTrainer,
    )


def test_import_action_dit():
    from openwam.model.action_backbone.dualsystem_dit import (  # noqa: F401
        ActionDiT,
        ActionDiTState,
    )


def test_import_moe_action_backbone():
    from openwam.model.action_backbone.shared_moe import ExpertFFNBlock, SharedMoEActionBackbone  # noqa: F401


def test_import_action_scheduler():
    from openwam.model.action_backbone.scheduler import ActionScheduler

    s = ActionScheduler()
    s.set_timesteps(20, shift=5.0)
    assert len(s.timesteps) == 20


def test_import_model_config():
    from openwam.model.video_backbone.wan.shared.core.loader import ModelConfig  # noqa: F401


def test_import_video_backbone():
    from openwam.model.video_backbone.wan.pipeline import WanVideoPipeline  # noqa: F401


def test_import_architecture_registry():
    from openwam.model import build_architecture, list_supported_architectures

    supported = list_supported_architectures()
    assert "dual_system_cross_attn" in supported
    assert "dual_system_self_attn" in supported
    assert "dual_system_idm" in supported
    assert "shared_backbone_vanilla" in supported
    assert "shared_backbone_moe" in supported

    configs = {
        "dual_system_cross_attn": {
            "framework": "dual_system",
            "variant": "joint_cross_attn",
            "detach_bridge": True,
            "bridge_layers": (0, 1),
            "action_dim": 7,
            "dim": 64,
            "ffn_dim": 128,
            "num_heads": 4,
            "video_dim": 128,
        },
        "dual_system_self_attn": {
            "framework": "dual_system",
            "variant": "joint_self_attn",
            "bridge_layers": (0, 1),
            "action_dim": 7,
            # joint_self_attn requires action dim == video_dim (the MoT driver
            # runs a single mixed attention with no inter-modality projection).
            "dim": 128,
            "ffn_dim": 256,
            "num_heads": 4,
            "video_dim": 128,
        },
        "dual_system_idm": {
            "framework": "dual_system",
            "variant": "idm",
            "bridge_layers": (0, 1),
            "action_dim": 7,
            "dim": 128,
            "ffn_dim": 256,
            "num_heads": 4,
            "video_dim": 128,
            "idm_video_cond_noise_prob": 0.5,
        },
        "shared_backbone_moe": {
            "framework": "shared_backbone",
            "variant": "moe",
            "action_dim": 7,
            "video_dim": 128,
            "expert_ffn_dim": 256,
            "expert_layers": (0, 1),
        },
        "shared_backbone_vanilla": {
            "framework": "shared_backbone",
            "variant": "vanilla",
            "action_dim": 7,
            "video_dim": 128,
        },
    }

    for arch_name in supported:
        arch = build_architecture(arch_name, configs[arch_name])
        assert arch is not None
