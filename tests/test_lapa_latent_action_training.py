import pytest
import torch
from omegaconf import OmegaConf


class _Resolved:
    registry_name = "dual_system_cross_attn"
    params = {}
    canonical = type("Canonical", (), {"framework": "dual_system", "variant": "joint_cross_attn"})()


class _Arch(torch.nn.Module):
    def __init__(self, action_dim=1024, use_proprio=False):
        super().__init__()
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.action_dim = action_dim
        self.uses_proprioception = use_proprio
        self.video_backbone = None
        self.action_backbone = torch.nn.Linear(1, 1)
        self._runtime = {}
        self.last_prepare_batch = None
        self.last_actions = None

    @property
    def backbones(self):
        return {}

    def set_dtype_device(self, dtype, device):  # noqa: ARG002
        return None

    def freeze_modules(self, names):  # noqa: ARG002
        return []

    def init_training_schedulers(self, num_timesteps=1000):  # noqa: ARG002
        return None

    def set_training_runtime(self, **kwargs):
        self._runtime.update(kwargs)

    def prepare_inputs(self, batch):
        self.last_prepare_batch = batch
        return {"input_latents": torch.zeros(len(batch), 1), "actions": None}

    def compute_loss(self, **kwargs):
        self.last_actions = kwargs["actions"]
        loss_video = torch.tensor(2.0)
        loss_action = torch.tensor(3.0)
        return {
            "loss": loss_video + loss_action,
            "loss_video": loss_video,
            "loss_action": loss_action,
        }

    def get_trainable_modules(self, freeze_list=()):  # noqa: ARG002
        return {}


class _FakeProvider:
    def __init__(self, cfg, *, device, dtype):  # noqa: ARG002
        self.calls = []

    def __call__(self, videos):
        self.calls.append(videos)
        return torch.ones(len(videos), 32, 1024)


def _cfg(*, enabled=True, action_dim=1024, use_proprio=False):
    if enabled:
        action_backbone = {
            "type": "latent",
            "latent_encoder": {
                "name": "lapa_dinov3",
                "output": {"action_dim": 1024, "tokens_per_pair": 16, "token_dim": 1024},
            },
        }
    else:
        action_backbone = {"type": "explicit"}
    cfg = {
        "training": {
            "initialize_model_on_cpu": False,
            "action_timestep_per_token": False,
            "use_gradient_checkpointing": False,
            "use_gradient_checkpointing_offload": False,
            "max_timestep_boundary": 1.0,
            "min_timestep_boundary": 0.0,
            "lambda_video": 1.0,
            "lambda_action": 1.0,
        },
        "model": {
            "architecture": {"action_dim": action_dim, "use_proprioception": use_proprio},
            "action_backbone": action_backbone,
            "freeze": [],
        },
        "project": {"seed": 1},
    }
    return OmegaConf.create(cfg)


def _patch_trainer(monkeypatch, *, arch):
    monkeypatch.setattr("openwam.model.resolve_architecture_config", lambda _m: _Resolved())
    monkeypatch.setattr("openwam.model.build_architecture", lambda _name, _params: arch)
    monkeypatch.setattr("openwam.model.action_backbone.latent_encoder.build_latent_action_provider", _FakeProvider)


def test_latent_action_trainer_allows_missing_dataset_action(monkeypatch):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    arch = _Arch(action_dim=1024)
    _patch_trainer(monkeypatch, arch=arch)
    trainer = OpenWAMTrainer(_cfg(enabled=True), accelerator=None, dataset=None)
    batch = [{"video": [torch.zeros(4, 4, 3), torch.ones(4, 4, 3)], "prompt": "move"}]

    result = trainer.compute_loss(batch)

    assert result["total"].item() == 5.0
    assert arch.last_actions.shape == (1, 32, 1024)
    assert "action" not in arch.last_prepare_batch[0]
    assert trainer.latent_action_provider.calls[0] == [batch[0]["video"]]


def test_non_latent_action_path_still_requires_action(monkeypatch):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    arch = _Arch(action_dim=20)
    _patch_trainer(monkeypatch, arch=arch)
    trainer = OpenWAMTrainer(_cfg(enabled=False, action_dim=20), accelerator=None, dataset=None)

    with pytest.raises(ValueError, match="lambda_action > 0 but no action"):
        trainer.compute_loss([{"video": [torch.zeros(4, 4, 3), torch.ones(4, 4, 3)], "prompt": "move"}])


def test_latent_action_rejects_proprioception(monkeypatch):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    arch = _Arch(action_dim=1024, use_proprio=True)
    _patch_trainer(monkeypatch, arch=arch)

    with pytest.raises(ValueError, match="use_proprioception=false"):
        OpenWAMTrainer(_cfg(enabled=True, use_proprio=True), accelerator=None, dataset=None)


def test_latent_action_rejects_action_dim_mismatch(monkeypatch):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    arch = _Arch(action_dim=20)
    _patch_trainer(monkeypatch, arch=arch)

    with pytest.raises(ValueError, match="does not match"):
        OpenWAMTrainer(_cfg(enabled=True, action_dim=20), accelerator=None, dataset=None)


def test_latent_action_rejects_token_dim_mismatch(monkeypatch):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    arch = _Arch(action_dim=1024)
    _patch_trainer(monkeypatch, arch=arch)
    cfg = _cfg(enabled=True)
    cfg.model.action_backbone.latent_encoder.output.token_dim = 512

    with pytest.raises(ValueError, match="token_dim=512"):
        OpenWAMTrainer(cfg, accelerator=None, dataset=None)
