"""Representation-contract sourcing (PR #57): the PONG must advertise the TRAINING truth.

``build_server_from_config`` merges deploy overrides ON TOP of the training cfg (deploy wins), but
the architecture's normalizer / binary dims / torso masking were all built from the training cfg
BEFORE that merge. The contract therefore binds to ``architecture.repr_contract`` (set by
``load_model``) — a deploy yaml carrying stray ``dataloader.*`` keys must not be able to make the
PONG advertise values the architecture isn't using.
"""

from types import SimpleNamespace

from omegaconf import OmegaConf

from openwam.deploy.model_loader import repr_contract_from_cfg
from openwam.deploy.server import PolicyServer

TRAINING_DL = {
    "base_proprio": "global_pose",
    "mobile_base": True,
    "mask_torso_action": False,
    "binary_action_dims": [24],
    "gripper_convention": "pretrain",
}


def test_repr_contract_from_cfg_reads_training_values():
    cfg = OmegaConf.create({"dataloader": dict(TRAINING_DL)})
    c = repr_contract_from_cfg(cfg)
    assert c == {
        "base_proprio": "global_pose",
        "mobile_base": True,
        "mask_torso_action": False,
        "binary_action_dims": [24],
        "gripper_convention": "pretrain",
    }


def test_repr_contract_defaults_are_historical():
    c = repr_contract_from_cfg(OmegaConf.create({"dataloader": {}}))
    assert c == {
        "base_proprio": "velocity",
        "mobile_base": False,
        "mask_torso_action": True,
        "binary_action_dims": [],
        "gripper_convention": None,  # pre-marker ckpt: clients must refuse (old gripper convention)
    }


def test_pong_contract_survives_malicious_deploy_overrides():
    """wayrise's repro: ckpt = global_pose/mobile/[9,24], deploy yaml overrides dataloader to
    velocity/fixed/[] — the merged self.cfg lies, the PONG must not."""
    training_cfg = OmegaConf.create({"dataloader": dict(TRAINING_DL)})
    contract = repr_contract_from_cfg(training_cfg)
    engine = SimpleNamespace(architecture=SimpleNamespace(repr_contract=contract))
    merged = OmegaConf.merge(
        training_cfg,
        OmegaConf.create({"dataloader": {
            "base_proprio": "velocity", "mobile_base": False, "mask_torso_action": True,
            "binary_action_dims": [],
        }}),
    )
    server = PolicyServer(engine=engine, cfg=merged)
    pong = server._ckpt_contract()
    assert pong["base_proprio"] == "global_pose"
    assert pong["mobile_base"] is True
    assert pong["mask_torso_action"] is False
    assert pong["binary_action_dims"] == [24]
    assert pong["gripper_convention"] == "pretrain"


def test_pong_contract_empty_without_architecture():
    """Engines without an architecture (or pre-contract paths) advertise nothing — clients apply
    their old-server compatibility rules instead of trusting a fabricated default."""
    server = PolicyServer(engine=SimpleNamespace(architecture=None), cfg=OmegaConf.create({}))
    assert server._ckpt_contract() == {}
