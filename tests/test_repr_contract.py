"""Representation contract must advertise checkpoint training truth."""

from types import SimpleNamespace

from omegaconf import OmegaConf

from openwam.deploy.model_loader import repr_contract_from_cfg
from openwam.deploy.server import PolicyServer

TRAINING_DL = {
    "action_mode": "robocasa365",
}


def test_repr_contract_from_cfg_reads_compact_values():
    contract = repr_contract_from_cfg(OmegaConf.create({"dataloader": TRAINING_DL}))
    assert contract == {"representation": "robocasa365"}


def test_repr_contract_defaults():
    contract = repr_contract_from_cfg(OmegaConf.create({"dataloader": {}}))
    assert contract == {"representation": "joint"}


def test_repr_contract_preserves_optional_legacy_markers():
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "action_mode": "legacy",
                "binary_action_dims": [3],
                "gripper_convention": "minus1_closed_plus1_open",
            }
        }
    )
    assert repr_contract_from_cfg(cfg) == {
        "representation": "legacy",
        "binary_action_dims": [3],
        "gripper_convention": "minus1_closed_plus1_open",
    }


def test_pong_contract_uses_architecture_not_deploy_override():
    training_cfg = OmegaConf.create({"dataloader": TRAINING_DL})
    contract = repr_contract_from_cfg(training_cfg)
    engine = SimpleNamespace(architecture=SimpleNamespace(repr_contract=contract))
    merged = OmegaConf.merge(
        training_cfg,
        OmegaConf.create({"dataloader": {"action_mode": "joint"}}),
    )
    pong = PolicyServer(engine=engine, cfg=merged)._ckpt_contract()
    assert pong == contract


def test_pong_contract_empty_without_architecture():
    server = PolicyServer(engine=SimpleNamespace(architecture=None), cfg=OmegaConf.create({}))
    assert server._ckpt_contract() == {}
