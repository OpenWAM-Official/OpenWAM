"""Regression guards: ensure the dispatch state machine is gone for good.

After the refactor that moves each architecture's forward into its own
class, ``ExecutionPlan``, ``RuntimeState``, and ``ActionState.runtime_state``
no longer exist. These tests fail loudly if any of those resurface (e.g.
via a bad merge or a copy-pasted snippet from an older branch).
"""

import pytest


def test_execution_plan_import_fails():
    """``ExecutionPlan`` must not be importable from base or any architecture base."""
    with pytest.raises(ImportError):
        from openwam.model.architectures.base import ExecutionPlan  # noqa: F401
    with pytest.raises(ImportError):
        from openwam.model.architectures.base import ExecutionPlan  # noqa: F401


def test_runtime_state_import_fails():
    """``RuntimeState`` must not be importable from base or any architecture base."""
    with pytest.raises(ImportError):
        from openwam.model.architectures.base import RuntimeState  # noqa: F401
    with pytest.raises(ImportError):
        from openwam.model.architectures.base import RuntimeState  # noqa: F401


def test_action_state_has_no_runtime_state_field():
    """``ActionState`` must expose a flat ``payload`` field — no ``runtime_state`` wrapper."""
    from dataclasses import fields

    from openwam.model.architectures.base import ActionState

    field_names = {f.name for f in fields(ActionState)}
    assert "runtime_state" not in field_names, (
        "ActionState.runtime_state was removed; tests/code reading it should now read .payload."
    )
    assert "payload" in field_names


def test_action_backbone_abc_has_no_5stage_adapter():
    """Neither action-stream ABC may declare 5-stage block-loop adapter methods."""
    from openwam.model.action_backbone.base import ActionDiTBackbone, SharedActionBackbone

    forbidden = ("before_loop", "run_block", "after_loop", "execution_plan")
    for abc in (SharedActionBackbone, ActionDiTBackbone):
        own = vars(abc)
        for name in forbidden:
            assert name not in own, (
                f"{abc.__name__}.{name} should be gone after the decoupling refactor; "
                "each subclass exposes only the methods its architecture's forward calls."
            )


def test_base_architecture_forward_is_abstract():
    """``BaseWAMArchitecture.forward`` must be abstract — no shared dispatch."""
    from openwam.model.architectures.base import BaseWAMArchitecture

    assert "forward" in BaseWAMArchitecture.__abstractmethods__


def test_moe_expert_dit_old_name_gone():
    """Old class name must not be importable (renamed to ``SharedMoEActionBackbone``)."""
    with pytest.raises(ImportError):
        from openwam.model.action_backbone.shared_action_backbone import MoEExpertDiT  # noqa: F401


def test_moe_expert_state_dataclass_gone():
    """``MoEExpertState`` dataclass was deleted along with the 5-stage adapter."""
    with pytest.raises(ImportError):
        from openwam.model.action_backbone.shared_action_backbone import MoEExpertState  # noqa: F401


def test_shared_vanilla_state_dataclass_gone():
    """``SharedVanillaState`` dataclass was deleted along with the 5-stage adapter."""
    with pytest.raises(ImportError):
        from openwam.model.action_backbone.shared_action_backbone import SharedVanillaState  # noqa: F401
