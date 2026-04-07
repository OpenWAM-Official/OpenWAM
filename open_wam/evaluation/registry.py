"""Evaluator registry for config-driven evaluator construction.

Usage:
    @register_evaluator("offline")
    class RoboTwinOfflineEvaluator(BaseEvaluator):
        ...

    evaluator = build_evaluator("offline", cfg, engine)
"""

from typing import Dict, Type

from open_wam.evaluation.base import BaseEvaluator
from open_wam.inference.base import BaseInferenceEngine


EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}


def register_evaluator(name: str):
    """Decorator to register an evaluator class by name."""
    def wrapper(cls):
        if name in EVALUATOR_REGISTRY:
            raise ValueError(f"Evaluator '{name}' already registered")
        EVALUATOR_REGISTRY[name] = cls
        return cls
    return wrapper


def build_evaluator(name: str, cfg, engine: BaseInferenceEngine) -> BaseEvaluator:
    """Instantiate a registered evaluator by name.

    Args:
        name: Registry key (e.g. "offline", "online", "libero").
        cfg: Hydra config.
        engine: Inference engine instance.

    Returns:
        Instantiated BaseEvaluator subclass.
    """
    if name not in EVALUATOR_REGISTRY:
        available = ", ".join(sorted(EVALUATOR_REGISTRY.keys()))
        raise KeyError(
            f"Unknown evaluator type '{name}'. Available: {available}"
        )
    return EVALUATOR_REGISTRY[name](cfg=cfg, engine=engine)


def list_registered_evaluators() -> list[str]:
    """Return list of registered evaluator type names."""
    return sorted(EVALUATOR_REGISTRY.keys())


# ---- Auto-registration of built-in evaluators ----

def _register_builtins():
    from open_wam.evaluation.robotwin_evaluator import (
        RoboTwinOfflineEvaluator,
        RoboTwinOnlineEvaluator,
    )
    from open_wam.evaluation.simpler_env_evaluator import SimplerEnvEvaluator
    from open_wam.evaluation.libero_evaluator import LIBEROEvaluator
    from open_wam.evaluation.robocasa_evaluator import RoboCasaEvaluator
    from open_wam.evaluation.calvin_evaluator import CalvinEvaluator
    from open_wam.evaluation.behavior_evaluator import BehaviorEvaluator

    register_evaluator("offline")(RoboTwinOfflineEvaluator)
    register_evaluator("online")(RoboTwinOnlineEvaluator)
    register_evaluator("simpler_env")(SimplerEnvEvaluator)
    register_evaluator("libero")(LIBEROEvaluator)
    register_evaluator("robocasa")(RoboCasaEvaluator)
    register_evaluator("calvin")(CalvinEvaluator)
    register_evaluator("behavior")(BehaviorEvaluator)


_register_builtins()
