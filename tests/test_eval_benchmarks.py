"""Tests for expanded evaluation benchmarks (SimplerEnv + LIBERO)."""

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from open_wam.evaluation.envs.base import BaseEnvAdapter


def test_simpler_env_adapter_imports():
    """SimplerEnvAdapter should be importable and inherit BaseEnvAdapter."""
    from open_wam.evaluation.envs import SimplerEnvAdapter

    assert issubclass(SimplerEnvAdapter, BaseEnvAdapter)


def test_libero_env_adapter_imports():
    """LIBEROEnvAdapter should be importable and inherit BaseEnvAdapter."""
    from open_wam.evaluation.envs import LIBEROEnvAdapter

    assert issubclass(LIBEROEnvAdapter, BaseEnvAdapter)


def test_libero_task_suites():
    """LIBERO task suite constants should be populated."""
    from open_wam.evaluation.envs.libero import (
        LIBERO_GOAL_TASKS,
        LIBERO_OBJECT_TASKS,
        LIBERO_SPATIAL_TASKS,
        LIBERO_SUITES,
    )

    assert len(LIBERO_SPATIAL_TASKS) == 10
    assert len(LIBERO_OBJECT_TASKS) == 10
    assert len(LIBERO_GOAL_TASKS) == 10
    assert set(LIBERO_SUITES.keys()) == {"libero_spatial", "libero_object", "libero_goal"}


def test_simpler_env_evaluator_imports():
    """SimplerEnvEvaluator should be importable from evaluation package."""
    from open_wam.evaluation import SimplerEnvEvaluator
    from open_wam.evaluation.base import BaseEvaluator

    assert issubclass(SimplerEnvEvaluator, BaseEvaluator)


def test_libero_evaluator_imports():
    """LIBEROEvaluator should be importable from evaluation package."""
    from open_wam.evaluation import LIBEROEvaluator
    from open_wam.evaluation.base import BaseEvaluator

    assert issubclass(LIBEROEvaluator, BaseEvaluator)


def test_simpler_env_task_lists():
    """SimplerEnv evaluator should have standard task lists."""
    from open_wam.evaluation.simpler_env_evaluator import (
        GOOGLE_ROBOT_TASKS,
        WIDOWX_TASKS,
    )

    assert len(GOOGLE_ROBOT_TASKS) >= 3
    assert len(WIDOWX_TASKS) >= 3
    assert all("google_robot" in t for t in GOOGLE_ROBOT_TASKS)
    assert all("widowx" in t for t in WIDOWX_TASKS)


class MockEnvAdapter(BaseEnvAdapter):
    """Mock env adapter for testing evaluators."""

    def __init__(self, max_steps: int = 5, success_at: int = 3):
        self._step = 0
        self._max_steps = max_steps
        self._success_at = success_at

    def reset(self):
        self._step = 0
        return {"image": None, "step": 0}

    def step(self, action):
        self._step += 1
        done = self._step >= self._max_steps
        success = self._step >= self._success_at
        return (
            {"image": None, "step": self._step},
            1.0 if success else 0.0,
            done,
            {"success": success and done},
        )

    def get_obs(self):
        return {"image": None, "step": self._step}


class MockEngine:
    def generate(self, conditions):
        return {"actions": np.zeros((10, 7), dtype=np.float32), "video": None}


def test_simpler_env_evaluator_single_env():
    """SimplerEnvEvaluator should work with a single mock env."""
    from open_wam.evaluation.simpler_env_evaluator import SimplerEnvEvaluator

    cfg = SimpleNamespace(
        eval=SimpleNamespace(
            num_episodes=3,
            max_steps_per_episode=5,
            policy=SimpleNamespace(history_len=2, execute_horizon=None),
        )
    )
    evaluator = SimplerEnvEvaluator(cfg, MockEngine())
    result = evaluator.evaluate(MockEnvAdapter())

    assert "success_rate" in result
    assert "num_episodes" in result
    assert result["num_episodes"] == 3


def test_libero_evaluator_single_env():
    """LIBEROEvaluator should work with a single mock env."""
    from open_wam.evaluation.libero_evaluator import LIBEROEvaluator

    cfg = SimpleNamespace(
        eval=SimpleNamespace(
            num_episodes=2,
            max_steps_per_episode=5,
            policy=SimpleNamespace(history_len=2, execute_horizon=None),
        )
    )
    evaluator = LIBEROEvaluator(cfg, MockEngine())
    result = evaluator.evaluate(MockEnvAdapter())

    assert "success_rate" in result
    assert "num_episodes" in result
    assert result["num_episodes"] == 2


def test_libero_adapter_language_instruction():
    """LIBEROEnvAdapter should return readable task instruction."""
    from open_wam.evaluation.envs.libero import LIBEROEnvAdapter

    adapter = LIBEROEnvAdapter(task_name="pick_up_the_black_bowl")
    instruction = adapter.get_language_instruction()
    assert instruction == "pick up the black bowl"


def test_simpler_env_adapter_language_instruction():
    """SimplerEnvAdapter should return readable task instruction."""
    from open_wam.evaluation.envs.simpler_env import SimplerEnvAdapter

    adapter = SimplerEnvAdapter(env_name="google_robot_pick_coke_can")
    instruction = adapter.get_language_instruction()
    assert instruction == "google robot pick coke can"


def test_all_env_adapters_in_envs_init():
    """All env adapters should be importable from envs package."""
    from open_wam.evaluation.envs import (
        BaseEnvAdapter,
        LIBEROEnvAdapter,
        RoboTwinEnvAdapter,
        SimplerEnvAdapter,
    )

    for cls in [RoboTwinEnvAdapter, SimplerEnvAdapter, LIBEROEnvAdapter]:
        assert issubclass(cls, BaseEnvAdapter)


def test_simpler_env_config_shape():
    """SimplerEnv config should merge directly under cfg.eval."""
    cfg = OmegaConf.load("configs/eval/simpler_env.yaml")
    assert cfg.type == "simpler_env"
    assert cfg.robot == "google_robot"
    assert cfg.policy.execute_horizon == 8


def test_libero_config_shape():
    """LIBERO config should expose task suites at the top eval level."""
    cfg = OmegaConf.load("configs/eval/libero.yaml")
    assert cfg.type == "libero"
    assert cfg.task_suites[0] == "libero_spatial"
