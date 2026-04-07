"""RoboCasa benchmark evaluator.

Evaluates across RoboCasa kitchen manipulation tasks with
per-task and aggregate success rate reporting.
"""

import logging

import numpy as np

from open_wam.evaluation.base import BaseEvaluator
from open_wam.evaluation.policy import WAMPolicy
from open_wam.inference.base import BaseInferenceEngine

logger = logging.getLogger(__name__)


# RoboCasa task categories
ROBOCASA_TASKS = [
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter",
    "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor",
    "CloseSingleDoor",
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnOnStove",
    "TurnOffStove",
    "ArrangeVegetables",
    "MicrowaveThawing",
    "RestockPantry",
    "PreSoakPan",
    "PrepareCoffee",
]


class RoboCasaEvaluator(BaseEvaluator):
    """Online evaluation on RoboCasa benchmark tasks.

    Runs closed-loop evaluation across one or more RoboCasa tasks,
    reporting per-task and aggregate success rates.

    Args:
        cfg: Hydra config (must contain ``cfg.eval``).
        engine: Inference engine for action generation.
    """

    def __init__(self, cfg, engine: BaseInferenceEngine):
        super().__init__(cfg, engine)

    def evaluate(self, env_adapter=None) -> dict:
        """Run evaluation across configured tasks.

        If ``env_adapter`` is provided, evaluates on that single environment.
        Otherwise, creates adapters for all tasks listed in config.

        Returns:
            dict with per-task success rates and aggregate metrics.
        """
        eval_cfg = self.cfg.eval
        num_episodes = getattr(eval_cfg, "num_episodes", 20)
        max_steps = getattr(eval_cfg, "max_steps_per_episode", 500)
        tasks = getattr(eval_cfg, "tasks", None)

        if env_adapter is not None:
            return self._evaluate_single(env_adapter, num_episodes, max_steps)

        # Multi-task mode
        if tasks is None:
            tasks = ROBOCASA_TASKS

        from open_wam.evaluation.envs.robocasa import RoboCasaEnvAdapter

        all_results = {}
        total_success = 0
        total_episodes = 0

        for task_name in tasks:
            logger.info("Evaluating RoboCasa task: %s", task_name)
            adapter = RoboCasaEnvAdapter(
                task_name=task_name,
                layout=getattr(eval_cfg, "layout", 0),
                style=getattr(eval_cfg, "style", 0),
                image_height=getattr(eval_cfg, "image_height", 256),
                image_width=getattr(eval_cfg, "image_width", 256),
                max_episode_steps=max_steps,
                seed=getattr(eval_cfg, "seed", 42),
            )
            try:
                result = self._evaluate_single(adapter, num_episodes, max_steps)
                all_results[task_name] = result
                total_success += result["successes"]
                total_episodes += result["num_episodes"]
            finally:
                adapter.close()

        return {
            "success_rate": total_success / total_episodes if total_episodes > 0 else 0.0,
            "total_episodes": total_episodes,
            "total_successes": total_success,
            "per_task": all_results,
            "num_tasks": len(tasks),
        }

    def _evaluate_single(self, env_adapter, num_episodes: int, max_steps: int) -> dict:
        """Evaluate on a single environment."""
        policy_cfg = getattr(self.cfg.eval, "policy", self.cfg.eval)
        policy = WAMPolicy(engine=self.engine, cfg=policy_cfg)

        successes = 0
        total_rewards = []

        for ep in range(num_episodes):
            obs = env_adapter.reset()
            policy.reset()
            ep_reward = 0.0

            for step in range(max_steps):
                action = policy.predict_action(obs)
                obs, reward, done, info = env_adapter.step(action)
                ep_reward += reward
                if done:
                    if info.get("success", False):
                        successes += 1
                    break

            total_rewards.append(ep_reward)

        return {
            "success_rate": successes / num_episodes if num_episodes > 0 else 0.0,
            "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
            "successes": successes,
            "num_episodes": num_episodes,
        }
