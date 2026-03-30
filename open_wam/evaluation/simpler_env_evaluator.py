"""SimplerEnv evaluator for Google Robot and WidowX tasks."""

import logging
from typing import Optional, List

import numpy as np

from open_wam.evaluation.base import BaseEvaluator
from open_wam.evaluation.policy import WAMPolicy
from open_wam.inference.base import BaseInferenceEngine

logger = logging.getLogger(__name__)

# Standard SimplerEnv task groups
GOOGLE_ROBOT_TASKS = [
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_open_top_drawer",
    "google_robot_close_top_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
]

WIDOWX_TASKS = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]


class SimplerEnvEvaluator(BaseEvaluator):
    """Online evaluation using SimplerEnv simulation.

    Runs closed-loop evaluation across one or more SimplerEnv tasks,
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
        max_steps = getattr(eval_cfg, "max_steps_per_episode", 200)
        tasks = getattr(eval_cfg, "tasks", None)

        if env_adapter is not None:
            # Single-env mode
            result = self._evaluate_single(env_adapter, num_episodes, max_steps)
            return result

        # Multi-task mode
        if tasks is None:
            robot = getattr(eval_cfg, "robot", "google_robot")
            tasks = GOOGLE_ROBOT_TASKS if robot == "google_robot" else WIDOWX_TASKS

        from open_wam.evaluation.envs.simpler_env import SimplerEnvAdapter

        all_results = {}
        total_success = 0
        total_episodes = 0

        for task_name in tasks:
            logger.info("Evaluating SimplerEnv task: %s", task_name)
            robot = "google_robot" if task_name.startswith("google_robot") else "widowx"
            adapter = SimplerEnvAdapter(
                env_name=task_name,
                robot=robot,
                image_height=getattr(eval_cfg, "image_height", 256),
                image_width=getattr(eval_cfg, "image_width", 256),
                max_episode_steps=max_steps,
            )
            try:
                result = self._evaluate_single(adapter, num_episodes, max_steps)
                all_results[task_name] = result
                total_success += result["successes"]
                total_episodes += result["num_episodes"]
            finally:
                adapter.close()

        aggregate = {
            "success_rate": total_success / total_episodes if total_episodes > 0 else 0.0,
            "total_episodes": total_episodes,
            "total_successes": total_success,
            "per_task": all_results,
            "num_tasks": len(tasks),
        }
        return aggregate

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
