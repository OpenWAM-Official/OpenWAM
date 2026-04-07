"""BEHAVIOR-1K benchmark evaluator.

Evaluates across BEHAVIOR-1K everyday household activities using OmniGibson,
with per-task success rate and optional partial completion scoring.
"""

import logging

import numpy as np

from open_wam.evaluation.base import BaseEvaluator
from open_wam.evaluation.policy import WAMPolicy
from open_wam.inference.base import BaseInferenceEngine

logger = logging.getLogger(__name__)


# Representative BEHAVIOR-1K task subset for evaluation
BEHAVIOR_TASKS = [
    "cleaning_up_the_kitchen_only",
    "putting_away_dishes_after_cleaning",
    "setting_up_candles",
    "laying_wood_floors",
    "assembling_gift_baskets",
    "packing_lunches",
    "sorting_books",
    "watering_houseplants",
    "cleaning_bathrooms",
    "organizing_school_stuff",
]

# Task categories for structured reporting
BEHAVIOR_CATEGORIES = {
    "kitchen": [
        "cleaning_up_the_kitchen_only",
        "putting_away_dishes_after_cleaning",
        "packing_lunches",
    ],
    "organization": [
        "sorting_books",
        "organizing_school_stuff",
        "assembling_gift_baskets",
    ],
    "maintenance": [
        "laying_wood_floors",
        "watering_houseplants",
        "cleaning_bathrooms",
        "setting_up_candles",
    ],
}


class BehaviorEvaluator(BaseEvaluator):
    """Online evaluation on BEHAVIOR-1K benchmark.

    Runs closed-loop evaluation across BEHAVIOR-1K activities,
    reporting per-task, per-category, and aggregate success rates.

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
            dict with per-task success rates, per-category aggregates,
            and overall metrics.
        """
        eval_cfg = self.cfg.eval
        num_episodes = getattr(eval_cfg, "num_episodes", 10)
        max_steps = getattr(eval_cfg, "max_steps_per_episode", 1000)
        tasks = getattr(eval_cfg, "tasks", None)

        if env_adapter is not None:
            return self._evaluate_single(env_adapter, num_episodes, max_steps)

        # Multi-task mode
        if tasks is None:
            tasks = BEHAVIOR_TASKS

        from open_wam.evaluation.envs.behavior import BehaviorEnvAdapter

        all_results = {}
        total_success = 0
        total_episodes = 0

        for task_name in tasks:
            logger.info("Evaluating BEHAVIOR-1K task: %s", task_name)
            adapter = BehaviorEnvAdapter(
                activity_name=task_name,
                scene_model=getattr(eval_cfg, "scene_model", "Rs_int"),
                robot=getattr(eval_cfg, "robot", "Fetch"),
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

        # Per-category aggregation
        per_category = {}
        for cat_name, cat_tasks in BEHAVIOR_CATEGORIES.items():
            cat_success = sum(all_results[t]["successes"] for t in cat_tasks if t in all_results)
            cat_episodes = sum(all_results[t]["num_episodes"] for t in cat_tasks if t in all_results)
            per_category[cat_name] = {
                "success_rate": cat_success / cat_episodes if cat_episodes > 0 else 0.0,
                "successes": cat_success,
                "num_episodes": cat_episodes,
            }

        return {
            "success_rate": total_success / total_episodes if total_episodes > 0 else 0.0,
            "total_episodes": total_episodes,
            "total_successes": total_success,
            "per_task": all_results,
            "per_category": per_category,
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
