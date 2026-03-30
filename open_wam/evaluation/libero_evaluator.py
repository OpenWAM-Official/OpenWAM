"""LIBERO benchmark evaluator.

Evaluates across LIBERO task suites (Spatial, Object, Goal) with
per-task and aggregate success rate reporting.
"""

import logging
from typing import Optional, List

import numpy as np

from open_wam.evaluation.base import BaseEvaluator
from open_wam.evaluation.policy import WAMPolicy
from open_wam.inference.base import BaseInferenceEngine

logger = logging.getLogger(__name__)


class LIBEROEvaluator(BaseEvaluator):
    """Online evaluation on LIBERO benchmark tasks.

    Runs closed-loop evaluation across one or more LIBERO task suites,
    reporting per-task and per-suite success rates.

    Args:
        cfg: Hydra config (must contain ``cfg.eval``).
        engine: Inference engine for action generation.
    """

    def __init__(self, cfg, engine: BaseInferenceEngine):
        super().__init__(cfg, engine)

    def evaluate(self, env_adapter=None) -> dict:
        """Run evaluation across configured LIBERO tasks.

        If ``env_adapter`` is provided, evaluates on that single environment.
        Otherwise, evaluates all tasks in the configured suite(s).

        Returns:
            dict with per-task success rates and aggregate metrics.
        """
        eval_cfg = self.cfg.eval
        num_episodes = getattr(eval_cfg, "num_episodes", 20)
        max_steps = getattr(eval_cfg, "max_steps_per_episode", 300)

        if env_adapter is not None:
            return self._evaluate_single(env_adapter, num_episodes, max_steps)

        # Multi-task suite evaluation
        from open_wam.evaluation.envs.libero import LIBEROEnvAdapter, LIBERO_SUITES

        suites = getattr(eval_cfg, "task_suites", ["libero_spatial"])
        if isinstance(suites, str):
            suites = [suites]

        all_suite_results = {}
        grand_total_success = 0
        grand_total_episodes = 0

        for suite_name in suites:
            tasks = LIBERO_SUITES.get(suite_name, [])
            if not tasks:
                logger.warning("Unknown LIBERO suite: %s, skipping", suite_name)
                continue

            suite_results = {}
            suite_success = 0
            suite_episodes = 0

            for task_name in tasks:
                logger.info("Evaluating LIBERO %s / %s", suite_name, task_name)
                adapter = LIBEROEnvAdapter(
                    task_name=task_name,
                    task_suite=suite_name,
                    image_height=getattr(eval_cfg, "image_height", 256),
                    image_width=getattr(eval_cfg, "image_width", 256),
                    max_episode_steps=max_steps,
                    seed=getattr(eval_cfg, "seed", 42),
                )
                try:
                    result = self._evaluate_single(adapter, num_episodes, max_steps)
                    suite_results[task_name] = result
                    suite_success += result["successes"]
                    suite_episodes += result["num_episodes"]
                finally:
                    adapter.close()

            all_suite_results[suite_name] = {
                "success_rate": suite_success / suite_episodes if suite_episodes > 0 else 0.0,
                "successes": suite_success,
                "num_episodes": suite_episodes,
                "num_tasks": len(tasks),
                "per_task": suite_results,
            }
            grand_total_success += suite_success
            grand_total_episodes += suite_episodes

        return {
            "success_rate": grand_total_success / grand_total_episodes if grand_total_episodes > 0 else 0.0,
            "total_episodes": grand_total_episodes,
            "total_successes": grand_total_success,
            "per_suite": all_suite_results,
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
