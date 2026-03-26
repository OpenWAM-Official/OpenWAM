"""RoboTwin offline and online evaluators wrapping legacy eval logic."""

import sys
import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from open_wam.evaluation.base import BaseEvaluator
from open_wam.inference.base import BaseInferenceEngine

_WAM_DIR = str(Path(__file__).resolve().parent.parent.parent / "examples" / "wanvideo" / "wam")
if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)


class RoboTwinOfflineEvaluator(BaseEvaluator):
    """Offline evaluation on RoboTwin validation sets.

    Iterates over a dataset, runs the inference engine on each sample,
    and computes action and video metrics against ground truth.

    Args:
        cfg: Hydra config (must contain ``cfg.eval``).
        engine: A :class:`BaseInferenceEngine` for generation.
    """

    def __init__(self, cfg, engine: BaseInferenceEngine):
        super().__init__(cfg, engine)

    def evaluate(self, dataset) -> dict:
        """Run offline evaluation on a dataset.

        Args:
            dataset: A dataset (BaseActionDataset or legacy VideoActionDataset)
                where each sample has video, action, prompt, etc.

        Returns:
            dict of aggregated metrics (action_mse, action_mae, video_psnr, etc.)
        """
        from eval_robotwin import compute_video_metrics  # noqa: E402
        from open_wam.inference.schedule import make_schedule

        eval_cfg = self.cfg.eval
        num_samples = min(
            getattr(eval_cfg, "num_eval_samples", len(dataset)),
            len(dataset),
        )

        all_action_mse = []
        all_action_mae = []
        all_video_metrics = []

        for i in range(num_samples):
            sample = dataset[i]

            # Build conditions from dataset sample
            conditions = {
                "prompt": sample.get("prompt", ""),
                "vace_reference_image": sample.get("vace_reference_image", sample.get("reference_image")),
                "vace_video": sample.get("vace_video", sample.get("context_video")),
                "seed": 42 + i,
            }

            result = self.engine.generate(conditions)
            pred_actions = result["actions"]
            gen_video = result["video"]

            # Action metrics
            gt_actions = sample.get("action", sample.get("action_trajectory"))
            if hasattr(gt_actions, "numpy"):
                gt_actions = gt_actions.numpy()
            if hasattr(pred_actions, "numpy"):
                pred_actions = pred_actions.numpy()

            # Denormalize if dataset provides stats
            if hasattr(dataset, "denormalize_action"):
                gt_actions_denorm = dataset.denormalize_action(gt_actions)
                # pred_actions from generate() are already denormalized by legacy code
                pred_denorm = pred_actions
            else:
                gt_actions_denorm = gt_actions
                pred_denorm = pred_actions

            n_steps = min(len(gt_actions_denorm), len(pred_denorm))
            gt_clip = gt_actions_denorm[:n_steps]
            pred_clip = pred_denorm[:n_steps]

            mse = float(np.mean((gt_clip - pred_clip) ** 2))
            mae = float(np.mean(np.abs(gt_clip - pred_clip)))
            all_action_mse.append(mse)
            all_action_mae.append(mae)

            # Video metrics (if ground truth video available)
            gt_video = sample.get("video")
            if gt_video is not None and gen_video is not None:
                try:
                    vm = compute_video_metrics(gen_video, gt_video)
                    all_video_metrics.append(vm)
                except Exception:
                    pass

        results = {
            "action_mse": float(np.mean(all_action_mse)) if all_action_mse else 0.0,
            "action_mae": float(np.mean(all_action_mae)) if all_action_mae else 0.0,
            "num_samples": num_samples,
        }

        if all_video_metrics:
            for key in all_video_metrics[0]:
                results[key] = float(np.mean([m[key] for m in all_video_metrics]))

        return results


class RoboTwinOnlineEvaluator(BaseEvaluator):
    """Online closed-loop evaluation in RoboTwin SAPIEN environment.

    Uses a WAMPolicy to interact with an environment adapter step-by-step.

    Args:
        cfg: Hydra config (must contain ``cfg.eval``).
        engine: A :class:`BaseInferenceEngine` for generation.
    """

    def __init__(self, cfg, engine: BaseInferenceEngine):
        super().__init__(cfg, engine)

    def evaluate(self, env_adapter) -> dict:
        """Run online closed-loop evaluation.

        Args:
            env_adapter: A :class:`BaseEnvAdapter` providing reset/step/get_obs.

        Returns:
            dict of aggregated metrics (success_rate, avg_reward, etc.)
        """
        from open_wam.evaluation.policy import WAMPolicy

        eval_cfg = self.cfg.eval
        num_episodes = getattr(eval_cfg, "num_episodes", 50)
        max_steps = getattr(eval_cfg, "max_steps_per_episode", 500)

        policy = WAMPolicy(engine=self.engine, cfg=eval_cfg.get("policy", eval_cfg))

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
            "num_episodes": num_episodes,
        }
