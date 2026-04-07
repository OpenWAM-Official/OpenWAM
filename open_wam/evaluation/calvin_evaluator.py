"""Calvin benchmark evaluator.

Evaluates using the Calvin long-horizon task chaining protocol:
1000 evaluation sequences, each with up to 5 subtasks to complete in order.
Each subtask allows a maximum of 360 environment steps.

Metrics follow the standard Calvin protocol:
- Average number of completed tasks (0-5) across all sequences
- Per-length success rate (completed exactly 1, 2, ..., 5 subtasks)
"""

import logging
from typing import Dict, List

import numpy as np

from open_wam.evaluation.base import BaseEvaluator
from open_wam.evaluation.policy import WAMPolicy
from open_wam.inference.base import BaseInferenceEngine

logger = logging.getLogger(__name__)


class CalvinEvaluator(BaseEvaluator):
    """Online evaluation on Calvin benchmark.

    Runs the standard Calvin evaluation protocol: long-horizon task chains
    with up to 5 subtasks per sequence.

    Args:
        cfg: Hydra config (must contain ``cfg.eval``).
        engine: Inference engine for action generation.
    """

    def __init__(self, cfg, engine: BaseInferenceEngine):
        super().__init__(cfg, engine)

    def evaluate(self, env_adapter=None) -> dict:
        """Run Calvin long-horizon evaluation.

        Returns:
            dict with average completed tasks, per-length success rates,
            and aggregate metrics.
        """
        eval_cfg = self.cfg.eval
        num_sequences = getattr(eval_cfg, "num_sequences", 1000)
        max_subtasks = getattr(eval_cfg, "max_subtasks", 5)
        max_subtask_steps = getattr(eval_cfg, "max_subtask_steps", 360)
        split = getattr(eval_cfg, "split", "D_D")
        dataset_path = getattr(eval_cfg, "dataset_path", "")

        if env_adapter is None:
            from open_wam.evaluation.envs.calvin import CalvinEnvAdapter

            env_adapter = CalvinEnvAdapter(
                dataset_path=dataset_path,
                split=split,
                image_height=getattr(eval_cfg, "image_height", 200),
                image_width=getattr(eval_cfg, "image_width", 200),
                max_subtask_steps=max_subtask_steps,
            )

        # Load annotation and initial states for evaluation sequences
        annotations = self._load_annotations(dataset_path, split, num_sequences)

        policy_cfg = getattr(self.cfg.eval, "policy", self.cfg.eval)
        policy = WAMPolicy(engine=self.engine, cfg=policy_cfg)

        # Track completions: how many sequences completed exactly k subtasks
        completion_counts = np.zeros(max_subtasks + 1, dtype=int)
        total_completed = 0

        try:
            for seq_idx in range(min(num_sequences, len(annotations))):
                seq = annotations[seq_idx]
                subtask_instructions = seq.get("instructions", [])
                initial_state = seq.get("initial_state", None)

                # Reset environment to sequence initial state
                if initial_state is not None:
                    obs = env_adapter.reset(
                        robot_obs=initial_state.get("robot_obs"),
                        scene_obs=initial_state.get("scene_obs"),
                    )
                else:
                    obs = env_adapter.reset()

                policy.reset()
                completed = 0

                for subtask_idx in range(min(max_subtasks, len(subtask_instructions))):
                    _instruction = subtask_instructions[subtask_idx]  # noqa: F841
                    subtask_success = False

                    for step in range(max_subtask_steps):
                        action = policy.predict_action(obs)
                        obs, reward, done, info = env_adapter.step(action)

                        if info.get("success", False):
                            subtask_success = True
                            break

                    if subtask_success:
                        completed += 1
                    else:
                        break  # Calvin protocol: stop on first failure

                completion_counts[completed] += 1
                total_completed += completed

                if (seq_idx + 1) % 100 == 0:
                    logger.info(
                        "Calvin eval: %d/%d sequences, avg completed: %.2f",
                        seq_idx + 1,
                        num_sequences,
                        total_completed / (seq_idx + 1),
                    )
        finally:
            env_adapter.close()

        num_evaluated = min(num_sequences, len(annotations))

        # Per-length success rates
        per_length = {}
        for k in range(1, max_subtasks + 1):
            # Fraction of sequences that completed at least k subtasks
            at_least_k = sum(completion_counts[k:])
            per_length[f"len_{k}"] = at_least_k / num_evaluated if num_evaluated > 0 else 0.0

        return {
            "avg_completed_tasks": total_completed / num_evaluated if num_evaluated > 0 else 0.0,
            "num_sequences": num_evaluated,
            "total_completed_subtasks": int(total_completed),
            "per_length_success": per_length,
            "completion_distribution": {str(k): int(completion_counts[k]) for k in range(max_subtasks + 1)},
        }

    def _load_annotations(
        self,
        dataset_path: str,
        split: str,
        num_sequences: int,
    ) -> List[Dict]:
        """Load Calvin evaluation annotations.

        Each annotation contains the subtask instruction sequence and
        the initial robot/scene state for that evaluation sequence.

        Falls back to synthetic placeholder annotations if the dataset
        is not available (for dry-run / config validation).
        """
        import os

        # Try to load from Calvin dataset
        ann_path = os.path.join(dataset_path, "lang_annotations/auto_lang_ann.npy")
        if os.path.exists(ann_path):
            data = np.load(ann_path, allow_pickle=True).item()
            annotations = []
            lang_anns = data.get("language", {}).get("ann", [])
            for i in range(min(num_sequences, len(lang_anns))):
                annotations.append(
                    {
                        "instructions": lang_anns[i] if isinstance(lang_anns[i], list) else [lang_anns[i]],
                        "initial_state": None,
                    }
                )
            return annotations

        # Fallback: generate placeholder sequences for testing
        logger.warning(
            "Calvin annotations not found at %s. Using placeholder sequences for dry-run evaluation.",
            ann_path,
        )
        placeholder_instructions = [
            "turn on the lightbulb",
            "move the slider to the left",
            "open the drawer",
            "pick up the blue block",
            "place the block in the drawer",
        ]
        return [{"instructions": placeholder_instructions, "initial_state": None} for _ in range(num_sequences)]
