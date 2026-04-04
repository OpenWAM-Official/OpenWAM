"""Calvin evaluation environment adapter.

Calvin (https://github.com/mees/calvin) is a benchmark for evaluating
language-conditioned policy learning for long-horizon robot manipulation.

It tests compositional generalization: given a sequence of language instructions,
the agent must execute up to 5 subtasks in sequence. Each subtask allows a
maximum of 360 environment steps.

Action format: 7D delta EEF (dx, dy, dz, droll, dpitch, dyaw, gripper)

Installation: See https://github.com/mees/calvin for setup instructions.
"""

from typing import Optional, Tuple

import numpy as np

from open_wam.evaluation.envs.base import BaseEnvAdapter


class CalvinEnvAdapter(BaseEnvAdapter):
    """Adapter for Calvin evaluation environments.

    Wraps Calvin's PyBullet-based environment to conform to
    :class:`BaseEnvAdapter`.

    Args:
        dataset_path: Path to Calvin dataset (e.g., "task_D_D" or "task_ABC_D").
        split: Evaluation split ("D_D" for seen, "ABC_D" for unseen scene transfer).
        image_height: Observation image height.
        image_width: Observation image width.
        max_subtask_steps: Maximum steps per subtask (Calvin standard: 360).
        action_scale: Scale factor applied to actions.
    """

    def __init__(
        self,
        dataset_path: str = "",
        split: str = "D_D",
        image_height: int = 200,
        image_width: int = 200,
        max_subtask_steps: int = 360,
        action_scale: float = 1.0,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.image_height = image_height
        self.image_width = image_width
        self.max_subtask_steps = max_subtask_steps
        self.action_scale = action_scale
        self._env = None
        self._step_count = 0

    def _lazy_init_env(self):
        """Lazily initialize the Calvin environment."""
        if self._env is not None:
            return

        try:
            from calvin_env.envs.play_table_env import PlayTableSimEnv
            import hydra
        except ImportError:
            raise ImportError(
                "Calvin environment is required for this adapter. "
                "See: https://github.com/mees/calvin for installation."
            )

        # Load Calvin environment with default config
        self._env = PlayTableSimEnv(
            tasks={},
            initial_state="neutral",
            cameras=("static", "gripper"),
            max_episode_length=self.max_subtask_steps,
        )

    def reset(self, robot_obs=None, scene_obs=None) -> dict:
        """Reset environment and return initial observation.

        Args:
            robot_obs: Optional initial robot state to reset to.
            scene_obs: Optional initial scene state to reset to.
        """
        self._lazy_init_env()
        if robot_obs is not None and scene_obs is not None:
            self._env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        else:
            self._env.reset()
        self._step_count = 0
        obs = self._env.get_obs()
        return self._process_obs(obs, {})

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """Execute one step with 7D delta EEF action.

        Args:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]

        Returns:
            (obs_dict, reward, done, info)
        """
        self._lazy_init_env()

        action = np.asarray(action, dtype=np.float32) * self.action_scale
        obs, reward, done, info = self._env.step(action)
        self._step_count += 1

        obs_dict = self._process_obs(obs, info)
        return obs_dict, float(reward), done, info

    def get_obs(self) -> dict:
        """Return current observation dict."""
        self._lazy_init_env()
        obs = self._env.get_obs()
        return self._process_obs(obs, {})

    def _process_obs(self, obs, info: dict) -> dict:
        """Convert Calvin observation to standard dict format."""
        from PIL import Image

        obs_dict = {"info": info}

        # Calvin provides "rgb_static" and "rgb_gripper" cameras
        if isinstance(obs, dict):
            for key in ["rgb_static", "rgb_obs"]:
                if key in obs:
                    img_array = obs[key]
                    if isinstance(img_array, np.ndarray):
                        if img_array.dtype != np.uint8:
                            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_array)
                        pil_img = pil_img.resize(
                            (self.image_width, self.image_height), Image.LANCZOS
                        )
                        obs_dict["image"] = pil_img
                    break

            # Proprioception: robot_obs contains [tcp_pos(3), tcp_orn(3), gripper(1)]
            if "robot_obs" in obs:
                obs_dict["state"] = np.asarray(obs["robot_obs"], dtype=np.float32)
        elif isinstance(obs, np.ndarray):
            pil_img = Image.fromarray(obs.astype(np.uint8))
            pil_img = pil_img.resize(
                (self.image_width, self.image_height), Image.LANCZOS
            )
            obs_dict["image"] = pil_img

        obs_dict["step"] = self._step_count
        return obs_dict

    def close(self):
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
