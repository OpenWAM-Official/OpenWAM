"""SimplerEnv evaluation environment adapter.

SimplerEnv (https://github.com/simpler-env/SimplerEnv) provides simulated
environments for evaluating real-world robot manipulation policies.

Built on SAPIEN/ManiSkill, it supports:
- Google Robot embodiment
- WidowX + Bridge embodiment
- Rigid-body and articulated object manipulation tasks

Action format: 7D delta EEF (dx, dy, dz, droll, dpitch, dyaw, gripper)

Installation: pip install simpler-env
"""

from typing import Tuple

import numpy as np

from open_wam.evaluation.envs.base import BaseEnvAdapter


class SimplerEnvAdapter(BaseEnvAdapter):
    """Adapter for SimplerEnv evaluation environments.

    Wraps SimplerEnv's gym-compatible interface to conform to
    :class:`BaseEnvAdapter`.

    Args:
        env_name: SimplerEnv environment name
            (e.g. "google_robot_pick_coke_can", "widowx_spoon_on_towel").
        robot: Robot type ("google_robot" or "widowx").
        image_height: Observation image height.
        image_width: Observation image width.
        max_episode_steps: Maximum steps per episode.
        action_scale: Scale factor applied to actions before sending to env.
    """

    def __init__(
        self,
        env_name: str = "google_robot_pick_coke_can",
        robot: str = "google_robot",
        image_height: int = 256,
        image_width: int = 256,
        max_episode_steps: int = 200,
        action_scale: float = 1.0,
    ):
        self.env_name = env_name
        self.robot = robot
        self.image_height = image_height
        self.image_width = image_width
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self._env = None
        self._step_count = 0

    def _lazy_init_env(self):
        """Lazily initialize the SimplerEnv environment."""
        if self._env is not None:
            return

        try:
            import simpler_env

            self._env = simpler_env.make(
                self.env_name,
                max_episode_steps=self.max_episode_steps,
            )
        except ImportError:
            raise ImportError(
                "SimplerEnv is required for this adapter. "
                "Install with: pip install simpler-env\n"
                "See: https://github.com/simpler-env/SimplerEnv"
            )

    def reset(self) -> dict:
        """Reset environment and return initial observation."""
        self._lazy_init_env()
        obs, info = self._env.reset()
        self._step_count = 0
        return self._process_obs(obs, info)

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """Execute one step with 7D delta EEF action.

        Args:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]

        Returns:
            (obs_dict, reward, done, info)
        """
        self._lazy_init_env()

        action = np.asarray(action, dtype=np.float32) * self.action_scale
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1

        done = terminated or truncated
        obs_dict = self._process_obs(obs, info)

        return obs_dict, float(reward), done, info

    def get_obs(self) -> dict:
        """Return current observation dict."""
        self._lazy_init_env()
        obs = self._env.get_obs() if hasattr(self._env, "get_obs") else {}
        return self._process_obs(obs, {})

    def get_language_instruction(self) -> str:
        """Get the natural language task instruction."""
        if self._env is not None and hasattr(self._env, "get_language_instruction"):
            return self._env.get_language_instruction()
        return self.env_name.replace("_", " ")

    def _process_obs(self, obs, info: dict) -> dict:
        """Convert raw observation to standard dict format."""
        from PIL import Image

        obs_dict = {"info": info}

        # Extract RGB image
        if isinstance(obs, dict):
            for key in ["image", "rgb", "agentview_rgb", "pixels"]:
                if key in obs:
                    img = obs[key]
                    if isinstance(img, np.ndarray):
                        pil_img = Image.fromarray(img.astype(np.uint8))
                        pil_img = pil_img.resize((self.image_width, self.image_height), Image.LANCZOS)
                        obs_dict["image"] = pil_img
                    break
            # Proprioception
            for key in ["joint_positions", "state", "robot_state"]:
                if key in obs:
                    obs_dict["state"] = obs[key]
                    break
        elif isinstance(obs, np.ndarray):
            # Some envs return raw image as observation
            pil_img = Image.fromarray(obs.astype(np.uint8))
            pil_img = pil_img.resize((self.image_width, self.image_height), Image.LANCZOS)
            obs_dict["image"] = pil_img

        obs_dict["step"] = self._step_count
        return obs_dict

    def close(self):
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
