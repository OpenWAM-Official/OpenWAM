"""BEHAVIOR-1K evaluation environment adapter.

BEHAVIOR-1K (https://behavior.stanford.edu/) provides a comprehensive
benchmark of 1000 everyday household activities in simulation using
OmniGibson (built on NVIDIA Isaac Sim / Omniverse).

Supports diverse tasks: navigation, manipulation, cleaning, cooking, etc.

Action format varies by controller; this adapter defaults to 7D delta EEF
(dx, dy, dz, droll, dpitch, dyaw, gripper) with OSC controller.

Installation: pip install omnigibson
See: https://behavior.stanford.edu/omnigibson/getting_started/installation.html
"""

from typing import Tuple

import numpy as np

from open_wam.evaluation.envs.base import BaseEnvAdapter


class BehaviorEnvAdapter(BaseEnvAdapter):
    """Adapter for BEHAVIOR-1K / OmniGibson evaluation environments.

    Wraps OmniGibson's environment interface to conform to
    :class:`BaseEnvAdapter`.

    Args:
        activity_name: BEHAVIOR activity name (e.g. "cleaning_up_the_kitchen_only").
        scene_model: OmniGibson scene (e.g. "Rs_int").
        robot: Robot model name (e.g. "Fetch", "Tiago").
        controller: Controller type ("OSC_POSE" for 7D delta EEF).
        image_height: Observation image height.
        image_width: Observation image width.
        max_episode_steps: Maximum steps per episode.
        action_scale: Scale factor applied to actions.
    """

    def __init__(
        self,
        activity_name: str = "cleaning_up_the_kitchen_only",
        scene_model: str = "Rs_int",
        robot: str = "Fetch",
        controller: str = "OSC_POSE",
        image_height: int = 256,
        image_width: int = 256,
        max_episode_steps: int = 1000,
        action_scale: float = 1.0,
    ):
        self.activity_name = activity_name
        self.scene_model = scene_model
        self.robot = robot
        self.controller = controller
        self.image_height = image_height
        self.image_width = image_width
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self._env = None
        self._step_count = 0

    def _lazy_init_env(self):
        """Lazily initialize the OmniGibson environment."""
        if self._env is not None:
            return

        try:
            import omnigibson as og
        except ImportError:
            raise ImportError(
                "OmniGibson is required for BEHAVIOR-1K evaluation. "
                "Install with: pip install omnigibson\n"
                "See: https://behavior.stanford.edu/omnigibson/getting_started/installation.html"
            )

        cfg = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": self.scene_model,
            },
            "robots": [
                {
                    "type": self.robot,
                    "obs_modalities": ["rgb", "proprio"],
                    "controller_config": {
                        "arm_0": {"name": self.controller},
                    },
                }
            ],
            "task": {
                "type": "BehaviorTask",
                "activity_name": self.activity_name,
            },
        }

        self._env = og.Environment(configs=cfg)

    def reset(self) -> dict:
        """Reset environment and return initial observation."""
        self._lazy_init_env()
        obs, info = self._env.reset()
        self._step_count = 0
        return self._process_obs(obs, info)

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """Execute one step.

        Args:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]
                   for OSC_POSE controller.

        Returns:
            (obs_dict, reward, done, info)
        """
        self._lazy_init_env()

        action = np.asarray(action, dtype=np.float64) * self.action_scale
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1

        done = terminated or truncated
        info["success"] = info.get("success", False)

        obs_dict = self._process_obs(obs, info)
        return obs_dict, float(reward), done, info

    def get_obs(self) -> dict:
        """Return current observation dict."""
        self._lazy_init_env()
        obs = self._env.get_obs() if hasattr(self._env, "get_obs") else {}
        return self._process_obs(obs, {})

    def _process_obs(self, obs, info: dict) -> dict:
        """Convert OmniGibson observation to standard dict format."""
        from PIL import Image

        obs_dict = {"info": info}

        # OmniGibson typically returns nested dict: obs[robot_name][modality]
        if isinstance(obs, dict):
            # Try common OmniGibson observation keys
            robot_obs = None
            for key in obs:
                if isinstance(obs[key], dict):
                    robot_obs = obs[key]
                    break

            if robot_obs is None:
                robot_obs = obs

            # Extract RGB
            for img_key in ["rgb", "agentview_rgb", "robot0_rgb"]:
                if img_key in robot_obs:
                    img_array = robot_obs[img_key]
                    if isinstance(img_array, np.ndarray):
                        if img_array.ndim == 4:
                            img_array = img_array[0]  # Remove batch dim
                        if img_array.shape[-1] == 4:
                            img_array = img_array[..., :3]  # RGBA -> RGB
                        if img_array.dtype != np.uint8:
                            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_array)
                        pil_img = pil_img.resize((self.image_width, self.image_height), Image.LANCZOS)
                        obs_dict["image"] = pil_img
                    break

            # Proprioception
            if "proprio" in robot_obs:
                obs_dict["state"] = np.asarray(robot_obs["proprio"], dtype=np.float32)

        obs_dict["step"] = self._step_count
        return obs_dict

    def close(self):
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
