"""RoboCasa evaluation environment adapter.

RoboCasa (https://github.com/robocasa/robocasa) provides large-scale simulation
benchmarks for everyday household manipulation tasks built on robosuite.

Supports 100+ tasks across kitchen environments with diverse objects and layouts.

Action format: 7D delta EEF (dx, dy, dz, droll, dpitch, dyaw, gripper)

Installation: pip install robocasa
"""

from typing import Tuple

import numpy as np

from open_wam.evaluation.envs.base import BaseEnvAdapter


class RoboCasaEnvAdapter(BaseEnvAdapter):
    """Adapter for RoboCasa evaluation environments.

    Wraps RoboCasa's robosuite-based gym interface to conform to
    :class:`BaseEnvAdapter`.

    Args:
        task_name: RoboCasa task name (e.g. "PnPCounterToCab").
        layout: Kitchen layout ID (0-9).
        style: Kitchen style ID (0-11).
        image_height: Observation image height.
        image_width: Observation image width.
        max_episode_steps: Maximum steps per episode.
        action_scale: Scale factor applied to actions before sending to env.
        seed: Random seed.
    """

    def __init__(
        self,
        task_name: str = "PnPCounterToCab",
        layout: int = 0,
        style: int = 0,
        image_height: int = 256,
        image_width: int = 256,
        max_episode_steps: int = 500,
        action_scale: float = 1.0,
        seed: int = 42,
    ):
        self.task_name = task_name
        self.layout = layout
        self.style = style
        self.image_height = image_height
        self.image_width = image_width
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.seed = seed
        self._env = None
        self._step_count = 0

    def _lazy_init_env(self):
        """Lazily initialize the RoboCasa environment."""
        if self._env is not None:
            return

        try:
            import robocasa  # noqa: F401
            import robosuite as suite
        except ImportError:
            raise ImportError(
                "RoboCasa and robosuite are required for this adapter. "
                "Install with: pip install robocasa robosuite\n"
                "See: https://github.com/robocasa/robocasa"
            )

        self._env = suite.make(
            self.task_name,
            robots=["PandaMobile"],
            controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
            layout_ids=self.layout,
            style_ids=self.style,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_heights=self.image_height,
            camera_widths=self.image_width,
            camera_names=["robot0_agentview_center"],
            horizon=self.max_episode_steps,
            seed=self.seed,
        )

    def reset(self) -> dict:
        """Reset environment and return initial observation."""
        self._lazy_init_env()
        obs = self._env.reset()
        self._step_count = 0
        return self._process_obs(obs, {})

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """Execute one step with 7D delta EEF action.

        Args:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]

        Returns:
            (obs_dict, reward, done, info)
        """
        self._lazy_init_env()

        action = np.asarray(action, dtype=np.float64) * self.action_scale
        obs, reward, done, info = self._env.step(action)
        self._step_count += 1

        success = self._env.check_success() if hasattr(self._env, "check_success") else False
        info["success"] = success

        obs_dict = self._process_obs(obs, info)
        return obs_dict, float(reward), done, info

    def get_obs(self) -> dict:
        """Return current observation dict."""
        self._lazy_init_env()
        obs = self._env._get_observations() if hasattr(self._env, "_get_observations") else {}
        return self._process_obs(obs, {})

    def get_language_instruction(self) -> str:
        """Get the natural language task instruction."""
        if self._env is not None and hasattr(self._env, "get_ep_meta"):
            meta = self._env.get_ep_meta()
            if "lang" in meta:
                return meta["lang"]
        return self.task_name.replace("_", " ")

    def _process_obs(self, obs, info: dict) -> dict:
        """Convert raw robosuite observation to standard dict format."""
        from PIL import Image

        obs_dict = {"info": info}

        # Extract RGB image from robosuite camera observation
        for key in ["robot0_agentview_center_image", "agentview_image", "frontview_image"]:
            if key in obs:
                img_array = obs[key]
                if isinstance(img_array, np.ndarray):
                    # robosuite returns (H, W, 3) uint8 or float images
                    if img_array.dtype == np.float64 or img_array.dtype == np.float32:
                        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_array[::-1])  # robosuite images are vertically flipped
                    pil_img = pil_img.resize((self.image_width, self.image_height), Image.LANCZOS)
                    obs_dict["image"] = pil_img
                break

        # Proprioception
        if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
            obs_dict["state"] = np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    obs["robot0_eef_quat"],
                ]
            )

        obs_dict["step"] = self._step_count
        return obs_dict

    def close(self):
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
