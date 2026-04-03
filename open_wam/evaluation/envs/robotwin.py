"""RoboTwin SAPIEN environment adapter.

RoboTwin (https://github.com/TianxingChen/RoboTwin) is a benchmark suite
for dual-arm robot manipulation built on SAPIEN:
- 50 manipulation tasks across ARX-X5 bimanual robots
- 14-DoF joint/EE delta actions (7 per arm)
- 4 cameras: head, third-person, left, right
- SAPIEN physics simulator

Installation: Follow https://github.com/TianxingChen/RoboTwin for setup.
"""

from typing import Tuple

import numpy as np

from open_wam.evaluation.envs.base import BaseEnvAdapter


class RoboTwinEnvAdapter(BaseEnvAdapter):
    """Adapter for RoboTwin 2.0 SAPIEN environments.

    Wraps the SAPIEN simulation environment to conform to the
    :class:`BaseEnvAdapter` interface.

    Args:
        task_name: RoboTwin task name (e.g. ``"adjust_bottle"``).
        robot: Robot embodiment (e.g. ``"arx-x5"``).
        headless: Whether to run without GUI rendering.
        image_height: Observation image height.
        image_width: Observation image width.
        max_episode_steps: Maximum steps per episode.
        action_scale: Scale factor applied to actions.
        seed: Random seed for environment.
    """

    def __init__(
        self,
        task_name: str,
        robot: str = "arx-x5",
        headless: bool = True,
        image_height: int = 256,
        image_width: int = 256,
        max_episode_steps: int = 500,
        action_scale: float = 1.0,
        seed: int = 42,
    ):
        self.task_name = task_name
        self.robot = robot
        self.headless = headless
        self.image_height = image_height
        self.image_width = image_width
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.seed = seed
        self._env = None
        self._step_count = 0

    def _lazy_init_env(self):
        """Lazily initialize the RoboTwin SAPIEN environment."""
        if self._env is not None:
            return

        try:
            from robotwin.envs import make_env
        except ImportError:
            raise ImportError(
                "RoboTwin is required for this adapter. "
                "Install it following: https://github.com/TianxingChen/RoboTwin\n"
                "The 'robotwin' package must be importable with "
                "'from robotwin.envs import make_env'."
            )

        self._env = make_env(
            task_name=self.task_name,
            robot=self.robot,
            headless=self.headless,
            seed=self.seed,
        )

    def reset(self) -> dict:
        """Reset the environment and return the initial observation."""
        self._lazy_init_env()
        obs = self._env.reset()
        self._step_count = 0
        return self._process_obs(obs)

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """Execute one step with 14D bimanual action.

        Args:
            action: (14,) array for bimanual robot or (7,) for single arm.

        Returns:
            (obs_dict, reward, done, info)
        """
        self._lazy_init_env()

        action = np.asarray(action, dtype=np.float64) * self.action_scale
        # RoboTwin expects 14D for bimanual, pad/truncate as needed
        if len(action) > 14:
            action = action[:14]
        elif len(action) < 14:
            action = np.pad(action, (0, 14 - len(action)))

        obs, reward, done, info = self._env.step(action)
        self._step_count += 1

        if self._step_count >= self.max_episode_steps:
            done = True

        obs_dict = self._process_obs(obs)
        return obs_dict, float(reward), done, info

    def get_obs(self) -> dict:
        """Return the current observation."""
        self._lazy_init_env()
        if hasattr(self._env, "get_obs"):
            obs = self._env.get_obs()
        elif hasattr(self._env, "_get_observations"):
            obs = self._env._get_observations()
        else:
            obs = {}
        return self._process_obs(obs)

    def get_language_instruction(self) -> str:
        """Get the natural language task instruction."""
        return self.task_name.replace("_", " ")

    def _process_obs(self, obs) -> dict:
        """Convert raw RoboTwin observation to standard dict format.

        Handles both dict-style observations (with camera keys) and
        raw image arrays.
        """
        from PIL import Image

        obs_dict = {}

        if isinstance(obs, dict):
            # Extract camera images
            for key in ["head_camera", "third_view_rgb", "left_camera", "right_camera"]:
                if key in obs:
                    img = obs[key]
                    if isinstance(img, np.ndarray):
                        if img.ndim == 3:
                            pil_img = Image.fromarray(img.astype(np.uint8))
                            pil_img = pil_img.resize(
                                (self.image_width, self.image_height), Image.LANCZOS
                            )
                            obs_dict[key] = pil_img

            # Primary image for WAM policy — prefer head_camera
            for key in ["head_camera", "third_view_rgb", "image", "rgb"]:
                if key in obs_dict:
                    obs_dict["image"] = obs_dict[key]
                    break

            # Robot state / proprioception
            for key in ["joint_positions", "joint_velocities", "eef_pos", "eef_quat",
                         "gripper_qpos", "state", "robot_state", "qpos"]:
                if key in obs:
                    val = obs[key]
                    if isinstance(val, np.ndarray):
                        obs_dict[key] = val.astype(np.float32)

            # Compact state vector
            state_parts = []
            for key in ["qpos", "joint_positions", "eef_pos", "gripper_qpos"]:
                if key in obs:
                    state_parts.append(np.asarray(obs[key], dtype=np.float32).flatten())
            if state_parts:
                obs_dict["state"] = np.concatenate(state_parts)

        elif isinstance(obs, np.ndarray):
            # Raw image observation
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
            if hasattr(self._env, "close"):
                self._env.close()
            self._env = None
