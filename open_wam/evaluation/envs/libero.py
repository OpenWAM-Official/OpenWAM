"""LIBERO evaluation environment adapter.

LIBERO (https://github.com/Lifelong-Robot-Learning/LIBERO) is a benchmark
suite for lifelong robot learning with:
- 130 manipulation tasks across 4 task suites (LIBERO-Spatial, LIBERO-Object,
  LIBERO-Goal, LIBERO-Long)
- Franka Panda robot in MuJoCo (via robosuite)
- 7-DoF delta EEF actions (dx, dy, dz, droll, dpitch, dyaw, gripper)
- agentview (128x128) and eye_in_hand (128x128) cameras

Installation: pip install libero
"""

from typing import Tuple

import numpy as np

from open_wam.evaluation.envs.base import BaseEnvAdapter

# LIBERO task suites
LIBERO_SPATIAL_TASKS = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_the_top_of_the_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_sheet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_sheet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_rack_and_place_it_on_the_plate",
]

LIBERO_OBJECT_TASKS = [
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
]

LIBERO_GOAL_TASKS = [
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl",
    "turn_on_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
]

LIBERO_SUITES = {
    "libero_spatial": LIBERO_SPATIAL_TASKS,
    "libero_object": LIBERO_OBJECT_TASKS,
    "libero_goal": LIBERO_GOAL_TASKS,
}


class LIBEROEnvAdapter(BaseEnvAdapter):
    """Adapter for LIBERO benchmark environments.

    Wraps LIBERO's robosuite-based environments to conform to
    :class:`BaseEnvAdapter`.

    Args:
        task_name: LIBERO task name string.
        task_suite: Suite name ("libero_spatial", "libero_object", "libero_goal").
        image_height: Observation image height.
        image_width: Observation image width.
        max_episode_steps: Maximum steps per episode.
        action_scale: Scale factor applied to actions.
        seed: Random seed for environment.
    """

    def __init__(
        self,
        task_name: str,
        task_suite: str = "libero_spatial",
        image_height: int = 256,
        image_width: int = 256,
        max_episode_steps: int = 300,
        action_scale: float = 1.0,
        seed: int = 42,
    ):
        self.task_name = task_name
        self.task_suite = task_suite
        self.image_height = image_height
        self.image_width = image_width
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.seed = seed
        self._env = None
        self._step_count = 0

    def _lazy_init_env(self):
        """Lazily initialize the LIBERO environment."""
        if self._env is not None:
            return

        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError:
            raise ImportError(
                "LIBERO is required for this adapter. "
                "Install with: pip install libero\n"
                "See: https://github.com/Lifelong-Robot-Learning/LIBERO"
            )

        bench = benchmark.get_benchmark(self.task_suite)
        task_idx = None
        for i, task in enumerate(bench.get_task_names()):
            if task == self.task_name:
                task_idx = i
                break
        if task_idx is None:
            raise ValueError(
                f"Task '{self.task_name}' not found in suite '{self.task_suite}'. Available: {bench.get_task_names()}"
            )

        task = bench.get_task(task_idx)
        task_bddl = bench.get_task_bddl_file_path(task_idx)
        env_args = {
            "bddl_file_name": task_bddl,
            "camera_heights": self.image_height,
            "camera_widths": self.image_width,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
        }
        self._env = OffScreenRenderEnv(**env_args)
        self._env.seed(self.seed)

    def reset(self) -> dict:
        """Reset environment and return initial observation."""
        self._lazy_init_env()
        obs = self._env.reset()
        self._step_count = 0
        return self._process_obs(obs)

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """Execute one step with 7D delta EEF action."""
        self._lazy_init_env()

        action = np.asarray(action, dtype=np.float64) * self.action_scale
        # LIBERO expects (7,) array
        if len(action) > 7:
            action = action[:7]
        elif len(action) < 7:
            action = np.pad(action, (0, 7 - len(action)))

        obs, reward, done, info = self._env.step(action)
        self._step_count += 1

        if self._step_count >= self.max_episode_steps:
            done = True

        obs_dict = self._process_obs(obs)
        return obs_dict, float(reward), done, info

    def get_obs(self) -> dict:
        """Return current observation dict."""
        self._lazy_init_env()
        obs = self._env._get_observations()
        return self._process_obs(obs)

    def get_language_instruction(self) -> str:
        """Get the natural language task instruction."""
        return self.task_name.replace("_", " ")

    def _process_obs(self, obs: dict) -> dict:
        """Convert raw LIBERO observation to standard dict format."""
        from PIL import Image

        obs_dict = {}

        # LIBERO provides agentview_image and robot0_eye_in_hand_image
        for key in ["agentview_image", "robot0_eye_in_hand_image"]:
            if key in obs:
                img = obs[key]
                if isinstance(img, np.ndarray):
                    # LIBERO images may need flipping (origin at bottom-left)
                    if img.ndim == 3:
                        img = img[::-1]
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    pil_img = pil_img.resize((self.image_width, self.image_height), Image.LANCZOS)
                    obs_dict[key] = pil_img

        # Primary image for WAM policy
        if "agentview_image" in obs_dict:
            obs_dict["image"] = obs_dict["agentview_image"]

        # Robot state / proprioception
        for key in ["robot0_joint_pos", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]:
            if key in obs:
                obs_dict[key] = np.asarray(obs[key], dtype=np.float32)

        # Compact state vector
        state_parts = []
        for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]:
            if key in obs:
                state_parts.append(np.asarray(obs[key], dtype=np.float32).flatten())
        if state_parts:
            obs_dict["state"] = np.concatenate(state_parts)

        obs_dict["step"] = self._step_count
        return obs_dict

    def close(self):
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
