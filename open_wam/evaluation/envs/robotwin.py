"""RoboTwin SAPIEN environment adapter.

This is a stub — the actual SAPIEN simulator is an external dependency
provided by the RoboTwin repository. This adapter defines the interface
for integrating with it.
"""

from typing import Tuple

from open_wam.evaluation.envs.base import BaseEnvAdapter


class RoboTwinEnvAdapter(BaseEnvAdapter):
    """Adapter for RoboTwin 2.0 SAPIEN environments.

    Wraps the SAPIEN simulation environment to conform to the
    :class:`BaseEnvAdapter` interface.

    Args:
        task_name: RoboTwin task name (e.g. ``"adjust_bottle"``).
        robot: Robot embodiment (e.g. ``"arx-x5"``).
        headless: Whether to run without GUI rendering.
    """

    def __init__(self, task_name: str, robot: str = "arx-x5", headless: bool = True):
        self.task_name = task_name
        self.robot = robot
        self.headless = headless
        self._env = None

    def reset(self) -> dict:
        """Reset the environment and return the initial observation."""
        raise NotImplementedError(
            "RoboTwinEnvAdapter requires the RoboTwin SAPIEN package. "
            "Install it and override this method with the actual env.reset() call."
        )

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """Execute one step in the environment."""
        raise NotImplementedError(
            "RoboTwinEnvAdapter requires the RoboTwin SAPIEN package."
        )

    def get_obs(self) -> dict:
        """Return the current observation."""
        raise NotImplementedError(
            "RoboTwinEnvAdapter requires the RoboTwin SAPIEN package."
        )
