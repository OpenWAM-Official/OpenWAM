"""Abstract base class for environment adapters."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseEnvAdapter(ABC):
    """Environment adapter supporting different simulators (SAPIEN, SimplerEnv, etc.)."""

    @abstractmethod
    def reset(self) -> dict:
        """Reset the environment and return initial observation."""
        ...

    @abstractmethod
    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """Execute action, return (obs, reward, done, info)."""
        ...

    @abstractmethod
    def get_obs(self) -> dict:
        """Return current observation dict."""
        ...
