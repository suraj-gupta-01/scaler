"""
OpenEnv - Minimal RL Environment Framework

A lightweight replacement for Gymnasium that provides the core RL environment
interface needed for OpenEnv compliance.

This module provides:
- Base Env class with reset/step interface
- Space classes for action/observation definitions
- Basic seeding and rendering support
"""

from abc import ABC, abstractmethod
from typing import Any, Dict as DictType, Optional, Tuple, Union
import numpy as np

# Import the existing OpenEnv Env class
try:
    from openenv.env import Env as BaseEnv
except ImportError:
    # Fallback if not available
    BaseEnv = object


class Space(ABC):
    """Abstract base class for observation and action spaces."""

    @abstractmethod
    def sample(self) -> Any:
        """Sample a random element from this space."""
        pass

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Check if x is contained in this space."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Discrete(Space):
    """A discrete space with n possible values (0 to n-1)."""

    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        return np.random.randint(self.n)

    def contains(self, x: Any) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class Box(Space):
    """A continuous space with bounds."""

    def __init__(self, low: Union[float, np.ndarray], high: Union[float, np.ndarray],
                 shape: Optional[Tuple[int, ...]] = None, dtype: Any = np.float32):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.shape = shape or self.low.shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        return (np.all(x >= self.low) and np.all(x <= self.high) and
                x.shape == self.shape and x.dtype == self.dtype)

    def __repr__(self) -> str:
        return f"Box({self.low}, {self.high}, {self.shape}, {self.dtype})"


class Dict(Space):
    """A dictionary of spaces."""

    def __init__(self, spaces: DictType[str, Space]):
        self.spaces = spaces

    def sample(self) -> DictType[str, Any]:
        return {key: space.sample() for key, space in self.spaces.items()}

    def contains(self, x: Any) -> bool:
        return (isinstance(x, dict) and
                all(key in self.spaces and self.spaces[key].contains(value)
                    for key, value in x.items()))

    def __repr__(self) -> str:
        return f"Dict({self.spaces})"


class Env(BaseEnv):
    """
    OpenEnv environment base class compatible with Gymnasium interface.

    Provides the standard RL environment interface compatible with OpenEnv.
    """

    def __init__(self, name: str = "OpenEnv", state_space=None, action_space=None, episode_max_length: int = 1000):
        if BaseEnv is not object:
            super().__init__(name, state_space, action_space, episode_max_length)
        self.metadata = {}
        self.render_mode = None
        self.np_random = np.random.RandomState()

    def reset(self, *, seed: Optional[int] = None, options: Optional[DictType[str, Any]] = None) -> Any:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Initial observation
        """
        if seed is not None:
            self.np_random.seed(seed)
        return self._reset(seed=seed, options=options)

    def _reset(self, *, seed: Optional[int] = None, options: Optional[DictType[str, Any]] = None) -> Any:
        """Internal reset implementation."""
        raise NotImplementedError("Subclasses must implement _reset")

    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, DictType[str, Any]]:
        """
        Take an action in the environment.

        Args:
            action: Action to take

        Returns:
            observation, reward, terminated, truncated, info
        """
        return self._step(action)

    def _step(self, action: Any) -> Tuple[Any, Any, bool, bool, DictType[str, Any]]:
        """Internal step implementation."""
        raise NotImplementedError("Subclasses must implement _step")

    def render(self) -> Any:
        """Render the environment."""
        return None

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        if seed is not None:
            self.np_random.seed(seed)

    @property
    def unwrapped(self) -> 'Env':
        """Return the unwrapped environment."""
        return self


# Create the spaces module
class _Spaces:
    """Container for space classes."""
    Discrete = Discrete
    Box = Box
    Dict = Dict

spaces = _Spaces()