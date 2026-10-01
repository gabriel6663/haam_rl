from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from haam_rl.adapters.python_twin import PythonPaintShopTwin
from haam_rl.domain.model import ACTION_COUNT


class PaintShopEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, episode_vehicles: int = 100) -> None:
        self.twin = PythonPaintShopTwin(episode_vehicles)
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(low=0, high=6, shape=(27,), dtype=np.int32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        state = self.twin.reset(seed)
        return state.as_vector(), {}

    def step(self, action: int):
        state, reward, terminated, info = self.twin.step(int(action))
        return state.as_vector(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        return self.twin.action_mask()
