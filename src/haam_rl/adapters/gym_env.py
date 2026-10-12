"""Gymnasium wrapper around the twin, with the `action_masks()` hook MaskablePPO expects."""
from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from haam_rl.adapters.observation import ObservationEncoder
from haam_rl.adapters.python_twin import PythonPaintShopTwin
from haam_rl.config import ExperimentConfig
from haam_rl.domain.model import ACTION_COUNT


class PaintShopEnv(gym.Env[np.ndarray, int]):
    """Each reset draws a new scenario.

    scenario_seeds=None: scenario seeds come from the environment's own RNG (training).
    scenario_seeds=[...]: resets cycle through that fixed list (validation and testing).
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, config: ExperimentConfig, scenario_seeds: Sequence[int] | None = None) -> None:
        self.config = config
        self.twin = PythonPaintShopTwin(config)
        self.encoder = ObservationEncoder(config.observation, config.plant.fill_target)
        self.scenario_seeds = list(scenario_seeds) if scenario_seeds is not None else None
        self._next_scenario = 0
        low, high = self.encoder.bounds()
        self.observation_space = spaces.Box(low=low, high=high, shape=(self.encoder.size,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_COUNT)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.scenario_seeds is not None:
            scenario = self.scenario_seeds[self._next_scenario % len(self.scenario_seeds)]
            self._next_scenario += 1
        else:
            scenario = int(self.np_random.integers(2**31 - 1))
        state = self.twin.reset(scenario)
        return self.encoder.encode(state), {"scenario_seed": scenario}

    def step(self, action: int):
        state, reward, terminated, info = self.twin.step(int(action))
        return self.encoder.encode(state), reward, terminated, False, dict(vars(info))

    def action_masks(self) -> np.ndarray:
        return self.twin.action_mask()
