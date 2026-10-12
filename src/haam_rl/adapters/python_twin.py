"""Fast in-process digital twin. Same rules as the FlexSim model, without the HTTP round trip."""
from __future__ import annotations

import numpy as np

from haam_rl.application.ports import StepInfo
from haam_rl.config import ExperimentConfig
from haam_rl.domain.dynamics import apply_action, is_terminal
from haam_rl.domain.heuristics import action_mask
from haam_rl.domain.model import PlantState
from haam_rl.domain.reward import reward
from haam_rl.domain.scenarios import generate_arrivals, initial_state


class PythonPaintShopTwin:
    """One episode = one scenario of `plant.vehicles` arrivals. Invalid actions raise immediately."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.arrivals: tuple[int, ...] = ()
        self.color_changes = 0
        self._state = initial_state((1,))
        self._mask: np.ndarray | None = None  # mask of the current state, computed on first use

    @property
    def state(self) -> PlantState:
        return self._state

    def reset(self, seed: int | None = None) -> PlantState:
        plant = self.config.plant
        self.arrivals = generate_arrivals(seed, plant.vehicles, plant.color_probabilities)
        self.color_changes = 0
        self._state, self._mask = initial_state(self.arrivals), None
        return self._state

    def action_mask(self) -> np.ndarray:
        """Mask of the current state. Cached: SB3 asks for it, then `step` checks the chosen action."""
        if self._mask is None:
            self._mask = action_mask(self._state, self.config.plant.fill_target, self.config.mask_mode)
            self._mask.setflags(write=False)
        return self._mask

    def step(self, action: int) -> tuple[PlantState, float, bool, StepInfo]:
        mask = self.action_mask()
        if not 0 <= action < mask.size or not mask[action]:
            raise ValueError(f"Invalid action {action} under mask mode '{self.config.mask_mode.value}'")
        before = self._state
        transition = apply_action(before, action, self.config.plant.fill_target)
        self.color_changes += transition.color_changed
        self._state, self._mask = transition.state, None
        value = reward(before, transition, self.config.reward, self.config.training.gamma)
        info = StepInfo(
            phase=transition.phase.value,
            painted_color=transition.painted_color,
            color_changed=transition.color_changed,
            color_changes=self.color_changes,
        )
        return self._state, value, is_terminal(self._state), info
