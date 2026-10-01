from __future__ import annotations

from collections import deque

import numpy as np

from haam_rl.application.use_cases import allowed_actions
from haam_rl.domain.model import BUFFER_COUNT, PlantState, decode_action
from haam_rl.domain.reward import shaped_reward

PAPER_COLOR_PROBABILITIES = np.asarray([0.06, 0.38, 0.29, 0.14, 0.10, 0.03], dtype=float)


class PythonPaintShopTwin:
    """Fast, deterministic-under-seed digital twin. It never accepts an invalid action."""

    def __init__(self, episode_vehicles: int = 100) -> None:
        self.episode_vehicles = episode_vehicles
        self._rng = np.random.default_rng()
        self._state = PlantState(0, 0, ((), (), (), (), ()), ())
        self.color_changes = 0

    @property
    def state(self) -> PlantState:
        return self._state

    def reset(self, seed: int | None = None) -> PlantState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.episode_vehicles < BUFFER_COUNT + 1:
            raise ValueError("episode_vehicles must be at least 6")
        arrivals = tuple((self._rng.choice(6, size=self.episode_vehicles, p=PAPER_COLOR_PROBABILITIES) + 1).tolist())
        initial_buffers = tuple((arrivals[index],) for index in range(BUFFER_COUNT))
        first, *remaining = arrivals[BUFFER_COUNT:]
        self.color_changes = 0
        self._state = PlantState(first, 0, initial_buffers, tuple(remaining))
        return self._state

    def action_mask(self) -> np.ndarray:
        return allowed_actions(self._state)

    def step(self, action: int) -> tuple[PlantState, float, bool, dict[str, float]]:
        mask = self.action_mask()
        if not 0 <= action < len(mask) or not mask[action]:
            raise ValueError(f"Invalid HAAM action: {action}")
        before = self._state
        input_buffer, output_buffer = decode_action(action)
        buffers = [deque(buffer) for buffer in before.buffers]
        painted = buffers[output_buffer].popleft()
        if before.incoming_color:
            buffers[input_buffer].append(before.incoming_color)
        next_incoming = before.remaining_arrivals[0] if before.remaining_arrivals else 0
        remaining = before.remaining_arrivals[1:] if before.remaining_arrivals else ()
        consecutive = before.consecutive_same_color + 1 if painted == before.last_painted_color else 0
        if before.last_painted_color and painted != before.last_painted_color:
            self.color_changes += 1
        after = PlantState(next_incoming, painted, tuple(tuple(b) for b in buffers), remaining, consecutive)
        reward = shaped_reward(before, after, painted)
        self._state = after
        done = after.incoming_color == 0 and not any(after.buffers)
        return after, reward, done, {"color_changes": float(self.color_changes), "painted_color": float(painted)}
