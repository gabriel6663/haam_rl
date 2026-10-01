from __future__ import annotations

from typing import Protocol

import numpy as np

from haam_rl.domain.model import PlantState


class PlantPort(Protocol):
    @property
    def state(self) -> PlantState: ...

    def reset(self, seed: int | None = None) -> PlantState: ...

    def step(self, action: int) -> tuple[PlantState, float, bool, dict[str, float]]: ...

    def action_mask(self) -> np.ndarray: ...
