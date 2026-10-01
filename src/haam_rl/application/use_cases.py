from __future__ import annotations

import numpy as np

from haam_rl.domain.heuristics import haam_action_mask
from haam_rl.domain.model import PlantState


def allowed_actions(state: PlantState) -> np.ndarray:
    return haam_action_mask(state)
