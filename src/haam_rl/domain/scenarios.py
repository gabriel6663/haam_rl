"""Arrival sequences. The color mix is the one used by the paper (from Krystek & Bysko, 2019)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from haam_rl.domain.model import BUFFER_COUNT, COLOR_COUNT, NO_VEHICLE, PlantState

PAPER_COLOR_PROBABILITIES = (0.06, 0.38, 0.29, 0.14, 0.10, 0.03)

# Disjoint seed ranges so that model selection never sees the test scenarios.
TEST_SEEDS = range(10_000, 10_030)           # 30 scenarios, the size of the paper's protocol
VALIDATION_SEEDS = range(20_000, 20_100)     # model selection during training
EXTENDED_TEST_SEEDS = range(30_000, 30_500)  # 500 scenarios for tighter estimates (SE ~0.15 changes)


def generate_arrivals(seed: int | None, vehicles: int, probabilities: Sequence[float] = PAPER_COLOR_PROBABILITIES) -> tuple[int, ...]:
    if len(probabilities) != COLOR_COUNT or not np.isclose(sum(probabilities), 1.0):
        raise ValueError(f"Need {COLOR_COUNT} color probabilities that sum to 1")
    rng = np.random.default_rng(seed)
    return tuple(int(c) + 1 for c in rng.choice(COLOR_COUNT, size=vehicles, p=probabilities))


def initial_state(arrivals: Sequence[int]) -> PlantState:
    """Empty buffers, the first vehicle waiting, nothing painted yet."""
    if not arrivals:
        raise ValueError("A scenario needs at least one vehicle")
    first, *rest = arrivals
    return PlantState(first, NO_VEHICLE, ((),) * BUFFER_COUNT, tuple(rest))
