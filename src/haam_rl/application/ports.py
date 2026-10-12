"""Interfaces the application layer depends on. Adapters implement them."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from haam_rl.domain.model import PlantState


@dataclass(frozen=True)
class StepInfo:
    phase: str
    painted_color: int
    color_changed: bool
    color_changes: int


class PlantPort(Protocol):
    """A plant that can be stepped: the Python twin, or any simulator wrapped to look like it."""

    @property
    def state(self) -> PlantState: ...

    def reset(self, seed: int | None = None) -> PlantState: ...

    def step(self, action: int) -> tuple[PlantState, float, bool, StepInfo]: ...


@dataclass
class Decision:
    """What a policy chose and why.

    `probabilities` is the distribution over the 25 actions after masking (zero on masked actions);
    `details` holds policy-specific extras, e.g. each ensemble member's vote.
    """

    action: int
    probabilities: np.ndarray
    mask: np.ndarray
    details: dict = field(default_factory=dict)


class Policy(Protocol):
    name: str
    fill_target: int

    def decide(self, state: PlantState) -> Decision: ...
