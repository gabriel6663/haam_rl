"""Core vocabulary of the paint shop: colors, buffers, actions and the plant state.

Layout (Choi et al., 2024, Fig. 2): one arrival queue, five FIFO buffers with five slots each,
and the painting machine. Colors are integers 1..6; 0 always means "no vehicle".
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

COLOR_COUNT = 6
BUFFER_COUNT = 5
BUFFER_CAPACITY = 5
TOTAL_SLOTS = BUFFER_COUNT * BUFFER_CAPACITY
ACTION_COUNT = BUFFER_COUNT * BUFFER_COUNT
STATE_VECTOR_SIZE = 2 + TOTAL_SLOTS  # incoming color, last painted color, 25 slots

NO_VEHICLE = 0

Buffer = tuple[int, ...]  # FIFO: index 0 is the front (next to be painted), index -1 the tail


class Phase(str, Enum):
    """What one decision does, which depends only on the current state.

    FILL:   a vehicle is waiting and the buffers hold fewer than `fill_target` vehicles.
            The incoming vehicle is stored; nothing is painted.
    STEADY: a vehicle is waiting and the buffers are at `fill_target`.
            One vehicle is extracted and painted, then the incoming one is stored.
    DRAIN:  no vehicle is waiting. One vehicle is extracted and painted.
    DONE:   nothing waiting and every buffer empty.
    """

    FILL = "fill"
    STEADY = "steady"
    DRAIN = "drain"
    DONE = "done"

    @property
    def inserts(self) -> bool:
        return self in (Phase.FILL, Phase.STEADY)

    @property
    def extracts(self) -> bool:
        return self in (Phase.STEADY, Phase.DRAIN)


@dataclass(frozen=True)
class PlantState:
    """Immutable snapshot of the plant.

    `remaining_arrivals` are the vehicles queued behind `incoming_color`. In the twin it is the
    whole rest of the scenario; through the HTTP API it is whatever look-ahead the caller sends.

    Every state built with `PlantState(...)` is validated. The plant dynamics build their output
    states with `_trusted`, which skips validation: those states are valid by construction and
    the planner creates hundreds of thousands of them per scenario.
    """

    incoming_color: int
    last_painted_color: int
    buffers: tuple[Buffer, ...]
    remaining_arrivals: tuple[int, ...] = ()
    consecutive_same_color: int = 0
    occupancy: int = field(init=False, repr=False, compare=False)  # vehicles stored in the buffers

    def __post_init__(self) -> None:
        if len(self.buffers) != BUFFER_COUNT or any(len(b) > BUFFER_CAPACITY for b in self.buffers):
            raise ValueError(f"Expected {BUFFER_COUNT} FIFO buffers of capacity {BUFFER_CAPACITY}")
        for color in (self.incoming_color, self.last_painted_color):
            if not NO_VEHICLE <= color <= COLOR_COUNT:
                raise ValueError(f"Colors must be in 0..{COLOR_COUNT} (0 means no vehicle)")
        for color in (*self.remaining_arrivals, *(c for b in self.buffers for c in b)):
            if not 1 <= color <= COLOR_COUNT:
                raise ValueError(f"Queued and stored vehicles must have a real color in 1..{COLOR_COUNT}")
        object.__setattr__(self, "occupancy", sum(len(buffer) for buffer in self.buffers))

    @classmethod
    def _trusted(cls, incoming_color: int, last_painted_color: int, buffers: tuple[Buffer, ...],
                 remaining_arrivals: tuple[int, ...], consecutive_same_color: int, occupancy: int) -> PlantState:
        """Build a state without validation. Only for the dynamics, whose outputs are valid by construction."""
        state = object.__new__(cls)
        for name, value in (("incoming_color", incoming_color), ("last_painted_color", last_painted_color),
                            ("buffers", buffers), ("remaining_arrivals", remaining_arrivals),
                            ("consecutive_same_color", consecutive_same_color), ("occupancy", occupancy)):
            object.__setattr__(state, name, value)
        return state

    def as_vector(self) -> np.ndarray:
        """The paper's 27-value state: incoming, last painted, then 5 buffers x 5 slots (front first)."""
        slots = [color for buffer in self.buffers for color in (*buffer, *([NO_VEHICLE] * (BUFFER_CAPACITY - len(buffer))))]
        return np.asarray([self.incoming_color, self.last_painted_color, *slots], dtype=np.int64)

    @classmethod
    def from_vector(cls, values: Sequence[int], upcoming: Sequence[int] = ()) -> PlantState:
        """Inverse of `as_vector`. Empty slots inside a buffer are compacted towards the front."""
        if len(values) != STATE_VECTOR_SIZE:
            raise ValueError(f"state_vector must have exactly {STATE_VECTOR_SIZE} values")
        incoming, last, *slots = (int(value) for value in values)
        buffers = tuple(
            tuple(color for color in slots[i : i + BUFFER_CAPACITY] if color != NO_VEHICLE)
            for i in range(0, TOTAL_SLOTS, BUFFER_CAPACITY)
        )
        return cls(incoming, last, buffers, tuple(int(c) for c in upcoming if c != NO_VEHICLE))


def decode_action(action: int) -> tuple[int, int]:
    """Action id -> (input buffer, output buffer), both 0-based. Matches FlexSim: id = in * 5 + out."""
    if not 0 <= action < ACTION_COUNT:
        raise ValueError(f"Action must be in 0..{ACTION_COUNT - 1}")
    return divmod(action, BUFFER_COUNT)


def encode_action(input_buffer: int, output_buffer: int) -> int:
    if not (0 <= input_buffer < BUFFER_COUNT and 0 <= output_buffer < BUFFER_COUNT):
        raise ValueError(f"Buffer indexes must be in 0..{BUFFER_COUNT - 1}")
    return input_buffer * BUFFER_COUNT + output_buffer
