from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

COLOR_COUNT = 6
BUFFER_COUNT = 5
BUFFER_CAPACITY = 5
ACTION_COUNT = BUFFER_COUNT * BUFFER_COUNT


@dataclass(frozen=True)
class PlantState:
    """Pure state of the paint-shop twin. A buffer is FIFO, with zero meaning empty."""

    incoming_color: int
    last_painted_color: int
    buffers: tuple[tuple[int, ...], ...]
    remaining_arrivals: tuple[int, ...]
    consecutive_same_color: int = 0

    def __post_init__(self) -> None:
        if len(self.buffers) != BUFFER_COUNT or any(len(b) > BUFFER_CAPACITY for b in self.buffers):
            raise ValueError("Expected five FIFO buffers of capacity five")
        for color in (self.incoming_color, self.last_painted_color, *self.remaining_arrivals):
            if not 0 <= color <= COLOR_COUNT:
                raise ValueError("Colors must be in 0..6 (0 means no vehicle)")
        if any(not 1 <= color <= COLOR_COUNT for b in self.buffers for color in b):
            raise ValueError("Stored vehicles must have a real color in 1..6")

    def as_vector(self) -> np.ndarray:
        slots = [color for buffer in self.buffers for color in (*buffer, *([0] * (BUFFER_CAPACITY - len(buffer))))]
        return np.asarray([self.incoming_color, self.last_painted_color, *slots], dtype=np.int32)

    @classmethod
    def from_vector(cls, values: Sequence[int]) -> "PlantState":
        if len(values) != 27:
            raise ValueError("state_vector must have exactly 27 values")
        incoming, last, *slots = (int(value) for value in values)
        buffers = tuple(tuple(color for color in slots[i : i + BUFFER_CAPACITY] if color != 0) for i in range(0, 25, BUFFER_CAPACITY))
        return cls(incoming, last, buffers, ())


def decode_action(action: int) -> tuple[int, int]:
    if not 0 <= action < ACTION_COUNT:
        raise ValueError("Action must be in 0..24")
    return divmod(action, BUFFER_COUNT)


def encode_action(input_buffer: int, output_buffer: int) -> int:
    if not (0 <= input_buffer < BUFFER_COUNT and 0 <= output_buffer < BUFFER_COUNT):
        raise ValueError("Buffer indexes must be in 0..4")
    return input_buffer * BUFFER_COUNT + output_buffer
