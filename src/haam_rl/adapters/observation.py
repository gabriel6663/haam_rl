"""Turns a PlantState into the network input vector.

RAW    (paper): [incoming, last painted, 25 slots, k look-ahead] as plain numbers 0..6.
ONEHOT        : each of those positions as a 7-way indicator (0 = empty, 1..6 = color), plus
                the phase (fill/steady/drain) and the buffer occupancy as a fraction.

Raw numbers impose a false order on colors (color 6 is not "more" than color 1); one-hot removes it.

Every encoder also describes its input as *groups* (one per slot, one per look-ahead position...)
so the web inspector can draw one node per meaningful input instead of one per indicator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from haam_rl.config import Encoding, ObservationConfig
from haam_rl.domain.dynamics import phase_of
from haam_rl.domain.model import (
    BUFFER_CAPACITY,
    BUFFER_COUNT,
    COLOR_COUNT,
    TOTAL_SLOTS,
    Phase,
    PlantState,
)

_PHASES = (Phase.FILL, Phase.STEADY, Phase.DRAIN)


@dataclass(frozen=True)
class FeatureGroup:
    label: str
    start: int   # first index in the encoded vector
    stop: int    # one past the last index
    kind: str    # "color" (a vehicle position), "phase" or "scalar"


def _position_labels(lookahead: int) -> list[str]:
    slots = [f"B{b + 1}.{s + 1}" for b in range(BUFFER_COUNT) for s in range(BUFFER_CAPACITY)]
    return ["entrante", "ultimo", *slots, *(f"llega+{k + 1}" for k in range(lookahead))]


class ObservationEncoder:
    def __init__(self, config: ObservationConfig, fill_target: int) -> None:
        self.config = config
        self.fill_target = fill_target
        labels = _position_labels(config.lookahead)
        width = 1 if config.encoding is Encoding.RAW else COLOR_COUNT + 1
        groups = [FeatureGroup(label, i * width, (i + 1) * width, "color") for i, label in enumerate(labels)]
        if config.encoding is Encoding.ONEHOT and config.include_phase:
            offset = groups[-1].stop
            groups.append(FeatureGroup("fase", offset, offset + len(_PHASES), "phase"))
            groups.append(FeatureGroup("ocupacion", offset + len(_PHASES), offset + len(_PHASES) + 1, "scalar"))
        self.groups = tuple(groups)
        self.size = groups[-1].stop

    def positions(self, state: PlantState) -> np.ndarray:
        """The color at every vehicle position: the paper's 27 values followed by the look-ahead."""
        ahead = list(state.remaining_arrivals[: self.config.lookahead])
        ahead += [0] * (self.config.lookahead - len(ahead))
        return np.concatenate([state.as_vector(), np.asarray(ahead, dtype=np.int64)])

    def encode(self, state: PlantState) -> np.ndarray:
        positions = self.positions(state)
        if self.config.encoding is Encoding.RAW:
            return positions.astype(np.float32)
        onehot = np.zeros((positions.size, COLOR_COUNT + 1), dtype=np.float32)
        onehot[np.arange(positions.size), positions] = 1.0
        parts = [onehot.ravel()]
        if self.config.include_phase:
            phase = phase_of(state, self.fill_target)
            parts.append(np.asarray([phase is p for p in _PHASES], dtype=np.float32))
            parts.append(np.asarray([state.occupancy / TOTAL_SLOTS], dtype=np.float32))
        return np.concatenate(parts)

    def bounds(self) -> tuple[float, float]:
        return (0.0, float(COLOR_COUNT)) if self.config.encoding is Encoding.RAW else (0.0, 1.0)

    def raw_labels(self) -> list[str]:
        """One label per encoded feature (used by the weight heatmap)."""
        labels = []
        for group in self.groups:
            if group.stop - group.start == 1:
                labels.append(group.label)
            elif group.kind == "phase":
                labels += [f"fase={p.value}" for p in _PHASES]
            else:
                labels += [f"{group.label}={c}" for c in range(group.stop - group.start)]
        return labels
