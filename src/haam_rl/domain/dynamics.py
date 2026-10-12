"""Plant physics: which actions are executable and what an action does. Pure functions only.

Action convention. An action is a pair (input buffer, output buffer). When a phase uses only one
half of the pair, the action is written on the diagonal so that every physical move has exactly
one action id:

    FILL   -> (i, i): store the incoming vehicle in buffer i
    STEADY -> (i, o): paint the front of buffer o, then store the incoming vehicle in buffer i
    DRAIN  -> (o, o): paint the front of buffer o

In STEADY the extraction happens first, so (o, o) is valid even when buffer o is full.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from haam_rl.domain.model import (
    ACTION_COUNT,
    BUFFER_CAPACITY,
    BUFFER_COUNT,
    NO_VEHICLE,
    Phase,
    PlantState,
    encode_action,
)

_DIAGONAL = tuple(encode_action(i, i) for i in range(BUFFER_COUNT))
_PAIRS = tuple((encode_action(i, o), i, o) for i in range(BUFFER_COUNT) for o in range(BUFFER_COUNT))
_INSERTS = frozenset({Phase.FILL, Phase.STEADY})
_EXTRACTS = frozenset({Phase.STEADY, Phase.DRAIN})


def phase_of(state: PlantState, fill_target: int) -> Phase:
    if state.incoming_color != NO_VEHICLE:
        return Phase.FILL if state.occupancy < fill_target else Phase.STEADY
    return Phase.DRAIN if state.occupancy > 0 else Phase.DONE


def executable_actions(state: PlantState, fill_target: int) -> list[int]:
    """Action ids that respect FIFO order, capacity and the phase convention above, ascending."""
    phase = phase_of(state, fill_target)
    buffers = state.buffers
    if phase is Phase.FILL:
        return [a for a, b in zip(_DIAGONAL, buffers) if len(b) < BUFFER_CAPACITY]
    if phase is Phase.DRAIN:
        return [a for a, b in zip(_DIAGONAL, buffers) if b]
    if phase is Phase.STEADY:
        return [a for a, i, o in _PAIRS if buffers[o] and (i == o or len(buffers[i]) < BUFFER_CAPACITY)]
    return []


def physical_action_mask(state: PlantState, fill_target: int) -> np.ndarray:
    mask = np.zeros(ACTION_COUNT, dtype=bool)
    mask[executable_actions(state, fill_target)] = True
    return mask


@dataclass(frozen=True)
class Transition:
    state: PlantState
    phase: Phase           # phase in which the action was taken
    painted_color: int     # 0 when nothing was painted (FILL)
    color_changed: bool    # painted a different color than the previous painted vehicle


def _execute(state: PlantState, action: int, phase: Phase) -> Transition:
    """Apply an action already known to be executable in `phase`."""
    input_buffer, output_buffer = divmod(action, BUFFER_COUNT)
    buffers = list(state.buffers)
    occupancy = state.occupancy

    painted = NO_VEHICLE
    if phase in _EXTRACTS:
        painted = buffers[output_buffer][0]
        buffers[output_buffer] = buffers[output_buffer][1:]
        occupancy -= 1
    if phase in _INSERTS:
        buffers[input_buffer] = (*buffers[input_buffer], state.incoming_color)
        occupancy += 1
        remaining = state.remaining_arrivals
        incoming, remaining = (remaining[0], remaining[1:]) if remaining else (NO_VEHICLE, ())
    else:
        incoming, remaining = NO_VEHICLE, ()

    last = state.last_painted_color
    if painted == NO_VEHICLE:
        consecutive, changed = state.consecutive_same_color, False
    else:
        changed = last != NO_VEHICLE and painted != last
        consecutive = state.consecutive_same_color + 1 if painted == last else 0
        last = painted

    next_state = PlantState._trusted(incoming, last, tuple(buffers), remaining, consecutive, occupancy)
    return Transition(next_state, phase, painted, changed)


def apply_action(state: PlantState, action: int, fill_target: int) -> Transition:
    """Execute one physically valid action and return the next state."""
    if action not in executable_actions(state, fill_target):
        raise ValueError(f"Action {action} is not physically executable in this state")
    return _execute(state, action, phase_of(state, fill_target))


def successors(state: PlantState, fill_target: int,
               actions: list[int] | None = None) -> Iterator[tuple[int, Transition]]:
    """Each executable action (or each of `actions`, already known to be executable) with its transition."""
    phase = phase_of(state, fill_target)
    for action in executable_actions(state, fill_target) if actions is None else actions:
        yield action, _execute(state, action, phase)


def is_terminal(state: PlantState) -> bool:
    return state.incoming_color == NO_VEHICLE and state.occupancy == 0
