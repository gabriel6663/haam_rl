from __future__ import annotations

import numpy as np

from haam_rl.domain.model import ACTION_COUNT, BUFFER_CAPACITY, PlantState, decode_action


def _diversity(buffer: tuple[int, ...]) -> int:
    return len(set(buffer))


def physical_action_mask(state: PlantState) -> np.ndarray:
    """Actions that preserve FIFO and buffer-capacity constraints."""
    mask = np.zeros(ACTION_COUNT, dtype=bool)
    for action in range(ACTION_COUNT):
        input_buffer, output_buffer = decode_action(action)
        can_output = bool(state.buffers[output_buffer])
        if state.incoming_color == 0:
            # Drain phase: the input half of the pair is a no-op once the arrival queue is empty.
            mask[action] = can_output
            continue
        has_space = len(state.buffers[input_buffer]) < BUFFER_CAPACITY
        frees_same_buffer = input_buffer == output_buffer and can_output
        mask[action] = can_output and (has_space or frees_same_buffer)
    return mask


def candidate_input_buffers(state: PlantState, physical_mask: np.ndarray) -> set[int]:
    """CM first; when it has no match, LP chooses least color-diverse buffers."""
    valid_inputs = {decode_action(a)[0] for a in np.flatnonzero(physical_mask)}
    if state.incoming_color == 0:
        return valid_inputs
    cm = {i for i in valid_inputs if state.buffers[i] and state.buffers[i][-1] == state.incoming_color}
    if cm:
        return cm
    lowest = min(_diversity(state.buffers[i]) for i in valid_inputs)
    return {i for i in valid_inputs if _diversity(state.buffers[i]) == lowest}


def candidate_output_buffers(state: PlantState, physical_mask: np.ndarray) -> set[int]:
    """CM on the next paintable vehicle; UCM breaks the no-match case."""
    valid_outputs = {decode_action(a)[1] for a in np.flatnonzero(physical_mask)}
    cm = {i for i in valid_outputs if state.buffers[i][0] == state.last_painted_color}
    if cm:
        return cm
    most_diverse = max(_diversity(state.buffers[i]) for i in valid_outputs)
    return {i for i in valid_outputs if _diversity(state.buffers[i]) == most_diverse}


def haam_action_mask(state: PlantState) -> np.ndarray:
    """HAAM candidate set: physical constraints intersected with LP/CM/UCM rules."""
    physical = physical_action_mask(state)

    if not physical.any():
        return physical

    inputs = candidate_input_buffers(state, physical)
    outputs = candidate_output_buffers(state, physical)

    heuristic_mask = np.asarray(
        [
            physical[action]
            and decode_action(action)[0] in inputs
            and decode_action(action)[1] in outputs
            for action in range(ACTION_COUNT)
        ],
        dtype=bool,
    )

    # Si las heurísticas son incompatibles, conservar las acciones físicas válidas
    return heuristic_mask if heuristic_mask.any() else physical
