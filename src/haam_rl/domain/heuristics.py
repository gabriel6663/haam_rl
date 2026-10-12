"""Heuristic rules from the paper (Section III-B) and the action masks built from them.

Input rules (where to store the incoming vehicle):
    CM  Color Memory     - a buffer whose tail has the incoming color.
    LP  Lowest Priority  - otherwise, the buffers with the fewest distinct colors (empty = 0).
Output rules (which buffer front to paint):
    CM  Color Memory     - a buffer whose front has the last painted color.
    UCM Unseen CM        - otherwise, the buffers with the most distinct colors.
Remaining ties go to the top buffer (lowest index), which defines the heuristic baseline policy.
"""
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import numpy as np

from haam_rl.domain.dynamics import executable_actions, phase_of
from haam_rl.domain.model import (
    ACTION_COUNT,
    BUFFER_COUNT,
    Buffer,
    Phase,
    PlantState,
    decode_action,
    encode_action,
)


class MaskMode(str, Enum):
    """Which rules filter the physically valid actions (used for ablations)."""

    HAAM = "haam"                # input rules + output rules, as in the paper
    HAAM_OUTPUT = "haam_output"  # output rules only; any physically valid input
    PHYSICAL = "physical"        # no heuristic filtering


def _diversity(buffer: Buffer) -> int:
    return len(set(buffer))


def input_candidates(buffers: tuple[Buffer, ...], incoming_color: int, allowed: Iterable[int]) -> set[int]:
    allowed = set(allowed)
    color_memory = {i for i in allowed if buffers[i] and buffers[i][-1] == incoming_color}
    if color_memory:
        return color_memory
    lowest = min(_diversity(buffers[i]) for i in allowed)
    return {i for i in allowed if _diversity(buffers[i]) == lowest}


def output_candidates(buffers: tuple[Buffer, ...], last_painted_color: int, allowed: Iterable[int]) -> set[int]:
    allowed = set(allowed)
    color_memory = {o for o in allowed if buffers[o][0] == last_painted_color}
    if color_memory:
        return color_memory
    most = max(_diversity(buffers[o]) for o in allowed)
    return {o for o in allowed if _diversity(buffers[o]) == most}


def _without_front(buffers: tuple[Buffer, ...], index: int) -> tuple[Buffer, ...]:
    return tuple(b[1:] if k == index else b for k, b in enumerate(buffers))


def allowed_actions(state: PlantState, fill_target: int, mode: MaskMode = MaskMode.HAAM) -> list[int]:
    """Executable actions that pass the heuristic rules selected by `mode`, ascending.

    In STEADY the input rules are evaluated on the buffers *after* the chosen extraction, because
    that is what the incoming vehicle will actually see.
    """
    executable = executable_actions(state, fill_target)
    if mode is MaskMode.PHYSICAL or not executable:
        return executable

    use_inputs = mode is MaskMode.HAAM
    buffers, incoming = state.buffers, state.incoming_color
    if phase_of(state, fill_target) is Phase.FILL:  # actions are (i, i): only the input matters
        valid = [a // BUFFER_COUNT for a in executable]
        chosen = input_candidates(buffers, incoming, valid) if use_inputs else valid
        return sorted(encode_action(i, i) for i in chosen)

    inputs_by_output: dict[int, list[int]] = {}
    for action in executable:
        i, o = divmod(action, BUFFER_COUNT)
        inputs_by_output.setdefault(o, []).append(i)
    allowed = []
    for o in output_candidates(buffers, state.last_painted_color, inputs_by_output):
        inputs = inputs_by_output[o]  # in DRAIN this is just [o]
        chosen = input_candidates(_without_front(buffers, o), incoming, inputs) if use_inputs and incoming else inputs
        allowed.extend(encode_action(i, o) for i in chosen)
    return sorted(allowed)


def action_mask(state: PlantState, fill_target: int, mode: MaskMode = MaskMode.HAAM) -> np.ndarray:
    """Boolean mask over the 25 actions; see `allowed_actions`."""
    mask = np.zeros(ACTION_COUNT, dtype=bool)
    mask[allowed_actions(state, fill_target, mode)] = True
    return mask


def heuristic_action(state: PlantState, fill_target: int) -> int:
    """Deterministic baseline: HAAM rules, ties broken by the top output buffer, then top input buffer."""
    candidates = allowed_actions(state, fill_target, MaskMode.HAAM)
    if not candidates:
        raise ValueError("No executable action: the episode is over")
    return int(min(candidates, key=lambda a: decode_action(int(a))[::-1]))
