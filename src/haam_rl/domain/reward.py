from __future__ import annotations

from haam_rl.domain.model import PlantState


def potential(state: PlantState) -> float:
    """State potential: lower color fragmentation and available color match are better."""
    fragmentation = sum(max(0, len(set(buffer)) - 1) for buffer in state.buffers)
    matching_fronts = sum(bool(buffer and buffer[0] == state.last_painted_color) for buffer in state.buffers)
    return -0.05 * fragmentation + 0.02 * matching_fronts


def shaped_reward(before: PlantState, after: PlantState, painted_color: int, gamma: float = 0.99) -> float:
    changed_color = painted_color != before.last_painted_color
    car_reward = -0.1 if changed_color else 0.1 + 0.02 * after.consecutive_same_color
    return car_reward + gamma * potential(after) - potential(before)
