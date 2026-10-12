"""Reward of the paper (Section III-A): R = C + P.

C, car reward:  -change_penalty when the painted color differs from the previous one,
                +same_color_reward + streak_bonus * streak otherwise (the paper's "0.1 + dup").
P, potential-based shaping (Ng et al., 1999): gamma * phi(s') - phi(s), with phi(terminal) = 0,
                so the optimal policy is unchanged by the shaping term.
"""
from __future__ import annotations

from dataclasses import dataclass

from haam_rl.domain.dynamics import Transition, is_terminal
from haam_rl.domain.model import NO_VEHICLE, PlantState


@dataclass(frozen=True)
class RewardConfig:
    change_penalty: float = 0.1
    same_color_reward: float = 0.1
    streak_bonus: float = 0.02
    fragmentation_weight: float = 0.05
    front_match_weight: float = 0.02
    use_shaping: bool = True


def potential(state: PlantState, config: RewardConfig) -> float:
    """Higher when buffers hold few colors each and some front matches the last painted color."""
    if is_terminal(state):
        return 0.0
    fragmentation = sum(max(0, len(set(buffer)) - 1) for buffer in state.buffers)
    matching_fronts = sum(
        bool(buffer) and buffer[0] == state.last_painted_color for buffer in state.buffers
    ) if state.last_painted_color != NO_VEHICLE else 0
    return -config.fragmentation_weight * fragmentation + config.front_match_weight * matching_fronts


def car_reward(before: PlantState, transition: Transition, config: RewardConfig) -> float:
    if transition.painted_color == NO_VEHICLE or before.last_painted_color == NO_VEHICLE:
        return 0.0  # nothing painted yet, or the very first vehicle: no change can be judged
    if transition.color_changed:
        return -config.change_penalty
    return config.same_color_reward + config.streak_bonus * transition.state.consecutive_same_color


def reward(before: PlantState, transition: Transition, config: RewardConfig, gamma: float) -> float:
    value = car_reward(before, transition, config)
    if config.use_shaping:
        value += gamma * potential(transition.state, config) - potential(before, config)
    return value
