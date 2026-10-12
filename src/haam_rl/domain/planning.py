"""Beam-search planning over the plant rules, replanned at every decision.

`lookahead_action` sees the waiting vehicle plus the next `horizon` arrivals. A real plant (and
FlexSim's waiting queue) knows a short horizon, so this is a deployable policy.

With `horizon=None` it sees every future arrival. That is not deployable; it is the reference
("oracle") that shows how few color changes a scenario allows. Replanning at every decision
matters: a single beam-search pass over the whole scenario finds clearly worse schedules.

`mask_mode` restricts the search to the actions a mask allows. With the HAAM mask and full
information it measures the ceiling of any policy that must obey the paper's heuristic mask.

Plans are ranked by their color changes plus an estimate of what the state they leave behind will
cost later (`LeafEstimate`). The default estimate is weight x fragmentation; a learned value
function can be plugged in instead (see `haam_rl.adapters.value_search`).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from haam_rl.domain.dynamics import is_terminal, successors
from haam_rl.domain.heuristics import MaskMode, allowed_actions
from haam_rl.domain.model import NO_VEHICLE, PlantState

# Beam entry: (score, color changes so far, state, first action of the plan; None at the root)
_Entry = tuple[float, int, PlantState, int | None]
# Estimated future color changes of many states at once (batched, so a neural network can be used).
LeafEstimate = Callable[[Sequence[PlantState]], Sequence[float]]


def fragmentation(state: PlantState) -> int:
    """Extra colors per buffer: 0 when every buffer holds a single color."""
    return sum(max(0, len(set(b)) - 1) for b in state.buffers)


@dataclass(frozen=True)
class FragmentationEstimate:
    """The default leaf estimate: weight x fragmentation (a picklable callable)."""

    weight: float

    def __call__(self, states: Sequence[PlantState]) -> list[float]:
        return [self.weight * fragmentation(s) for s in states]


def fragmentation_estimate(weight: float) -> LeafEstimate:
    return FragmentationEstimate(weight)


def _best_plan(root: PlantState, fill_target: int, width: int, estimate: LeafEstimate,
               finished: Callable[[PlantState], bool], mask_mode: MaskMode) -> _Entry:
    """Expand every allowed action, deduplicate, keep the `width` best; return the best leaf.

    All entries in the beam have made the same number of decisions, so they share the arrival
    queue and the phase; two entries with the same buffers and last color are duplicates.
    Entries are ranked by color changes so far + the estimated cost of the state left behind.
    """
    beam: list[_Entry] = [(0.0, 0, root, None)]
    while not finished(beam[0][2]):
        best: dict[tuple, tuple[int, PlantState, int]] = {}
        for _, changes, state, first in beam:
            allowed = None if mask_mode is MaskMode.PHYSICAL else allowed_actions(state, fill_target, mask_mode)
            for action, transition in successors(state, fill_target, allowed):
                total = changes + transition.color_changed
                key = (transition.state.buffers, transition.state.last_painted_color)
                if key not in best or best[key][0] > total:
                    best[key] = (total, transition.state, action if first is None else first)
        candidates = list(best.values())
        estimates = estimate([state for _, state, _ in candidates])
        scored = [(total + future, total, state, first) for (total, state, first), future in zip(candidates, estimates)]
        # Stable sort on the score alone: ties keep insertion order (lower action ids first).
        beam = sorted(scored, key=lambda entry: entry[0])[:width]
    return beam[0]


@dataclass(frozen=True)
class KnownFuture:
    """What a decision can plan over: the state cut to the known arrivals, and when a plan must stop."""

    root: PlantState
    more_unknown: bool  # vehicles beyond the known ones may still arrive

    def finished(self, state: PlantState) -> bool:
        return is_terminal(state) or (self.more_unknown and state.incoming_color == NO_VEHICLE)


def known_future(state: PlantState, horizon: int | None) -> KnownFuture:
    """horizon=None: every arrival in `state.remaining_arrivals` is known (perfect information)."""
    if horizon is None:
        return KnownFuture(state, False)
    more_unknown = state.incoming_color != NO_VEHICLE and len(state.remaining_arrivals) >= horizon
    return KnownFuture(replace(state, remaining_arrivals=state.remaining_arrivals[:horizon]), more_unknown)


def best_plan(state: PlantState, fill_target: int, horizon: int | None = 10, width: int = 32,
              fragmentation_weight: float = 0.2, mask_mode: MaskMode = MaskMode.PHYSICAL,
              estimate: LeafEstimate | None = None) -> tuple[float, int]:
    """(score, first action) of the best plan the beam finds over the known vehicles.

    `estimate` replaces the default leaf estimate (fragmentation_weight x fragmentation)."""
    future = known_future(state, horizon)
    estimate = estimate or fragmentation_estimate(fragmentation_weight)
    score, _, _, first = _best_plan(future.root, fill_target, width, estimate, future.finished, mask_mode)
    if first is None:
        raise ValueError("No executable action: the episode is over")
    return score, first


def lookahead_action(state: PlantState, fill_target: int, horizon: int | None = 10, width: int = 32,
                     fragmentation_weight: float = 0.2, mask_mode: MaskMode = MaskMode.PHYSICAL) -> int:
    """First action of the best plan over the vehicles that are known.

    horizon=None uses every arrival in `state.remaining_arrivals` (perfect information).
    Otherwise the plan stops when the known vehicles run out and more may follow. Without the
    fragmentation term (weight 0) the planner is myopic and no better than the heuristic.
    """
    return best_plan(state, fill_target, horizon, width, fragmentation_weight, mask_mode)[1]
