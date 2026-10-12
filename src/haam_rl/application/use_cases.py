"""Policies that need no learning, and the paper's ensemble inference method (Algorithm 2)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from haam_rl.application.ports import Decision, Policy
from haam_rl.domain.heuristics import MaskMode, action_mask, heuristic_action
from haam_rl.domain.model import ACTION_COUNT, PlantState
from haam_rl.domain.planning import lookahead_action


class HeuristicPolicy:
    """The baseline the paper compares against: LP/CM/UCM with top-buffer tie-break."""

    name = "heuristic"

    def __init__(self, fill_target: int) -> None:
        self.fill_target = fill_target

    def decide(self, state: PlantState) -> Decision:
        action = heuristic_action(state, self.fill_target)
        probabilities = np.zeros(ACTION_COUNT)
        probabilities[action] = 1.0
        return Decision(action, probabilities, action_mask(state, self.fill_target, MaskMode.HAAM))


class LookaheadPlannerPolicy:
    """Replans with beam search over the vehicles it can see and executes the first action.

    Uses the same information as an RL agent with look-ahead: the waiting vehicle plus `horizon`
    queued ones. No training. See `haam_rl.domain.planning.lookahead_action`.
    horizon=None sees every future arrival: the perfect-information reference ("oracle").
    mask_mode restricts the search to the actions that mask allows (e.g. HAAM, to measure the
    ceiling of policies that obey the paper's mask).
    """

    def __init__(self, fill_target: int, horizon: int | None = 10, width: int = 32,
                 fragmentation_weight: float = 0.2, mask_mode: MaskMode = MaskMode.PHYSICAL) -> None:
        self.fill_target = fill_target
        self.horizon = horizon
        self.width = width
        self.fragmentation_weight = fragmentation_weight
        self.mask_mode = MaskMode(mask_mode)
        mask = "" if self.mask_mode is MaskMode.PHYSICAL else f",mask={self.mask_mode.value}"
        self.name = f"oracle(w={width}{mask})" if horizon is None else f"planner(h={horizon},w={width}{mask})"

    def decide(self, state: PlantState) -> Decision:
        action = lookahead_action(state, self.fill_target, self.horizon, self.width,
                                  self.fragmentation_weight, self.mask_mode)
        probabilities = np.zeros(ACTION_COUNT)
        probabilities[action] = 1.0
        return Decision(action, probabilities, action_mask(state, self.fill_target, self.mask_mode),
                        {"horizon": self.horizon, "width": self.width})


@dataclass(frozen=True)
class Vote:
    action: int
    mode: str                    # "hard" or "soft"
    member_actions: tuple[int, ...]
    top_count: int
    threshold: float
    scores: np.ndarray           # weighted sum of member probabilities


def ensemble_vote(member_probabilities: np.ndarray, weights: Sequence[float] | None = None,
                  hard_threshold: float | None = None) -> Vote:
    """Hard voting when enough members agree, soft voting otherwise (paper, Eq. 2-4).

    Each member votes for its most likely masked action. With M members the default threshold is
    M/2 - 1 (Eq. 2) and hard voting is used when the top action has strictly more votes (Eq. 3).
    Note: the paper's prose says ">= 4 of 10", which matches `> M/2 - 2`; pass `hard_threshold`
    to reproduce either reading. Soft voting sums the members' masked probabilities (Eq. 4).
    """
    probabilities = np.asarray(member_probabilities, dtype=float)
    members = probabilities.shape[0]
    weights = np.ones(members) if weights is None else np.asarray(weights, dtype=float)
    threshold = members / 2 - 1 if hard_threshold is None else hard_threshold

    member_actions = probabilities.argmax(axis=1)
    counts = np.bincount(member_actions, minlength=probabilities.shape[1])
    scores = weights @ probabilities
    top = int(counts.max())
    if top > threshold:
        tied = np.flatnonzero(counts == top)
        action, mode = int(tied[np.argmax(scores[tied])]), "hard"  # ties between top actions -> soft score
    else:
        action, mode = int(np.argmax(scores)), "soft"
    return Vote(action, mode, tuple(int(a) for a in member_actions), top, float(threshold), scores)


class EnsemblePolicy:
    def __init__(self, members: Sequence[Policy], weights: Sequence[float] | None = None,
                 hard_threshold: float | None = None) -> None:
        if not members:
            raise ValueError("An ensemble needs at least one member")
        fill_targets = {m.fill_target for m in members}
        if len(fill_targets) != 1:
            raise ValueError(f"Ensemble members were trained on different plants (fill_target {fill_targets})")
        self.members = list(members)
        self.weights = weights
        self.hard_threshold = hard_threshold
        self.fill_target = fill_targets.pop()
        self.name = f"ensemble({len(self.members)})"

    def decide(self, state: PlantState) -> Decision:
        return self._combine([member.decide(state) for member in self.members])

    def decide_batch(self, states: Sequence[PlantState]) -> list[Decision]:
        """One batched call per member (when supported), then one vote per state."""
        per_member = [decide_all(member, states) for member in self.members]
        return [self._combine(list(decisions)) for decisions in zip(*per_member)]

    def _combine(self, decisions: list[Decision]) -> Decision:
        vote = ensemble_vote(np.stack([d.probabilities for d in decisions]), self.weights, self.hard_threshold)
        total = vote.scores.sum()
        mask = np.logical_or.reduce([d.mask for d in decisions])
        return Decision(vote.action, vote.scores / total if total else vote.scores, mask, {
            "vote_mode": vote.mode,
            "member_actions": list(vote.member_actions),
            "top_count": vote.top_count,
            "threshold": vote.threshold,
        })


def decide_all(policy: Policy, states: Sequence[PlantState]) -> list[Decision]:
    """Decisions for many states, batched when the policy implements `decide_batch`."""
    batch = getattr(policy, "decide_batch", None)
    return list(batch(states)) if batch is not None else [policy.decide(s) for s in states]
