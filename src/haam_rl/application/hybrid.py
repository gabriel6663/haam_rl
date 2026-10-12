"""Policies that combine search with other policies.

Every decision maker here scores a candidate first action the same way: simulate the vehicles that
are known (the waiting one plus the look-ahead) and add the color changes of the simulated plan to
an estimate of what the state it leaves will cost later (`LeafEstimate`: weight x fragmentation by
default, or a learned value network). Because the scores share one scale, proposals from different
decision makers can be compared directly. The referee is only as good as that estimate.

RolloutPolicy   - the rollout algorithm (Bertsekas): for every allowed first action, finish the
                  known horizon with a base policy (e.g. a trained network) and keep the action
                  whose simulated plan scores best. It improves on its base policy by construction
                  of the score, at the cost of one short simulation per candidate action.
CommitteePolicy - several decision makers propose actions with their simulated scores (beam
                  planner, rollouts with different base policies...); a common referee keeps the
                  best proposal. This is "several models deciding together", arbitrated by
                  simulation instead of by votes.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from haam_rl.application.ports import Decision, Policy
from haam_rl.application.use_cases import decide_all
from haam_rl.domain.dynamics import apply_action
from haam_rl.domain.heuristics import MaskMode, action_mask, allowed_actions
from haam_rl.domain.model import ACTION_COUNT, PlantState
from haam_rl.domain.planning import LeafEstimate, best_plan, fragmentation_estimate, known_future


class Proposer(Protocol):
    name: str

    def proposals(self, state: PlantState) -> dict[int, float]:
        """Candidate first actions with the score of the plan that starts with each (lower is better)."""
        ...


def _decision(action: int, state: PlantState, fill_target: int, mask_mode: MaskMode, details: dict) -> Decision:
    probabilities = np.zeros(ACTION_COUNT)
    probabilities[action] = 1.0
    return Decision(action, probabilities, action_mask(state, fill_target, mask_mode), details)


def _best(proposals: dict[int, float]) -> int:
    return min(proposals, key=lambda action: (proposals[action], action))  # ties: lowest action id


class PlannerProposer:
    """The beam-search planner as a committee member: proposes its best plan's first action."""

    def __init__(self, fill_target: int, horizon: int | None = 10, width: int = 32,
                 estimate: LeafEstimate | None = None, mask_mode: MaskMode = MaskMode.PHYSICAL) -> None:
        self.fill_target, self.horizon, self.width = fill_target, horizon, width
        self.estimate, self.mask_mode = estimate or fragmentation_estimate(0.2), MaskMode(mask_mode)
        self.name = f"planner(h={horizon},w={width})"

    def proposals(self, state: PlantState) -> dict[int, float]:
        score, action = best_plan(state, self.fill_target, self.horizon, self.width,
                                  mask_mode=self.mask_mode, estimate=self.estimate)
        return {action: score}

    def decide(self, state: PlantState) -> Decision:
        proposals = self.proposals(state)
        return _decision(_best(proposals), state, self.fill_target, self.mask_mode, {"scores": proposals})


class RolloutPolicy:
    def __init__(self, base: Policy, fill_target: int, horizon: int | None = 10,
                 estimate: LeafEstimate | None = None, mask_mode: MaskMode = MaskMode.PHYSICAL) -> None:
        self.base = base
        self.fill_target = fill_target
        self.horizon = horizon
        self.estimate = estimate or fragmentation_estimate(0.2)
        self.mask_mode = MaskMode(mask_mode)
        self.name = f"rollout({getattr(base, 'name', 'base')},h={horizon})"

    def proposals(self, state: PlantState) -> dict[int, float]:
        future = known_future(state, self.horizon)
        simulations = []  # [first action, current state, color changes so far]
        for action in allowed_actions(state, self.fill_target, self.mask_mode):
            transition = apply_action(future.root, action, self.fill_target)
            simulations.append([action, transition.state, int(transition.color_changed)])
        running = [k for k, sim in enumerate(simulations) if not future.finished(sim[1])]
        while running:  # all candidate plans advance together: one batched base decision per step
            decisions = decide_all(self.base, [simulations[k][1] for k in running])
            still_running = []
            for k, decision in zip(running, decisions):
                transition = apply_action(simulations[k][1], decision.action, self.fill_target)
                simulations[k][1] = transition.state
                simulations[k][2] += transition.color_changed
                if not future.finished(transition.state):
                    still_running.append(k)
            running = still_running
        futures = self.estimate([leaf for _, leaf, _ in simulations])
        return {action: changes + future for (action, _, changes), future in zip(simulations, futures)}

    def decide(self, state: PlantState) -> Decision:
        proposals = self.proposals(state)
        return _decision(_best(proposals), state, self.fill_target, self.mask_mode, {"scores": proposals})


class CommitteePolicy:
    def __init__(self, members: Sequence[Proposer], fill_target: int,
                 mask_mode: MaskMode = MaskMode.PHYSICAL) -> None:
        if not members:
            raise ValueError("A committee needs at least one member")
        self.members = list(members)
        self.fill_target = fill_target
        self.mask_mode = MaskMode(mask_mode)
        self.name = "committee(" + " + ".join(m.name for m in self.members) + ")"

    def decide(self, state: PlantState) -> Decision:
        merged: dict[int, float] = {}
        winners: dict[int, str] = {}
        for member in self.members:
            for action, score in member.proposals(state).items():
                if action not in merged or score < merged[action]:
                    merged[action], winners[action] = score, member.name
        action = _best(merged)
        return _decision(action, state, self.fill_target, self.mask_mode,
                         {"proposed_by": winners[action], "scores": merged})
