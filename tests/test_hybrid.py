"""Hybrid decision makers: rollouts, committees and the value-guided planner."""
import pickle

import numpy as np
import torch

from haam_rl.adapters.value_search import LeafValue, ValueNetwork, encode_leaves
from haam_rl.application.hybrid import CommitteePolicy, PlannerProposer, RolloutPolicy
from haam_rl.application.use_cases import HeuristicPolicy
from haam_rl.domain.dynamics import apply_action, executable_actions, is_terminal
from haam_rl.domain.model import PlantState
from haam_rl.domain.planning import best_plan, fragmentation_estimate
from haam_rl.domain.scenarios import generate_arrivals, initial_state

FULL = 25


def test_rollout_scores_every_executable_first_action():
    state = initial_state(generate_arrivals(11, 60))
    for _ in range(30):  # into the steady phase
        state = apply_action(state, HeuristicPolicy(FULL).decide(state).action, FULL).state
    rollout = RolloutPolicy(HeuristicPolicy(FULL), FULL, horizon=5)
    proposals = rollout.proposals(state)
    assert set(proposals) == set(executable_actions(state, FULL))
    assert rollout.decide(state).action == min(proposals, key=lambda a: (proposals[a], a))


def test_committee_keeps_the_best_scored_proposal_and_plays_valid_games():
    committee = CommitteePolicy([PlannerProposer(FULL, horizon=4, width=4),
                                 RolloutPolicy(HeuristicPolicy(FULL), FULL, horizon=4)], FULL)
    state = initial_state(generate_arrivals(5, 40))
    while not is_terminal(state):
        decision = committee.decide(state)
        assert decision.details["scores"][decision.action] == min(decision.details["scores"].values())
        state = apply_action(state, decision.action, FULL).state


def test_explicit_default_estimate_matches_the_planner():
    state = initial_state(generate_arrivals(2, 50))
    for _ in range(27):
        state = apply_action(state, HeuristicPolicy(FULL).decide(state).action, FULL).state
    assert best_plan(state, FULL, 6, 8) == best_plan(state, FULL, 6, 8, estimate=fragmentation_estimate(0.2))


def test_leaf_encoding_and_value_are_invariant_to_buffer_order():
    state = PlantState(0, 3, ((1, 2), (3,), (), (4, 4, 5), (2, 2, 2)))
    shuffled = PlantState(0, 3, (state.buffers[3], state.buffers[0], state.buffers[4], state.buffers[1], state.buffers[2]))
    torch.manual_seed(0)
    network = ValueNetwork(16)
    with torch.no_grad():
        a, b = network(torch.as_tensor(encode_leaves([state, shuffled])))
    assert torch.allclose(a, b, atol=1e-6)
    value = LeafValue(network, scale=2.0)
    first = value([state, shuffled])
    assert first == value([state, shuffled]) and len(value.cache) == 2  # cached per exact state
    assert np.isclose(first[0], first[1], atol=1e-5)


def test_sb3_policy_pickles_by_path(tmp_path):
    from haam_rl.adapters.sb3_policy import SB3Policy
    from haam_rl.adapters.training import train
    from haam_rl.config import ExperimentConfig

    config = ExperimentConfig().with_overrides(training={"total_timesteps": 256, "n_envs": 1, "n_steps": 128,
                                                         "batch_size": 64, "eval_freq": 10**9})
    train(config, seed=0, run_dir=tmp_path, verbose=0)
    policy = SB3Policy.load(tmp_path / "final_model.zip")
    clone = pickle.loads(pickle.dumps(policy))
    state = initial_state(generate_arrivals(1, 30))
    assert clone.decide(state).action == policy.decide(state).action


def test_finished_episodes_cost_nothing_more():
    value = LeafValue(ValueNetwork(8), scale=1.0)
    finished = PlantState(0, 4, ((), (), (), (), ()))
    assert value([finished]) == [0.0] and not value.cache


def test_leaf_value_cache_never_loses_requested_states():
    value = LeafValue(ValueNetwork(8), cache_size=3)
    states = [PlantState(0, c, ((c,), (), (), (), ())) for c in range(1, 7)]
    value(states[:3])                  # fills the cache
    out = value([states[0], *states[3:]])  # one hit + three new states, cache limit 3
    assert len(out) == 4 and len(value.cache) == 3
