"""Plant rules, heuristics, reward and planning. No ML dependencies."""
import numpy as np
import pytest

from haam_rl.domain.dynamics import apply_action, is_terminal, phase_of, physical_action_mask
from haam_rl.domain.heuristics import MaskMode, action_mask, heuristic_action
from haam_rl.domain.model import Phase, PlantState, decode_action, encode_action
from haam_rl.domain.planning import lookahead_action
from haam_rl.domain.reward import RewardConfig, potential, reward
from haam_rl.domain.scenarios import TEST_SEEDS, generate_arrivals, initial_state

FULL = 25


def play(arrivals, choose, fill_target=FULL):
    state, phases, changes, painted = initial_state(arrivals), [], 0, []
    while not is_terminal(state):
        transition = apply_action(state, choose(state), fill_target)
        phases.append(transition.phase)
        changes += transition.color_changed
        if transition.painted_color:
            painted.append(transition.painted_color)
        state = transition.state
    return phases, changes, painted


def test_state_vector_round_trip_compacts_gaps():
    state = PlantState(3, 2, ((1, 3), (), (2,), (), ()), (4, 5))
    assert state.as_vector().shape == (27,)
    rebuilt = PlantState.from_vector(state.as_vector(), upcoming=[4, 5])
    assert rebuilt == PlantState(3, 2, ((1, 3), (), (2,), (), ()), (4, 5))
    gapped = [3, 2, 0, 1, 0, 0, 0] + [0] * 20
    assert PlantState.from_vector(gapped).buffers[0] == (1,)


def test_episode_fills_then_runs_steady_then_drains():
    arrivals = generate_arrivals(7, 100)
    phases, _, painted = play(arrivals, lambda s: heuristic_action(s, FULL))
    assert phases[:25] == [Phase.FILL] * 25
    assert phases[25:100] == [Phase.STEADY] * 75
    assert phases[100:] == [Phase.DRAIN] * 25
    assert sorted(painted) == sorted(arrivals)  # every vehicle is painted exactly once


def test_full_buffers_force_the_incoming_vehicle_into_the_freed_buffer():
    state = PlantState(2, 1, ((1,) * 5, (2,) * 5, (3,) * 5, (4,) * 5, (5,) * 5), ())
    assert phase_of(state, FULL) is Phase.STEADY
    valid = {decode_action(a) for a in np.flatnonzero(physical_action_mask(state, FULL))}
    assert valid == {(o, o) for o in range(5)}


def test_fill_and_drain_actions_live_on_the_diagonal():
    filling = PlantState(2, 0, ((), (), (), (), ()), ())
    assert {decode_action(a) for a in np.flatnonzero(physical_action_mask(filling, FULL))} == {(i, i) for i in range(5)}
    draining = PlantState(0, 1, ((1,), (), (2,), (), ()), ())
    assert {decode_action(a) for a in np.flatnonzero(physical_action_mask(draining, FULL))} == {(0, 0), (2, 2)}


def test_invalid_action_fails_loudly():
    state = PlantState(2, 0, ((), (), (), (), ()), ())
    with pytest.raises(ValueError):
        apply_action(state, encode_action(0, 1), FULL)


def test_heuristic_rules():
    # Output CM: the front of buffer 2 matches the last painted color.
    state = PlantState(4, 3, ((1, 1), (2,), (3, 1), (5, 6, 1), ()), ())
    assert decode_action(heuristic_action(state, fill_target=6))[1] == 2
    # Input CM in FILL: buffer 1's tail already holds color 2.
    filling = PlantState(2, 0, ((1,), (2,), (), (), ()), ())
    assert heuristic_action(filling, FULL) == encode_action(1, 1)
    # Input LP in FILL: no tail matches, so an empty (least diverse) buffer is preferred.
    filling = PlantState(4, 0, ((1,), (2,), (), (), ()), ())
    assert heuristic_action(filling, FULL) == encode_action(2, 2)
    # Output UCM: no front matches, so the buffer with the most distinct colors is painted from.
    draining = PlantState(0, 6, ((1, 1), (2, 3, 4), (5,), (), ()), ())
    assert heuristic_action(draining, FULL) == encode_action(1, 1)


@pytest.mark.parametrize("fill_target", [25, 20, 10])
def test_masks_are_nested_and_never_empty_before_the_end(fill_target):
    rng = np.random.default_rng(0)
    state = initial_state(generate_arrivals(3, 60))
    while not is_terminal(state):
        physical = action_mask(state, fill_target, MaskMode.PHYSICAL)
        output_only = action_mask(state, fill_target, MaskMode.HAAM_OUTPUT)
        haam = action_mask(state, fill_target, MaskMode.HAAM)
        assert haam.any() and np.all(haam <= output_only) and np.all(output_only <= physical)
        state = apply_action(state, int(rng.choice(np.flatnonzero(physical))), fill_target).state


def test_heuristic_reproduces_the_paper_baseline_level():
    """The paper reports 34.38 changes for its heuristic; full buffers give a comparable 32.3 here."""
    changes = [play(generate_arrivals(s, 100), lambda st: heuristic_action(st, FULL))[1] for s in TEST_SEEDS]
    assert np.mean(changes) == pytest.approx(32.33, abs=0.01)


def test_shaping_potential_is_zero_at_the_end_and_shaping_telescopes():
    config, gamma = RewardConfig(), 1.0
    state = initial_state(generate_arrivals(5, 30))
    shaped_total = unshaped_total = 0.0
    while not is_terminal(state):
        transition = apply_action(state, heuristic_action(state, FULL), FULL)
        shaped_total += reward(state, transition, config, gamma)
        unshaped_total += reward(state, transition, RewardConfig(use_shaping=False), gamma)
        state = transition.state
    assert potential(state, config) == 0.0
    # With gamma = 1 the shaping terms cancel except for -phi(s0), which is 0 for empty buffers.
    assert shaped_total == pytest.approx(unshaped_total)


def test_planner_returns_executable_actions_in_every_phase():
    arrivals = generate_arrivals(TEST_SEEDS[0], 40)
    phases, _, painted = play(arrivals, lambda s: lookahead_action(s, 10, horizon=3, width=4), fill_target=10)
    assert set(phases) == {Phase.FILL, Phase.STEADY, Phase.DRAIN} and len(painted) == 40


def test_perfect_information_oracle_beats_the_heuristic():
    arrivals = generate_arrivals(TEST_SEEDS[0], 100)
    oracle = play(arrivals, lambda s: lookahead_action(s, FULL, horizon=None, width=8))[1]
    assert oracle < play(arrivals, lambda s: heuristic_action(s, FULL))[1]


def test_lookahead_planner_beats_the_heuristic_with_the_same_information_horizon():
    seeds = list(TEST_SEEDS)[:3]
    planned = [play(generate_arrivals(s, 100), lambda st: lookahead_action(st, FULL, horizon=5, width=8))[1] for s in seeds]
    greedy = [play(generate_arrivals(s, 100), lambda st: heuristic_action(st, FULL))[1] for s in seeds]
    assert sum(planned) < sum(greedy)


def test_planner_never_leaves_its_mask():
    from haam_rl.domain.heuristics import allowed_actions

    state = initial_state(generate_arrivals(TEST_SEEDS[1], 60))
    while not is_terminal(state):
        action = lookahead_action(state, FULL, horizon=4, width=6, mask_mode=MaskMode.HAAM)
        assert action in allowed_actions(state, FULL, MaskMode.HAAM)
        state = apply_action(state, action, FULL).state


def test_golden_results_on_the_first_test_scenarios():
    """Exact per-scenario values; any change to the plant rules or policies must update these knowingly."""
    seeds = list(TEST_SEEDS)[:5]
    heuristic = [play(generate_arrivals(s, 100), lambda st: heuristic_action(st, FULL))[1] for s in seeds]
    planner = [play(generate_arrivals(s, 100), lambda st: lookahead_action(st, FULL, 10, 32))[1] for s in seeds]
    assert heuristic == [29, 37, 31, 35, 29]
    assert planner == [21, 24, 28, 23, 23]


def test_trusted_states_equal_validated_ones():
    state = initial_state(generate_arrivals(3, 30))
    transition = apply_action(state, heuristic_action(state, FULL), FULL)
    rebuilt = PlantState(*(getattr(transition.state, f) for f in
                           ("incoming_color", "last_painted_color", "buffers", "remaining_arrivals", "consecutive_same_color")))
    assert rebuilt == transition.state and rebuilt.occupancy == transition.state.occupancy
