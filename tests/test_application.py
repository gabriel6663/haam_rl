"""Ensemble voting, evaluation statistics, configs, observation encoding and the Gym wrapper."""
from pathlib import Path

import numpy as np
import pytest

from haam_rl.adapters.gym_env import PaintShopEnv
from haam_rl.adapters.observation import ObservationEncoder
from haam_rl.adapters.runner import evaluation_twin
from haam_rl.application.evaluation import evaluate, paired_comparison
from haam_rl.application.use_cases import HeuristicPolicy, ensemble_vote
from haam_rl.config import Encoding, ExperimentConfig, ObservationConfig
from haam_rl.domain.model import PlantState


def probs(*actions_and_masses):
    out = np.zeros(25)
    for action, mass in actions_and_masses:
        out[action] = mass
    return out


def test_ensemble_uses_hard_voting_when_members_agree():
    members = np.stack([probs((3, 0.6), (4, 0.4))] * 3 + [probs((4, 0.9), (3, 0.1))] * 2)
    vote = ensemble_vote(members)  # 5 members, threshold 1.5, action 3 has 3 votes
    assert (vote.mode, vote.action) == ("hard", 3)


def test_ensemble_falls_back_to_soft_voting_when_votes_scatter():
    members = np.stack([probs((a, 0.5), (9, 0.45), (b, 0.05)) for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7)]])
    vote = ensemble_vote(members, hard_threshold=1)  # every action has 1 vote, not > 1
    assert (vote.mode, vote.action) == ("soft", 9)


def test_single_member_ensemble_is_the_member():
    member = probs((7, 0.7), (2, 0.3))
    assert ensemble_vote(member[None, :]).action == 7


def test_paired_comparison():
    stats = paired_comparison([28, 29, 30, 31], [32, 33, 34, 35])
    assert stats.mean_difference == -4 and stats.wins == 4 and stats.reduction_pct == pytest.approx(4 / 33.5 * 100)
    assert stats.ci95 == (-4.0, -4.0)


def test_config_round_trip(tmp_path: Path):
    config = ExperimentConfig().with_overrides(observation={"lookahead": 7}, mask_mode="physical")
    config.save(tmp_path / "c.json")
    assert ExperimentConfig.load(tmp_path / "c.json") == config


@pytest.mark.parametrize("encoding,lookahead,size", [(Encoding.RAW, 0, 27), (Encoding.RAW, 3, 30), (Encoding.ONEHOT, 10, 37 * 7 + 4)])
def test_observation_sizes(encoding, lookahead, size):
    encoder = ObservationEncoder(ObservationConfig(encoding=encoding, lookahead=lookahead, include_phase=encoding is Encoding.ONEHOT), 25)
    state = PlantState(2, 1, ((1, 2), (), (), (), ()), (3, 4))
    vector = encoder.encode(state)
    assert vector.shape == (size,) == (encoder.size,)
    assert len(encoder.raw_labels()) == size
    if encoding is Encoding.ONEHOT:
        assert vector[: 37 * 7].reshape(37, 7).sum(axis=1).tolist() == [1.0] * 37


def test_gym_env_steps_with_its_mask():
    env = PaintShopEnv(ExperimentConfig(), scenario_seeds=[1, 2])
    observation, info = env.reset()
    assert info["scenario_seed"] == 1 and env.observation_space.contains(observation)
    terminated = False
    while not terminated:
        _, _, terminated, _, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert env.reset()[1]["scenario_seed"] == 2


def test_heuristic_policy_evaluates_on_a_physical_twin():
    config = ExperimentConfig()
    results = evaluate(lambda: evaluation_twin(config), HeuristicPolicy(25), [10_000])
    assert results[0].color_changes > 0 and len(results[0].painted) == 100


def test_runner_gives_the_same_results_with_processes_and_in_lockstep():
    from haam_rl.adapters.runner import run_scenarios
    from haam_rl.application.use_cases import LookaheadPlannerPolicy

    config = ExperimentConfig()
    planner = LookaheadPlannerPolicy(25, horizon=3, width=4)
    seeds = [10_000, 10_001, 10_002, 10_003]
    sequential = [r.color_changes for r in run_scenarios(config, planner, seeds, workers=1)]
    parallel = [r.color_changes for r in run_scenarios(config, planner, seeds, workers=2)]
    assert sequential == parallel


def test_reduction_interval_brackets_the_estimate():
    rng = np.random.default_rng(0)
    baseline = rng.integers(28, 38, size=60).astype(float)
    candidate = baseline - rng.integers(0, 4, size=60)
    stats = paired_comparison(candidate, baseline, resamples=5_000)
    assert stats.reduction_ci95[0] <= stats.reduction_pct <= stats.reduction_ci95[1]
    assert stats.ci95[0] <= stats.mean_difference <= stats.ci95[1] and stats.p_value < 0.01


def test_twin_mask_is_cached_and_read_only():
    from haam_rl.adapters.python_twin import PythonPaintShopTwin

    twin = PythonPaintShopTwin(ExperimentConfig())
    twin.reset(1)
    mask = twin.action_mask()
    assert twin.action_mask() is mask and not mask.flags.writeable
    twin.step(int(np.flatnonzero(mask)[0]))
    assert twin.action_mask() is not mask


def test_discounted_returns_restart_at_each_episode():
    from haam_rl.adapters.distillation import discounted_returns

    rewards = np.array([1.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    episode = np.array([7, 7, 7, 9, 9])
    np.testing.assert_allclose(discounted_returns(rewards, episode, 0.5), [1.75, 1.5, 1.0, 3.0, 2.0])


def test_command_line_overrides():
    from haam_rl.config import parse_overrides

    overrides = parse_overrides(["training.policy=buffer_pair", "training.net_arch=[128,128]", "name=x", "plant.fill_target=24"])
    config = ExperimentConfig().with_overrides(**overrides)
    assert config.training.policy == "buffer_pair" and config.training.net_arch == (128, 128)
    assert config.name == "x" and config.plant.fill_target == 24
