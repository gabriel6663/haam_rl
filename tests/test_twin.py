import numpy as np
import pytest

from haam_rl.adapters.gym_env import PaintShopEnv
from haam_rl.domain.heuristics import haam_action_mask
from haam_rl.domain.model import PlantState, encode_action


def test_state_vector_is_compatible_with_the_paper_shape():
    state = PlantState(3, 2, ((1, 3), (), (2,), (), ()), (4, 5))
    assert state.as_vector().shape == (27,)
    rebuilt = PlantState.from_vector(state.as_vector())
    assert rebuilt.incoming_color == 3
    assert rebuilt.last_painted_color == 2
    assert rebuilt.buffers[0] == (1, 3)


def test_haam_mask_rejects_empty_output_buffers():
    state = PlantState(3, 2, ((2,), (), (), (), ()), ())
    mask = haam_action_mask(state)
    assert mask.any()
    assert mask[encode_action(0, 0)]
    assert not any(mask[encode_action(i, 1)] for i in range(5))


def test_every_masked_action_is_executable():
    env = PaintShopEnv(episode_vehicles=20)
    env.reset(seed=7)
    for _ in range(10):
        mask = env.action_masks()
        assert mask.dtype == np.bool_
        action = int(np.flatnonzero(mask)[0])
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break


def test_invalid_action_fails_loudly():
    env = PaintShopEnv(episode_vehicles=20)
    env.reset(seed=7)
    invalid = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(ValueError, match="Invalid HAAM action"):
        env.step(invalid)
