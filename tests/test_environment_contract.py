import numpy as np

from me5418_harvesting import HarvestingEnv


def test_reset_and_step_follow_gymnasium_contract() -> None:
    env = HarvestingEnv()
    observation, info = env.reset(seed=7)
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    assert info["seed"] == 7

    next_observation, reward, terminated, truncated, step_info = env.step(0)
    assert next_observation.shape == observation.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "max_normalized_deformation" in step_info


def test_deterministic_mode_is_seed_reproducible() -> None:
    first = HarvestingEnv({"execution_noise_fraction": 0.0})
    second = HarvestingEnv({"execution_noise_fraction": 0.0})
    first.reset(seed=11)
    second.reset(seed=11)
    for action in [2, 2, 4, 0, 6]:
        first_transition = first.step(action)
        second_transition = second.step(action)
        np.testing.assert_allclose(first_transition[0], second_transition[0])
        assert first_transition[1:4] == second_transition[1:4]
        np.testing.assert_allclose(
            first_transition[4]["executed_motion"],
            second_transition[4]["executed_motion"],
        )
