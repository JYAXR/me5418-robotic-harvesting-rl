"""Check cache isolation, invalidation, rejection handling and seeded RNG parity."""
from copy import deepcopy

import numpy as np
import pytest

from me5418_harvesting.envs.harvesting_env import HarvestingEnv
from me5418_harvesting.envs.scene_cache import CertifiedSceneCache
from me5418_harvesting.evaluation import scene_feasibility


@pytest.fixture
def certify(monkeypatch):
    calls = []

    def check(env):
        calls.append(env.scene.end_effector.copy())
        return {"success": True, "steps": 7, "reason": "test_certificate"}

    monkeypatch.setattr(scene_feasibility, "verify_reference_route", check)
    return calls


def test_hit_restores_rng_and_isolates_episode_mutations(certify):
    cfg = {"verify_reference_route": True, "execution_noise_fraction": 0.1}
    cache = CertifiedSceneCache()
    a = HarvestingEnv(cfg, scene_cache=cache, profile_reset=True)
    b = HarvestingEnv(cfg)
    obs, info = a.reset(seed=12)
    expected_obs, expected_info = b.reset(seed=12)
    np.testing.assert_array_equal(obs, expected_obs)
    assert info == expected_info
    state = deepcopy(a.np_random.bit_generator.state)
    a.step(2)
    a.scene.branches[0].current_nodes[:] += 3
    info["reference_route"]["steps"] = 999
    restored, restored_info = a.reset(seed=12)
    np.testing.assert_array_equal(restored, expected_obs)
    assert restored_info == expected_info and a.np_random.bit_generator.state == state
    assert a.np_random_seed == b.np_random_seed == 12
    assert a.step_count == a.total_travel == a.contact_steps == a.peak_deformation == 0
    assert len(a.trajectory) == 1 and len(certify) == 2
    assert a.last_reset_profile["cache_status"] == "hit"
    assert a.last_reset_profile["certification_actions"] == 0
    for action in (2, 0, 4, 1):
        left, right = a.step(action), b.step(action)
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:4] == right[1:4]
        for key in left[4]:
            np.testing.assert_array_equal(left[4][key], right[4][key])


@pytest.mark.parametrize("name,value", [("max_steps", 120), ("noise_fraction", 0.02),
                                      ("contact_radius", 0.04), ("relaxation_rate", 0.2)])
def test_config_changes_miss(certify, name, value):
    env = HarvestingEnv({"verify_reference_route": True}, scene_cache=CertifiedSceneCache())
    env.reset(seed=5)
    setattr(env, name, value)
    env.reset(seed=5)
    assert len(certify) == 2 and env.last_reset_profile["cache_status"] == "miss"


def test_nested_config_changes_miss(certify):
    env = HarvestingEnv({"verify_reference_route": True}, scene_cache=CertifiedSceneCache())
    env.reset(seed=5)
    env.reward_weights["success"] += 1
    env.reset(seed=5)
    assert len(certify) == 2


def test_unseeded_and_options_bypass(certify):
    cache = CertifiedSceneCache()
    env = HarvestingEnv({"verify_reference_route": True}, scene_cache=cache)
    env.reset(seed=5)
    env.reset()
    assert env.last_reset_profile["cache_status"] == "bypass"
    env.reset(seed=5, options={})
    assert env.last_reset_profile["cache_status"] == "bypass" and len(cache) == 1


def test_rejected_candidates_and_failures(monkeypatch):
    count = 0

    def check(env):
        nonlocal count
        count += 1
        return {"success": count % 2 == 0, "steps": 7, "reason": "test_candidate"}

    monkeypatch.setattr(scene_feasibility, "verify_reference_route", check)
    cache = CertifiedSceneCache()
    env = HarvestingEnv({"verify_reference_route": True}, scene_cache=cache)
    obs, info = env.reset(seed=6)
    assert len(info["reference_route_checks"]) == 2 and count == 2
    again, same_info = env.reset(seed=6)
    np.testing.assert_array_equal(obs, again)
    assert same_info == info and count == 2
    cache.clear()
    monkeypatch.setattr(scene_feasibility, "verify_reference_route",
                        lambda env: {"success": False, "steps": 0, "reason": "failed"})
    with pytest.raises(ValueError, match="Could not certify"):
        env.reset(seed=6)
    assert len(cache) == 0


def test_lru_and_shared_cache(certify):
    cache = CertifiedSceneCache(1)
    a = HarvestingEnv({"verify_reference_route": True}, scene_cache=cache)
    b = HarvestingEnv({"verify_reference_route": True}, scene_cache=cache)
    a.reset(seed=1)
    b.reset(seed=1)
    assert b.last_reset_profile["cache_status"] == "hit"
    a.reset(seed=2)
    b.reset(seed=1)
    assert len(certify) == 3 and len(cache) == 1
    assert a.scene is not b.scene


def test_invalid_capacity_and_uncertified_bypass():
    for capacity in (True, 0, -1):
        with pytest.raises(ValueError):
            CertifiedSceneCache(capacity)
    cache = CertifiedSceneCache()
    env = HarvestingEnv(scene_cache=cache)
    env.reset(seed=1)
    assert len(cache) == 0 and env.last_reset_profile["cache_status"] == "bypass"


def test_subclass_bypass(certify):
    class CustomEnv(HarvestingEnv):
        pass

    cache = CertifiedSceneCache()
    env = CustomEnv({"verify_reference_route": True}, scene_cache=cache)
    env.reset(seed=5)
    assert len(cache) == 0 and env.last_reset_profile["cache_status"] == "bypass"


def test_timing_is_separate_and_preserves_evaluator_result(certify):
    from types import SimpleNamespace

    from me5418_harvesting.agents.training import evaluate

    env = HarvestingEnv({"verify_reference_route": True, "max_steps": 2},
                        scene_cache=CertifiedSceneCache(), profile_reset=True)
    agent = SimpleNamespace(act=lambda obs: 2)
    original = evaluate(agent, env, 1, 12)
    timing = []
    cached = evaluate(agent, env, 1, 12, timings=timing)
    assert original == cached and len(timing) == 1
    assert timing[0]["reset"]["cache_status"] == "hit"
    assert timing[0]["reset"]["certification_actions"] == 0
    assert all(timing[0][key] >= 0 for key in
               ("episode_seconds", "reset_seconds", "policy_seconds", "physics_seconds"))
