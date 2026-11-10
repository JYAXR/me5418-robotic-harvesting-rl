"""Ensure reference checks use real transitions and leave callers unchanged."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from me5418_harvesting import HarvestingEnv
from me5418_harvesting.envs.branch_physics import Branch
from me5418_harvesting.envs.scene_generator import Scene
from me5418_harvesting.evaluation.scene_feasibility import verify_reference_route

ROOT = Path(__file__).resolve().parents[1]


def easy_config():
    return yaml.safe_load((ROOT / "configs/environment_easy.yaml").read_text(encoding="utf-8"))


def test_reference_success_preserves_scene_counters_and_rng():
    env = HarvestingEnv({**easy_config(), "execution_noise_fraction": 0.01})
    try:
        env.reset(seed=3)
        original_scene = env.scene
        observation = env._observation().copy()
        trajectory = deepcopy(env.trajectory)
        rng_state = deepcopy(env.np_random.bit_generator.state)
        result = verify_reference_route(env)
        assert result["success"] and not result["damage"]
        assert not result["planning_failure"]
        assert result["steps"] > 0 and result["travel_distance"] > 0
        assert result["return"] > 0
        assert env.scene is original_scene
        np.testing.assert_array_equal(env._observation(), observation)
        np.testing.assert_array_equal(env.trajectory, trajectory)
        assert env.step_count == env.total_travel == env.contact_steps == env.peak_deformation == 0
        assert json.dumps(env.np_random.bit_generator.state) == json.dumps(rng_state)
        assert result == verify_reference_route(env)
        json.dumps(result, allow_nan=False)
    finally:
        env.close()


def test_budget_failure_is_not_reported_as_planning_failure():
    env = HarvestingEnv({**easy_config(), "max_steps": 1})
    try:
        env.reset(seed=0)
        result = verify_reference_route(env)
        assert not result["success"] and not result["planning_failure"]
        assert result["budget_exhausted"] and result["steps"] == 1
        assert result["reason"] == "budget_exhausted"
        assert env.step_count == 0
    finally:
        env.close()


def test_blocked_reference_route_records_planner_failure_without_claiming_infeasibility():
    env = HarvestingEnv()
    try:
        env.reset(seed=0)
        nodes = np.array([[0.5, 0.0], [0.5, 1.0]], dtype=np.float32)
        env.scene = Scene(np.array([0.1, 0.5]), np.array([0.9, 0.5]),
                          [Branch(nodes.copy(), nodes.copy(), 1e6, 0.1)])
        result = verify_reference_route(env)
        assert result["planning_failure"] and not result["success"]
        assert result["reason"] == "planner_found_no_route"
        assert result["steps"] == 0 and not result["damage"]
    finally:
        env.close()


def test_reference_requires_a_reset_scene():
    env = HarvestingEnv()
    with pytest.raises(RuntimeError, match="Reset"):
        verify_reference_route(env)


def test_batch_keeps_every_seed_when_generation_fails():
    script = ROOT / "scripts/check_scene_feasibility.py"
    spec = importlib.util.spec_from_file_location("check_scene_feasibility", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = easy_config()
    config["workspace_high"] = [0.5, 1.0]
    config["scene"]["max_attempts"] = 1
    config["verify_reference_route"] = True
    original_config = deepcopy(config)
    passed_configs = []
    original_environment = module.HarvestingEnv

    def record_environment(settings):
        passed_configs.append(settings)
        return original_environment(settings)

    module.HarvestingEnv = record_environment
    result = module.check_scenes(config, seed=10, count=3)
    assert config == original_config
    assert passed_configs[0]["verify_reference_route"] is False
    assert result["route_filter_disabled"] is True
    assert result["requested_scenes"] == 3
    assert [row["seed"] for row in result["records"]] == [10, 11, 12]
    assert all(not row["generation_success"] for row in result["records"])
    assert result["successful_reference_routes"] == 0
