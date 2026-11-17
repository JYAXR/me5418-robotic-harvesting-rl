"""Geometry, immutable priors, exact retention, and bounded query coverage."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent, ObservationEncoder
from me5418_harvesting.agents.recovery_policy import GeometryEncoder, RecoveryAgent, ResidualPolicy
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recovery_study", ROOT / "docs/experiments/your_method/009_geometry_retention/run_study.py"
)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)
torch.set_num_threads(1)


def test_geometry_has_directional_clearance_and_ignores_branch_slot_order():
    env = HarvestingEnv({})
    try:
        encoder = GeometryEncoder(env)
        base = np.zeros(env.observation_size, dtype=np.float32)
        base[:4] = [0.5, 0.5, 0, 0.3]
        base[-1] = 1
        width = 4 * env.nodes_per_branch + 4
        nodes = np.column_stack((np.linspace(0.3, 0.7, env.nodes_per_branch),
                                 np.full(env.nodes_per_branch, 0.56))).astype(np.float32)
        base[4:4 + 4 * env.nodes_per_branch] = np.tile(nodes.reshape(-1), 2)
        base[4 + width - 4:4 + width] = [1, 0, 0.1, 1]
        before = base.copy()
        result = encoder.extend(base)
        features = result[-40:].reshape(8, 5)
        assert features[4, 0] < features[5, 0]
        assert features[4, 0] < 0
        assert features[4, 3] > features[5, 3]
        np.testing.assert_array_equal(base, before)
        moved = base.copy()
        moved[4 + width:4 + 2 * width] = base[4:4 + width]
        moved[4:4 + width] = 0
        np.testing.assert_array_equal(encoder.extend(moved)[-40:], result[-40:])
        empty = before.copy()
        empty[4:4 + width] = 0
        assert np.isfinite(encoder.extend(empty)).all()
        with pytest.raises(ValueError):
            encoder.extend(np.full(env.observation_size, np.nan))
    finally:
        env.close()


def test_geometry_is_consistent_under_workspace_translation():
    env = HarvestingEnv({})
    moved = HarvestingEnv({"workspace_low": [2.0, 3.0], "workspace_high": [3.0, 4.0]})
    try:
        base = np.zeros(env.observation_size, dtype=np.float32)
        base[:4] = [.5, .5, .2, .2]
        np.testing.assert_allclose(GeometryEncoder(env).extend(base),
                                   GeometryEncoder(moved).extend(base), atol=1e-6)
    finally:
        env.close()
        moved.close()


def test_conversion_keeps_greedy_choices_and_freezes_inherited_weights(tmp_path):
    env = HarvestingEnv({"max_steps": 2})
    config = DQNConfig(hidden_sizes=(16, 16))
    original = DQNAgent(env, config, seed=0)
    path = tmp_path / "original.pt"
    original.save(path, {"max_steps": 2}, {})
    converted, converted_env, _ = RecoveryAgent.from_source(path)
    try:
        raw, _ = env.reset(seed=0)
        encoded = converted.encoder(raw)
        assert len(encoded) == env.observation_size + 40
        np.testing.assert_array_equal(encoded[:-40], ObservationEncoder(env)(raw))
        assert converted.act(raw) == original.act(raw)
        net = converted.learner.online
        net(torch.from_numpy(encoded)[None]).sum().backward()
        assert all(p.grad is None for p in net.inherited.parameters())
        assert any(p.grad is not None for p in net.correction.parameters())
        saved = tmp_path / "converted.pt"
        converted.save(saved, {"max_steps": 2}, {"new_td_updates": 0})
        loaded, loaded_env, meta = RecoveryAgent.load(saved)
        try:
            assert loaded.act(raw) == converted.act(raw)
            assert meta["new_td_updates"] == 0
            for key, value in net.state_dict().items():
                assert torch.equal(value, loaded.learner.online.state_dict()[key])
        finally:
            loaded_env.close()
        with pytest.raises(ValueError):
            RecoveryAgent.load(path)
    finally:
        env.close()
        converted_env.close()


def test_projection_rejects_an_unsafe_block_and_restores_adam_state():
    inherited = nn.Linear(1, 8)
    with torch.no_grad():
        inherited.weight.zero_()
        inherited.bias.fill_(-1)
        inherited.bias[0] = .001
        inherited.bias[1] = 0
    net = ResidualPolicy(inherited, 1)
    opt = torch.optim.Adam(net.correction.parameters(), lr=.1)
    x, y = torch.zeros(1, 41), torch.tensor([0])
    weights, state = deepcopy(net.correction.state_dict()), deepcopy(opt.state_dict())
    (-net(x)[0, 1]).backward()
    opt.step()
    assert not study.retain(net, x, y)
    assert study.project_block(net, opt, weights, state, x, y) == 0
    assert study.retain(net, x, y)
    assert opt.state_dict() == state
    assert all(torch.equal(value, net.correction.state_dict()[key]) for key, value in weights.items())


def test_query_selection_reaches_late_trajectory_without_duplicate_geometry():
    candidates = [{"step": step, "blocked_run": step % 9, "clearance": .1,
                   "fingerprint": str(step)} for step in range(1, 201)]
    selected = study.choose_queries(candidates)
    assert len(selected) == 8
    for low, high in ((1, 25), (26, 50), (51, 100), (101, 200)):
        assert {kind for row, kind in selected if low <= row["step"] <= high} == {"risk", "coverage"}
    for row in candidates:
        row["fingerprint"] = "same geometry"
    assert len(study.choose_queries(candidates)) == 1


def test_action_probes_are_frozen_and_labels_match_physics():
    env = HarvestingEnv({"max_steps": 4})
    try:
        env.reset(seed=0)
        before = study.c.scene_state(env)
        raw, bad, checks = study.action_probes(env, -1)
        assert raw == bad
        assert study.c.scene_state(env) == before
        for flag, check in zip(bad, checks):
            info = check["info"]
            assert flag == (info["damage"] or (np.linalg.norm(info["executed_motion"]) <= 1e-6 and not info["success"]))
    finally:
        env.close()
