"""Check bounded trajectory reconstruction with synthetic policies and dynamics."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gradient018_trajectory_test", ROOT / "docs/experiments/018_gradient_diagnostics/trajectory_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def raw_observation(step):
    return np.asarray([step, 0.0, 0.25, 0.5, 1 - step / 200], dtype=np.float32)


class FixtureEncoder:
    VERSION = 1

    def __init__(self, env):
        self.size = env.observation_size

    def __call__(self, observation):
        assert observation.shape == (self.size,)
        return observation.copy()


class FixturePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.rand(1)
        self.register_buffer("marker", torch.tensor(0.0))
        self.calls = 0

    def forward(self, observation):
        self.calls += 1
        assert observation.shape == (1, 5)
        values = torch.zeros((1, 8))
        action = 6 if int(observation[0, 0]) == 39 and int(self.marker) == 8500 else 1
        values[0, action] = 2
        return values


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    directory = tmp_path / "treatment/td"
    directory.mkdir(parents=True)
    rows, checkpoint_hashes = [], {}
    for update in probe.CHECKPOINTS:
        path = directory / f"update_{update}.pt"
        torch.save({"format_version": 1, "encoder_version": 1,
                    "environment_config": {"execution_noise_fraction": 0.0},
                    "dqn_config": asdict(probe.DQNConfig(hidden_sizes=(2, 2))),
                    "online": {"marker": torch.tensor(float(update))}}, path)
        checkpoint_hashes[path.name] = probe.file_hash(path)
        steps = 200 if update == 8500 else 41
        actions = [1] * steps
        if update == 8500:
            actions[39] = 6
        rows.append({"update": update, "scenes": [{"seed": 10002, "rollout": {
            "seed": 10002, "steps": steps, "actions": actions,
            "robot_path": [[float(i), 0.0] for i in range(steps + 1)],
            "success": update != 8500}}]})
    (directory / "summary.json").write_text(json.dumps({"checkpoint_sha256": checkpoint_hashes}))
    (directory / "observations.json").write_text(json.dumps(rows))
    environments, models = [], []

    class FixtureEnvironment:
        observation_size = 5

        def __init__(self, config):
            self.config = config
            self.step_count = 0
            self.actions = []
            self.resets = []
            self.closed = False
            self.terminate_at = None
            environments.append(self)

        def reset(self, seed):
            self.resets.append(seed)
            return raw_observation(0), {}

        def step(self, action):
            self.actions.append(action)
            self.step_count += 1
            return raw_observation(self.step_count), 0.0, self.step_count == self.terminate_at, False, {}

        def close(self):
            self.closed = True

    def network(*args):
        model = FixturePolicy()
        models.append(model)
        return model

    monkeypatch.setattr(probe, "HarvestingEnv", FixtureEnvironment)
    monkeypatch.setattr(probe, "ObservationEncoder", FixtureEncoder)
    monkeypatch.setattr(probe, "build_q_network", network)
    batch = SimpleNamespace(observations=torch.zeros((117, 5)), actions=torch.ones(117, dtype=torch.long))
    batch.observations[76] = torch.from_numpy(raw_observation(0))
    batch.observations[115] = torch.from_numpy(raw_observation(39))
    episodes = [{"seed": 10002, "start": 76, "stop": 117}]
    return SimpleNamespace(reference=tmp_path, directory=directory, batch=batch, episodes=episodes,
                           environments=environments, models=models, rows=rows)


def test_reconstruction_counts_one_prefix_and_preserves_rng_and_sources(fixture):
    before = torch.get_rng_state().clone()
    paths = list(fixture.directory.glob("*"))
    hashes = {path: probe.file_hash(path) for path in paths}
    core, metadata, evidence = probe.build_trajectory_probe(fixture.reference, fixture.batch, fixture.episodes)
    assert torch.equal(before, torch.get_rng_state())
    assert len(fixture.environments) == 1
    env = fixture.environments[0]
    assert env.resets == [10002] and env.actions == [1] * 39 and env.closed
    assert [model.calls for model in fixture.models] == [40, 40, 40]
    assert evidence["accounting"] == {
        "environment_resets": 1, "environment_steps": 39, "completed_episodes": 0,
        "prefix_policy_forwards": 117, "step40_policy_forwards": 3,
        "optimizer_updates": 0, "new_demonstrations": 0}
    assert core["name"] == "scene_10002_actual_step_40"
    assert tuple(core["observation"].shape) == (1, 5)
    assert core["demonstrated_action"] == metadata["reference_action"] == 1
    assert metadata["reference_action_kind"] == "successful_policy_action"
    assert evidence["verification"]["historical_per_step_q_available"] is False
    assert len(evidence["prefix"]) == 39
    assert [row["greedy_action"] for row in evidence["step40_policies"]] == [1, 6, 1]
    assert [row["south_small_minus_east_large"] for row in evidence["step40_policies"]] == [2, -2, 2]
    comparison = evidence["demonstration_state_comparison"]
    assert comparison["global_transition_index"] == 115 and comparison["exactly_equal"]
    assert comparison["maximum_absolute_difference"] == 0
    assert hashes == {path: probe.file_hash(path) for path in paths}
    assert json.loads(json.dumps(evidence, allow_nan=False)) == evidence
    assert probe.observation_hash(np.asarray(evidence["encoded_observation"])) == metadata["encoded_observation_sha256"]


def test_matching_position_does_not_imply_matching_complete_observation(fixture):
    fixture.batch.observations[115, 2] += 0.01
    _, _, evidence = probe.build_trajectory_probe(fixture.reference, fixture.batch, fixture.episodes)
    comparison = evidence["demonstration_state_comparison"]
    assert not comparison["exactly_equal"] and not comparison["allclose_atol_1e_7_rtol_0"]
    assert comparison["different_feature_count"] == 1
    assert comparison["maximum_absolute_difference"] == pytest.approx(0.01, abs=1e-7)


@pytest.mark.parametrize("fault", ["action_prefix", "position_prefix", "step40", "outcome", "missing_row"])
def test_source_log_inconsistencies_are_rejected_before_environment_actions(fixture, fault):
    rows = fixture.rows
    rollout = rows[0]["scenes"][0]["rollout"]
    if fault == "action_prefix":
        rollout["actions"][5] = 2
    elif fault == "position_prefix":
        rollout["robot_path"][5][0] += 0.25
    elif fault == "step40":
        rollout["actions"][39] = 6
    elif fault == "outcome":
        rollout["success"] = False
    else:
        rows.pop()
    (fixture.directory / "observations.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        probe.build_trajectory_probe(fixture.reference, fixture.batch, fixture.episodes)
    assert not fixture.environments


@pytest.mark.parametrize("fault", ["hash", "configuration", "noise"])
def test_changed_checkpoints_and_undeclared_environment_work_are_rejected(fixture, fault):
    path = fixture.directory / "update_7500.pt"
    payload = torch.load(path, weights_only=True)
    if fault == "configuration":
        payload["dqn_config"]["gamma"] = 0.98
    elif fault == "noise":
        payload["environment_config"]["execution_noise_fraction"] = 0.2
    else:
        payload["online"]["marker"] += 1
    torch.save(payload, path)
    if fault != "hash":
        summary = probe.read_json(fixture.directory / "summary.json")
        for update in probe.CHECKPOINTS:
            other = fixture.directory / f"update_{update}.pt"
            if fault == "noise":
                saved = torch.load(other, weights_only=True)
                saved["environment_config"] = payload["environment_config"]
                torch.save(saved, other)
            summary["checkpoint_sha256"][other.name] = probe.file_hash(other)
        (fixture.directory / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        probe.build_trajectory_probe(fixture.reference, fixture.batch, fixture.episodes)
    assert not fixture.environments


def test_certification_is_skipped_only_with_exact_original_initial_state(fixture):
    summary = probe.read_json(fixture.directory / "summary.json")
    for update in probe.CHECKPOINTS:
        path = fixture.directory / f"update_{update}.pt"
        payload = torch.load(path, weights_only=True)
        payload["environment_config"]["verify_reference_route"] = True
        torch.save(payload, path)
        summary["checkpoint_sha256"][path.name] = probe.file_hash(path)
    (fixture.directory / "summary.json").write_text(json.dumps(summary))
    _, _, evidence = probe.build_trajectory_probe(fixture.reference, fixture.batch, fixture.episodes)
    gate = evidence["reset_certification"]
    assert gate["original_verify_reference_route"] is True
    assert gate["reconstruction_verify_reference_route"] is False
    assert gate["initial_encoded_observation_exact"] is True
    assert gate["original_initial_encoded_observation_sha256"] == gate["reconstructed_initial_encoded_observation_sha256"]
    assert fixture.environments[0].config["verify_reference_route"] is False


def test_different_accepted_scene_stops_before_any_policy_or_action(fixture):
    fixture.batch.observations[76, 2] += 0.01
    with pytest.raises(ValueError, match="exact original accepted scene"):
        probe.build_trajectory_probe(fixture.reference, fixture.batch, fixture.episodes)
    assert fixture.environments[0].resets == [10002]
    assert fixture.environments[0].actions == [] and fixture.environments[0].closed
    assert [model.calls for model in fixture.models] == [0, 0, 0]


def test_recomputed_wrong_action_is_rejected_without_advancing_environment(fixture, monkeypatch):
    monkeypatch.setattr(probe, "policy_values", lambda *args: [2.0] + [0.0] * 7)
    with pytest.raises(ValueError, match="policy action differs"):
        probe.build_trajectory_probe(fixture.reference, fixture.batch, fixture.episodes)
    env = fixture.environments[0]
    assert env.resets == [10002] and env.actions == [] and env.closed


def test_reconstruction_rejects_early_completion_and_closes_environment(fixture, monkeypatch):
    environment_class = probe.HarvestingEnv

    def factory(config):
        env = environment_class(config)
        env.terminate_at = 3
        return env

    monkeypatch.setattr(probe, "HarvestingEnv", factory)
    with pytest.raises(ValueError, match="completed an episode"):
        probe.build_trajectory_probe(fixture.reference, fixture.batch, fixture.episodes)
    assert fixture.environments[0].actions == [1] * 3
    assert fixture.environments[0].closed


def test_position_validation_does_not_round_source_log_to_float32(fixture):
    for row in fixture.rows:
        row["scenes"][0]["rollout"]["robot_path"][1][0] += 1e-10
    (fixture.directory / "observations.json").write_text(json.dumps(fixture.rows))
    with pytest.raises(ValueError, match="Reconstructed position differs"):
        probe.build_trajectory_probe(fixture.reference, fixture.batch, fixture.episodes)
    assert len(fixture.environments[0].actions) == 1


def test_invalid_comparison_demonstration_fails_before_reset(fixture):
    fixture.batch.actions[115] = 2
    with pytest.raises(ValueError, match="step-40 action differs"):
        probe.build_trajectory_probe(fixture.reference, fixture.batch, fixture.episodes)
    assert not fixture.environments
