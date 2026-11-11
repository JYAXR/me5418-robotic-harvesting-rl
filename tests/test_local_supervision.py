"""Check one-row weighting, exact neutral updates and immutable continuation state."""

import importlib.util
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch, demonstration_margin_loss
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "local016_test", ROOT / "docs/experiments/016_local_supervision/run_local_supervision.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture(counter=2199):
    torch.set_num_threads(1)
    torch.manual_seed(73)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8), learning_rate=1e-4,
                                   batch_size=3, target_update_interval=200), seed=73)
    batch = TransitionBatch(
        observations=torch.randn(3, env.observation_size), actions=torch.tensor([0, 1, 2]),
        rewards=torch.tensor([0.2, -0.1, 1.0]),
        next_observations=torch.randn(3, env.observation_size),
        terminated=torch.tensor([False, False, True]), truncated=torch.zeros(3, dtype=torch.bool),
    )
    for _ in range(2):
        agent.learner.update(batch, demonstration_weight=1)
    agent.learner.update_count = counter
    return agent, env, environment, batch


def test_neutral_selected_weight_exactly_matches_original_updates_across_sync():
    source, source_env, environment, batch = fixture()
    snapshot = runner.adam.training_snapshot(source, environment)
    baseline, baseline_env = runner.extension.restore_continuation(snapshot)
    treatment, treatment_env = runner.extension.restore_continuation(snapshot)
    try:
        assert not runner.exact_equal(snapshot["online"], snapshot["target"])
        for step in (2200, 2201, 2202):
            ordinary = baseline.learner.update(batch, demonstration_weight=1.0, demonstration_margin=0.8)
            weighted = runner.weighted_update(treatment.learner, batch, selected_row=1, selected_weight=1.0)
            assert asdict(ordinary) == asdict(weighted)
            assert ordinary.target_synchronized == (step == 2200)
            assert runner.exact_equal(runner.adam.training_snapshot(baseline, environment),
                                      runner.adam.training_snapshot(treatment, environment))
        assert runner.exact_equal(runner.adam.training_snapshot(source, environment), snapshot)
    finally:
        source_env.close()
        baseline_env.close()
        treatment_env.close()


def test_weight_two_changes_only_selected_hinge_coefficient_and_keeps_batch_denominator():
    q = torch.tensor([[0., 2., 1.], [1., 0., 2.], [0., 2., 3.]], requires_grad=True)
    actions = torch.tensor([0, 1, 2])
    loss = runner.weighted_margin_loss(q, actions, 1, 2.0)
    loss.backward()
    actual_gradient = q.grad.clone()
    q.grad = None
    original = demonstration_margin_loss(q, actions, 0.8)
    extra = torch.relu(torch.maximum(q[1, 0], q[1, 2]) + 0.8 - q[1, 1]) / 3
    independent = original + extra
    independent.backward()
    assert float(loss.detach()) == pytest.approx(float(independent.detach()))
    assert torch.equal(actual_gradient, q.grad)
    assert torch.equal(actual_gradient[0], torch.tensor([-1 / 3, 1 / 3, 0.]))
    assert torch.equal(actual_gradient[1], torch.tensor([0., -2 / 3, 2 / 3]))
    assert torch.equal(actual_gradient[2], torch.zeros(3))
    # Normalizing by the sum of weights would also reduce every ordinary sample.
    assert float(loss.detach()) != pytest.approx(8.4 / 4)


def test_supervision_probe_preserves_snapshot_rng_and_mixed_module_modes():
    agent, env, environment, batch = fixture()
    agent.learner.online.train()
    list(agent.learner.online.modules())[-1].eval()
    modes = [module.training for module in agent.learner.online.modules()]
    before = runner.adam.training_snapshot(agent, environment)
    try:
        result = runner.supervision_probe(agent, batch, 1, 2.0)
        assert result["weighted_margin"] == pytest.approx(
            result["unweighted_margin"] + result["selected_hinge"] / len(batch.actions), abs=1e-7)
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), before)
        assert [module.training for module in agent.learner.online.modules()] == modes
    finally:
        env.close()


@pytest.mark.parametrize("row,weight", [
    (-1, 2.), (3, 2.), (True, 2.), (1.0, 2.), (1, 0.), (1, -1.), (1, float("nan")),
    (1, float("inf")), (1, True),
])
def test_invalid_local_weight_rejected_before_mutation(row, weight):
    agent, env, environment, batch = fixture()
    before = runner.adam.training_snapshot(agent, environment)
    try:
        with pytest.raises(ValueError):
            runner.weighted_update(agent.learner, batch, row, weight)
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), before)
    finally:
        env.close()


def observation(agent, step):
    return {"update": step, "total_updates_on_branch": 40000 + step,
            "optimizer_state": runner.extension.control.optimizer_state_summary(agent.learner),
            "scenes": [{"seed": seed, "rollout": {"success": True, "steps": 3}}
                       for seed in (10000, 10001, 10002)],
            "aggregate": {"evaluation_environment_steps": 9}}


def fake_observer(agent, env, environment, batch, returns, episodes, phase, step, prior, directory):
    assert phase == "extension/td" and prior == 40000
    agent.save(directory / f"update_{step}.pt", environment, {"objective": phase, "updates": step})
    return observation(agent, step)


def test_arm_uses_exact_absolute_clock_and_separate_supervision_records(tmp_path, monkeypatch):
    agent, env, environment, batch = fixture(2199)
    snapshot = runner.adam.training_snapshot(agent, environment)
    direct, direct_env = runner.extension.restore_continuation(snapshot)
    expected_records = [asdict(runner.weighted_update(direct.learner, batch, 1, 2.0)) for _ in range(2)]
    expected_snapshot = runner.adam.training_snapshot(direct, environment)
    expected_start = observation(agent, 2199)
    monkeypatch.setattr(runner.multi, "observe", fake_observer)
    try:
        result = runner.execute_arm(
            agent, env, environment, batch, torch.zeros(3),
            [{"seed": 10001, "start": 0, "stop": 3}], tmp_path, "treatment", expected_start,
            start=2199, final=2201, checkpoints=(2199, 2200, 2201), selected_row=1, reference_steps=(1, 2),
        )
        directory = tmp_path / "treatment/td"
        records = runner.read_updates(directory / "updates.jsonl")
        assert records == [{"update": step, "extension_update": index + 1, **metrics}
                           for index, (step, metrics) in enumerate(zip((2200, 2201), expected_records))]
        assert runner.exact_equal(torch.load(directory / "training_state.pt", weights_only=True), expected_snapshot)
        supplemental = runner.read_updates(directory / "supervision.jsonl")
        assert len(supplemental) == 2
        assert all(row["selected_sample_weight"] == 2 and row["normalization_denominator"] == 3
                   for row in supplemental)
        assert result["selected_sample_weight"] == 2
        assert result["optimizer_updates"] == 2
        assert result["target_synchronizations"] == 1
        assert result["evaluation_environment_steps"] == 27
    finally:
        env.close()
        direct_env.close()


def test_initial_behavior_mismatch_stops_before_any_update(tmp_path, monkeypatch):
    agent, env, environment, batch = fixture(2000)
    before = runner.adam.training_snapshot(agent, environment)
    expected = observation(agent, 2000)
    expected["aggregate"]["evaluation_environment_steps"] = 99
    monkeypatch.setattr(runner.multi, "observe", fake_observer)
    try:
        with pytest.raises(ValueError, match="exact source behavior"):
            runner.execute_arm(agent, env, environment, batch, torch.zeros(3),
                               [{"seed": 10001, "start": 0, "stop": 3}], tmp_path, "baseline", expected,
                               start=2000, final=2001, checkpoints=(2000, 2001), selected_row=1,
                               reference_steps=(1,))
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), before)
        assert (tmp_path / "baseline/td/updates.jsonl").read_text() == ""
    finally:
        env.close()


def test_existing_destination_is_rejected_before_source_reads(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("Existing output must fail before reference-data loading")

    monkeypatch.setattr(runner, "load_inputs", forbidden)
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run(tmp_path / "reference", tmp_path / "archive", tmp_path)


def test_fixed_target_and_schedules_are_not_an_extra_data_or_td_weight_change():
    assert (runner.ROW, runner.SCENE, runner.STEP, runner.ACTION) == (55, 10001, 21, 0)
    assert runner.NORMALIZATION == 117
    assert runner.ARM_WEIGHTS == {"baseline": 1., "treatment": 2.}
    assert runner.CHECKPOINTS == (2000, *range(2001, 2011), 2200, 2201, *range(2500, 10001, 500))
    assert len(runner.CHECKPOINTS) * 3 * 2 == 174
    assert runner.SENTINEL_STEPS == (6, 18, 20, 21, 22, 23, 24)
    assert runner.TARGET_SOURCE_UPDATES == tuple(range(2400, 9401, 1000))


def test_selected_row_must_match_diagnosed_encoded_state_not_only_action_and_position():
    observations = torch.zeros(117, 341)
    batch = SimpleNamespace(observations=observations, actions=torch.zeros(117, dtype=torch.long))
    episode = {"seed": 10001, "start": 35, "stop": 76}
    digest = runner.rescue.diagnostic.array_hash(observations[55].numpy())
    assert runner.selected_identity(batch, episode, digest)["global_transition_index"] == 55
    # Change an internal environment feature while retaining robot coordinates and labels.
    observations[55, 10] = 0.1
    with pytest.raises(ValueError, match="encoded state exactly matches"):
        runner.selected_identity(batch, episode, digest)


def test_accounting_separates_shared_inherited_and_actual_new_work(tmp_path):
    for arm, count in (("baseline", 2), ("treatment", 1)):
        directory = tmp_path / arm / "td"
        directory.mkdir(parents=True)
        runner.fit.save_json(directory / "observations.json", [{
            "scenes": [{}, {}, {}], "aggregate": {"evaluation_environment_steps": 117}}])
        (directory / "updates.jsonl").write_text("".join('{"update":2001}\n' for _ in range(count)))
        (directory / "sentinel_q.jsonl").write_text("{}\n" * (count + 1))
    result = runner.write_accounting(tmp_path)
    assert result["optimizer_updates_total"] == 3
    assert result["inherited_shared_optimizer_updates"] == 42000
    assert result["evaluation_environment_steps"] == 234
    assert result["scheduled_evaluation_episodes"] == 6
    assert result["sentinel_records"] == 5
    assert result["sentinel_forward_passes"] == 35


def test_baseline_gate_rejects_changed_logs_before_treatment(tmp_path):
    output, reference = tmp_path / "output", tmp_path / "reference"
    current, previous = output / "baseline/td", reference / "td"
    current.mkdir(parents=True)
    previous.mkdir(parents=True)
    for directory in (current, previous):
        runner.fit.save_json(directory / "observations.json", [])
    (current / "updates.jsonl").write_text('{"update":2001,"loss":1.0}\n')
    (previous / "updates.jsonl").write_text('{"update":2001,"loss":2.0}\n')
    with pytest.raises(ValueError, match="baseline update logs"):
        runner.compare_baseline(output, reference)
    assert not (output / "baseline_equivalence.json").exists()


def test_snapshot_input_is_not_aliased_by_weighted_update():
    agent, env, environment, batch = fixture()
    snapshot = runner.adam.training_snapshot(agent, environment)
    before = deepcopy(snapshot)
    restored, restored_env = runner.extension.restore_continuation(snapshot)
    try:
        runner.weighted_update(restored.learner, batch, 1, 2.0)
        assert runner.exact_equal(snapshot, before)
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), before)
    finally:
        env.close()
        restored_env.close()
