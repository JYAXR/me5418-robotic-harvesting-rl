"""Check that the independent audit rejects hidden boundary-state changes."""

import importlib.util
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "single_factor_verification_test", ROOT / "docs/experiments/012_divergence_control/verify_control.py",
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


@pytest.fixture
def snapshot():
    torch.set_num_threads(1)
    torch.manual_seed(55)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8), batch_size=3), seed=55)
    batch = TransitionBatch(
        observations=torch.randn(3, env.observation_size), actions=torch.tensor([0, 1, 2]),
        rewards=torch.tensor([1.0, 2.0, 3.0]), next_observations=torch.randn(3, env.observation_size),
        terminated=torch.tensor([False, False, True]), truncated=torch.tensor([False, False, False]),
    )
    agent.learner.update(batch, demonstration_weight=1.0)
    payload = verifier.multi.diagnostic.adam.training_snapshot(agent, environment)
    settings = {"environment": environment, "dqn": asdict(agent.learner.config)}
    settings["dqn"]["hidden_sizes"] = list(settings["dqn"]["hidden_sizes"])
    try:
        yield payload, settings
    finally:
        env.close()


@pytest.mark.parametrize("factor,value,treatment,weight", [
    ("demonstration_weight", 2.0, False, 1.0), ("demonstration_weight", 2.0, True, 2.0),
    ("learning_rate", 0.0001, False, 1.0), ("learning_rate", 0.0001, True, 1.0),
])
def test_independent_boundary_keeps_exact_adam_moments(snapshot, factor, value, treatment, weight):
    original, settings = snapshot
    agent, env = verifier.audit.independent_restore(original)
    try:
        # The independent loader deliberately does not restore RNG; boundary validation still does.
        agent.rng.bit_generator.state = deepcopy(original["agent_rng_state"])
        torch.set_rng_state(original["torch_rng_state"])
        actual_weight = verifier.independent_intervention(agent, factor, value, treatment)
        payload = verifier.multi.diagnostic.adam.training_snapshot(agent, settings["environment"])
        verifier.verify_initial_state(payload, original, settings, factor, value, treatment)
        assert actual_weight == weight
        verifier.require_equal(payload["optimizer"]["state"], original["optimizer"]["state"], "test moments")
        assert agent.learner.update_count == 0
        assert agent.learner.optimizer.defaults["lr"] == (
            value if factor == "learning_rate" and treatment else settings["dqn"]["learning_rate"])
    finally:
        env.close()


@pytest.mark.parametrize("mutation", ["moment", "step", "gamma", "weights", "target", "rng"])
def test_initial_audit_rejects_undeclared_state_mutations(snapshot, mutation):
    original, settings = snapshot
    agent, env = verifier.audit.independent_restore(original)
    try:
        agent.rng.bit_generator.state = deepcopy(original["agent_rng_state"])
        torch.set_rng_state(original["torch_rng_state"])
        verifier.independent_intervention(agent, "demonstration_weight", 2.0, True)
        payload = verifier.multi.diagnostic.adam.training_snapshot(agent, settings["environment"])
        if mutation in ("moment", "step"):
            state = next(iter(payload["optimizer"]["state"].values()))
            state["exp_avg" if mutation == "moment" else "step"].add_(1)
        elif mutation == "gamma":
            payload["dqn_config"]["gamma"] = 0.8
        elif mutation in ("weights", "target"):
            next(iter(payload["online" if mutation == "weights" else "target"].values())).add_(1)
        else:
            payload["torch_rng_state"][0] ^= 1
        with pytest.raises(ValueError, match="Verification failed"):
            verifier.verify_initial_state(payload, original, settings, "demonstration_weight", 2.0, True)
    finally:
        env.close()


@pytest.mark.parametrize("factor,value", [("both", 2), ("learning_rate", 0),
                                         ("learning_rate", float("nan")), ("demonstration_weight", True)])
def test_single_factor_validation_rejects_unsupported_or_invalid_values(factor, value):
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.expected_settings({"dqn": {}}, factor, value, True)
