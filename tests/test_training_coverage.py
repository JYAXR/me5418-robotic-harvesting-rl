"""Check controlled data expansion without modifying the original objective at N=117."""

import importlib.util
from pathlib import Path

import pytest
import torch

from me5418_harvesting.agents.dqn_learning import TransitionBatch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("test029", ROOT / "docs/experiments/029_training_coverage/run_coverage.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def test_split_excludes_prior_holdout_and_training_scenes():
    assert len(r.NEW_SEEDS) == 30 and len(r.TEST_SEEDS) == 100
    groups = [set(r.OLD_SEEDS), set(r.NEW_SEEDS), set(r.TEST_SEEDS), set(range(280000, 280100))]
    for i, group in enumerate(groups):
        assert all(not group & other for other in groups[i + 1:])


def test_expanded_margin_preserves_weights_and_data_mean():
    values = torch.zeros((130, 8), requires_grad=True)
    actions = torch.zeros(130, dtype=torch.long)
    loss = r.core.focused_margin_loss(values, actions, 2)
    assert float(loss.detach()) == pytest.approx(0.8 * 132 / 130)
    loss.backward()
    assert values.grad[40, 0] == pytest.approx(-2 / 130)
    assert values.grad[55, 0] == pytest.approx(-2 / 130)
    assert values.grad[0, 0] == pytest.approx(-1 / 130)
    assert values.grad[-1, 0] == pytest.approx(-1 / 130)
    recovery = torch.zeros((1, 8), requires_grad=True)
    assert float(r.core.recovery_term(recovery).detach()) == pytest.approx(0.8 / 117)


def test_exact_old_control_and_independent_expanded_step():
    torch.set_num_threads(1)
    source, batch, recovery, _ = r.source_inputs()
    expanded = TransitionBatch(*(torch.cat((getattr(batch, key), getattr(batch, key)[:11]), 0)
                                for key in TransitionBatch.__dataclass_fields__))
    agents = [r.restore(source) for _ in range(4)]
    try:
        for agent, _ in agents:
            r.equal(r.snapshot(agent, source["environment_config"]), source, "Exact inherited state")
        control = r.legacy.core.recovery_td_update(agents[0][0].learner, batch, recovery)
        actual = r.core.recovery_td_update(agents[1][0].learner, batch, recovery)
        r.equal(control, actual, "Exact old loss and update")
        r.equal(r.snapshot(agents[0][0], source["environment_config"]), r.snapshot(agents[1][0], source["environment_config"]), "Exact old updated state")
        actual = r.core.recovery_td_update(agents[2][0].learner, expanded, recovery)
        independent = r.core.independent_update(agents[3][0].learner, expanded, recovery)
        r.equal(actual, independent, "Expanded independent loss and update")
        r.equal(r.snapshot(agents[2][0], source["environment_config"]), r.snapshot(agents[3][0], source["environment_config"]), "Expanded independent state")
    finally:
        for _, env in agents:
            env.close()
