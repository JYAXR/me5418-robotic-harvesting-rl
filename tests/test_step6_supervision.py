"""Check isolated step-six weighting with unchanged recovery supervision."""
import importlib.util
from pathlib import Path

import pytest
import torch

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("focused_test", ROOT / "docs/experiments/027_step6_supervision/focused_update.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

OLD_SPEC = importlib.util.spec_from_file_location("prior_recovery_test", ROOT / "docs/experiments/025_recovery_supervision/recovery_update.py")
old_core = importlib.util.module_from_spec(OLD_SPEC)
OLD_SPEC.loader.exec_module(old_core)


def fixture():
    torch.set_num_threads(1)
    env = HarvestingEnv({"max_steps": 3})
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8), learning_rate=0.0001, target_update_interval=200), seed=91)
    torch.manual_seed(91)
    batch = TransitionBatch(torch.randn(117, env.observation_size), torch.arange(117) % 8,
        torch.linspace(-0.3, 1, 117), torch.randn(117, env.observation_size),
        torch.arange(117) % 4 == 0, torch.zeros(117, dtype=torch.bool))
    recovery = torch.randn(1, env.observation_size)
    agent.learner.update(batch, demonstration_weight=1.0)
    agent.learner.update_count = 2199
    payload = m.baseline.snapshot(agent, {"max_steps": 3})
    env.close()
    return payload, batch, recovery


def test_weight_one_exactly_reproduces_prior_and_target_clock():
    payload, batch, recovery = fixture()
    a, ea = m.v.prior.independent_restore(payload)
    b, eb = m.v.prior.independent_restore(payload)
    try:
        old = old_core.recovery_td_update(a.learner, batch, recovery)
        new = m.recovery_td_update(b.learner, batch, recovery, 1.0)
        assert {key:new[key] for key in old} == old
        m.baseline.equal(m.baseline.snapshot(a, {"max_steps":3}), m.baseline.snapshot(b, {"max_steps":3}), "zero-addition full state")
        assert new["target_synchronized"] and b.learner.update_count == 2200
        assert new["recovery_loss_before"] == old["recovery_loss_before"]
    finally:
        ea.close(); eb.close()


def test_independent_math_matches_with_only_supervised_addition():
    payload, batch, recovery = fixture()
    agents = [m.v.prior.independent_restore(payload) for _ in range(3)]
    try:
        before = {name:getattr(batch,name).clone() for name in batch.__dataclass_fields__}
        old = old_core.recovery_td_update(agents[0][0].learner, batch, recovery)
        actual = m.recovery_td_update(agents[1][0].learner, batch, recovery)
        check = m.independent_update(agents[2][0].learner, batch, recovery)
        assert actual == check
        m.baseline.equal(m.baseline.snapshot(agents[1][0], {"max_steps":3}), m.baseline.snapshot(agents[2][0], {"max_steps":3}), "independent full state")
        assert actual["td_loss_before"] == old["td_loss_before"]
        assert actual["demonstration_loss_before"] > old["demonstration_loss_before"]
        assert actual["recovery_loss_before"] == old["recovery_loss_before"]
        assert actual["loss_before"] == pytest.approx(old["loss_before"] + actual["demonstration_loss_before"] - old["demonstration_loss_before"])
        assert actual["recovery_loss_before"] > 0
        assert not torch.equal(next(agents[0][0].learner.online.parameters()), next(agents[1][0].learner.online.parameters()))
        assert all(torch.equal(getattr(batch,n),value) for n,value in before.items())
    finally:
        for _,env in agents:
            env.close()


def test_only_row40_gradient_changes_and_recovery_gradient_is_preserved():
    original = torch.arange(936,dtype=torch.float32).reshape(117,8).requires_grad_()
    actions = torch.zeros(117,dtype=torch.long)
    recovery = torch.arange(8,dtype=torch.float32).reshape(1,8).requires_grad_()
    old=m.focused_margin_loss(original,actions,1.) + m.recovery_term(recovery)
    new=m.focused_margin_loss(original,actions,2.) + m.recovery_term(recovery)
    old_grad,old_extra=torch.autograd.grad(old,(original,recovery),retain_graph=True)
    new_grad,new_extra=torch.autograd.grad(new,(original,recovery))
    expected=old_grad.clone();expected[40]*=2
    assert torch.equal(expected,new_grad)
    assert torch.equal(old_grad[55],new_grad[55])
    assert torch.equal(old_extra,new_extra)
    assert (new-old).item()==pytest.approx(m.local.margin_terms(original,actions)[40].item()/117,abs=1e-5)


@pytest.mark.parametrize("weight", [True,-1.,0.,0.5,float("nan")])
def test_rejects_other_coefficients(weight):
    _,batch,recovery = fixture()
    with pytest.raises(ValueError):
        m.validate(batch,recovery,weight)


def test_rejects_extra_td_row_and_bad_recovery():
    _,batch,recovery = fixture()
    with pytest.raises(ValueError):
        m.validate(batch,recovery.repeat(2,1),1.)
    with pytest.raises(ValueError):
        m.validate(batch,recovery.double(),1.)
    recovery[0,0] = float("nan")
    with pytest.raises(ValueError):
        m.validate(batch,recovery,1.)


@pytest.mark.parametrize("tail_action", [2,4])
def test_conditional_probe_never_forces_recovery_action(monkeypatch, tail_action):
    from types import SimpleNamespace

    import numpy as np
    spec = importlib.util.spec_from_file_location("recovery_runner_test", ROOT / "docs/experiments/027_step6_supervision/run_step6_supervision.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    class Agent:
        def act(self, obs, epsilon=0.):
            return tail_action
        def encoder(self, obs):
            return obs
    def loop(agent, env, seed, part):
        actions = [agent.act(np.array([i],dtype=np.float32),epsilon=0.) for i in range(26)]
        return {"actions":actions}
    monkeypatch.setattr(runner.v.multi.diagnostic.fit,"fixed_scene_rollout",loop)
    agent=Agent()
    result=runner.conditional_rollout(agent,None,SimpleNamespace(),torch.tensor([[24.]]),[6]*24)
    assert result["rollout"]["actions"] == [6]*24 + [tail_action]*2
    assert result["recovery_action"] == tail_action
    assert result["scripted_prefix_actions"] == 24
    assert result["autonomous_tail_actions"] == 2
    assert "act" not in agent.__dict__
