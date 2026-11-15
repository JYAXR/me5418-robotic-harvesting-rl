"""Audit the step-six single-action correction without simulator work."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("step6_test", ROOT / "docs/experiments/026_step6_rescue/run_step6_rescue.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def pair():
    rows = [{"step": i, "position": [i, 0], "next_position": [i + 1, 0],
             "encoded_observation_sha256": str(i), "q_values": [0.] * 8,
             "policy_action": 2, "executed_action": 2, "forced_action": False,
             "reward": 1.} for i in range(1, 27)]
    control = {"records": rows}
    corrected = deepcopy(control)
    corrected["records"][5].update(executed_action=1, forced_action=True)
    return control, corrected


def test_exact_action_selection():
    for intervene in (False, True):
        for step in range(1, 201):
            for policy in range(8):
                assert m.corrected_action(policy, step, intervene) == (1 if intervene and step == 6 else policy)


def test_valid_pair_leaves_later_decisions_autonomous():
    control, corrected = pair()
    assert m.paired_fork(control, corrected)["identical_prefix"]
    assert corrected["records"][24]["executed_action"] == 2


@pytest.mark.parametrize("index,key,value", [
    (0, "reward", 2.), (4, "executed_action", 1),
    (5, "q_values", [1.] * 8), (5, "encoded_observation_sha256", "changed"),
    (5, "executed_action", 0), (5, "forced_action", False),
    (24, "forced_action", True), (24, "executed_action", 4),
])
def test_rejects_extra_changes(index, key, value):
    control, corrected = pair()
    corrected["records"][index][key] = value
    with pytest.raises(ValueError):
        m.paired_fork(control, corrected)


def test_rejects_forced_control():
    control, corrected = pair()
    control["records"][25]["forced_action"] = True
    with pytest.raises(ValueError):
        m.paired_fork(control, corrected)


def test_required_helper_interfaces():
    assert callable(m.diagnostic.array_hash)
    assert callable(m.diagnostic.first_divergence)
    assert callable(m.v.multi.diagnostic.fit.fixed_scene_rollout)
    assert callable(m.v.multi.diagnostic.fit.episode_metrics)


@pytest.mark.parametrize("intervene", [False,True])
def test_independent_observer_avoids_reset_and_checks_transitions(monkeypatch, intervene):
    from types import SimpleNamespace

    import numpy as np
    import torch

    class Network(torch.nn.Module):
        def forward(self, inputs):
            return torch.tensor([[0.,1.,2.,0.,0.,0.,0.,0.]])
    class Agent:
        def __init__(self):
            self.learner=SimpleNamespace(online=Network())
        def act(self, obs, epsilon=0.):
            return 2
        def encoder(self, obs):
            return obs
    class Env:
        def __init__(self):
            self.index=0
        def step(self, action):
            self.index+=1
            return np.array([self.index,0],dtype=np.float32),0.,self.index==8,False,{
                "executed_motion":np.array([1.,0.]),"collision_blocked":False}
    def loop(agent, env, seed, part):
        # Simulate reset work before the first actual policy decision.
        env.step(7)
        env.index=0
        obs=np.array([0.,0.],dtype=np.float32)
        for _ in range(8):
            action=agent.act(obs,epsilon=0.)
            obs,*_=env.step(action)
        return {"steps":8}
    monkeypatch.setattr(m.v.multi.diagnostic.fit,"fixed_scene_rollout",loop)
    records=[]
    for i in range(8):
        obs=np.array([i,0],dtype=np.float32)
        records.append({"step":i+1,"position":obs.tolist(),"next_position":[float(i+1),0.],
            "encoded_observation_sha256":m.diagnostic.array_hash(obs),"q_values":[0.,1.,2.,0.,0.,0.,0.,0.],
            "policy_action":2,"executed_action":1 if intervene and i==5 else 2,
            "forced_action":intervene and i==5,"reward":0.,"terminated":i==7,"truncated":False,
            "stationary":False,"collision_blocked":False,"executed_motion":[1.,0.]})
    agent,env=Agent(),Env()
    expected={"records":records,"rollout":{"steps":8}}
    assert m.independent_rollout(agent,env,None,expected,intervene)==expected["rollout"]
    assert "act" not in agent.__dict__ and "step" not in env.__dict__
    expected["records"][3]["reward"]=1.
    with pytest.raises(ValueError,match="full transition"):
        m.independent_rollout(agent,env,None,expected,intervene)
    assert "act" not in agent.__dict__ and "step" not in env.__dict__
