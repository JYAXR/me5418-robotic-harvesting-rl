"""Check matching loaders, old baseline, split and cache transparency."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('final_comparison_runner',HERE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
torch.set_num_threads(1)


def test_catalogue_and_split():
    assert len(r.MODELS)==len({z[0] for z in r.MODELS})==24
    assert len(r.OLD)==33 and len(r.NEW)==100 and not set(r.OLD)&set(r.NEW)
    assert all((r.ROOT/z[3]).is_file() for z in r.MODELS)
    assert sum(z[1]=='dqn' for z in r.MODELS)==15


def test_all_native_loaders_and_same_environment():
    policies=[r.load_policy(r.ROOT/z[3]) for z in r.MODELS]
    config=policies[0][1]
    assert all(r.canonical(c)==r.canonical(config) for _,c in policies)
    env=r.HarvestingEnv(config);obs,_=env.reset(seed=10000)
    for agent,_ in policies:
        action=agent.act(obs)
        assert 0<=action<8 and action==agent.act(obs)
    env.close()


def test_original034_three_reference_rollouts():
    agent,config=r.load_policy(r.ROOT/r.MODELS[0][3])
    env=r.HarvestingEnv(config)
    for seed,steps in zip((10000,10001,10002),(40,47,43)):
        row,_=r.rollout(agent,env,seed)
        assert row['metrics']['success'] and not row['metrics']['damage']
        assert row['metrics']['steps']==steps
        assert abs(sum(x['reward'] for x in row['trace'])-row['metrics']['raw_return'])<1e-9
    env.close()


def test_scene_cache_does_not_change_rollout():
    agent,config=r.load_policy(r.ROOT/r.MODELS[14][3])
    env=r.HarvestingEnv(config,scene_cache=r.CertifiedSceneCache(2))
    first,initial=r.rollout(agent,env,10000)
    second,again=r.rollout(agent,env,10000)
    assert r.canonical(first)==r.canonical(second)
    assert r.canonical(initial)==r.canonical(again)
    assert first['metrics']['steps']==len(first['trace'])
    env.close()
