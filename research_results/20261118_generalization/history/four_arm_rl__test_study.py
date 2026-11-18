"""Pre-execution checks use old training scenes only, never the held-out bank."""
import copy
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).parent))
import study as s
from evaluate import TrialEnv
torch.set_num_threads(1)


def test_budget_splits_and_rewards():
    assert len(s.schedule())==66 and set(s.schedule())==set(s.OLD+s.NEWTRAIN)
    assert not set(s.OLD+s.NEWTRAIN)&set(s.TEST)
    assert len(range(64,s.TOTAL+1,4))==3285
    assert s.reward_for(1,True,False,'control')==1
    assert abs(s.reward_for(1,True,False,'stagnation')-.7)<1e-12
    assert s.reward_for(-50,False,True,'damage')==-100
    assert s.reward_for(1,True,False,'noise')==1
    assert s.epsilon(0)==.2 and abs(s.epsilon(s.TOTAL-1)-.05)<1e-12


def test_restore_exact_source_and_single_update(tmp_path):
    agent,env,rng,state=s.c.restore(s.SOURCE,None)
    assert all(torch.equal(v,state['online'][k]) for k,v in agent.learner.online.state_dict().items())
    assert agent.learner.update_count==12000
    demos,_,_,recovery,groups=s.c.old_data(state['environment_config'])
    rows=[]
    for i in range(64):
        rows.append(dict(obs=demos['observations'][i].numpy(),next=demos['next_observations'][i].numpy(),
            action=int(demos['actions'][i]),raw_reward=float(demos['rewards'][i]),stagnant=False,damage=False,
            done=bool(demos['terminated'][i]),truncated=bool(demos['truncated'][i]),seed=10000))
    before={k:v.clone() for k,v in agent.learner.online.state_dict().items()}
    result=s.update(agent,rows,[],demos,groups,recovery,np.random.default_rng(9),'control')
    assert np.isfinite(result['loss']) and agent.learner.update_count==12001
    assert any(not torch.equal(v,before[k]) for k,v in agent.learner.online.state_dict().items())
    s.pack_rows(tmp_path/'rows.npz',rows)
    restored=s.unpack_rows(tmp_path/'rows.npz')
    assert len(restored)==64 and np.array_equal(restored[0]['obs'],rows[0]['obs'])
    env.close()


def test_noise_bounds_and_reset_pairing():
    config=torch.load(s.PREPARED,weights_only=True)['environment_config']
    clean=TrialEnv(config,0.,0);noisy=TrialEnv(config,.1,0)
    a,_=clean.reset(seed=10000);b,_=noisy.reset(seed=10000)
    np.testing.assert_array_equal(a,b)
    for action in range(8):
        commanded,executed=noisy._action_motion(action)
        size=noisy.small_step if action<4 else noisy.large_step
        assert np.max(np.abs(executed-commanded))<=.1*size+1e-7
    x,_=s.reset_env(clean,10000,0,0,0.)
    y,_=s.reset_env(noisy,10000,0,0,.1)
    np.testing.assert_array_equal(x,y)
    assert s.b.canonical(clean.np_random.bit_generator.state)==s.b.canonical(noisy.np_random.bit_generator.state)
    clean.close();noisy.close()


def test_same_control_and_noise_reward_without_perturbation():
    for stagnant in (False,True):
        for damage in (False,True):assert s.reward_for(-3,stagnant,damage,'control')==s.reward_for(-3,stagnant,damage,'noise')
