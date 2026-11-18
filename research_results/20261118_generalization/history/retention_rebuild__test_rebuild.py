import importlib.util
from pathlib import Path
import numpy as np
import torch

spec=importlib.util.spec_from_file_location('retention_rebuild_tests',Path(__file__).with_name('run.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
c=r.c


def fresh():
    torch.set_num_threads(1)
    return c.restore(c.INITIAL,c.p.CertifiedSceneCache(1))


def test_retained_data_and_returns():
    a,e,_,state=fresh();e.close()
    data,batch,ret,recovery,groups=c.old_data(state['environment_config'])
    assert len(groups)==33 and len(batch.actions)==1440 and recovery.shape==(1,341)
    assert len(set(list(groups)+c.NEW+c.DEV+c.TEST))==148
    assert torch.isfinite(ret).all()


def test_balancing_and_phase_boundaries():
    groups={0:[0,1],1:[2],2:[3,4,5],3:[6]}
    w=c.scene_weights(groups,7,[0,1])
    for ids in groups.values():assert torch.allclose(w[ids].sum(),torch.tensor(.25))
    assert [c.phase_for('full_staged',n) for n in (3000,3001,5000,5001)]==['margin','calibration','calibration','td']


def test_margin_and_targeted_weights():
    torch.manual_seed(3);q=torch.randn(120,8);actions=torch.arange(120)%8
    terms=c.margin_terms(q,actions)
    assert torch.equal(terms.mean(),c.p.demonstration_margin_loss(q,actions,.8))
    weights=torch.ones(120);weights[[40,55]]=2
    assert torch.equal(c.weighted_margin(q,actions), (terms*weights).mean())
    weights[[98,102,103]]=2
    assert torch.equal(c.weighted_margin(q,actions,True),(terms*weights).mean())


def test_phase_optimizer_transition():
    a,e,_,state=fresh();e.close()
    data,batch,returns,recovery,_=c.old_data(state['environment_config'])
    c.begin_td_with_optimizer(a.learner,preserve_optimizer=False)
    c.prepare_update(a,batch,returns,recovery,'supervised')
    assert a.learner.update_count==1
    c.begin_td_with_optimizer(a.learner,preserve_optimizer=False)
    assert not a.learner.optimizer.state and a.learner.update_count==0
    c.prepare_update(a,batch,returns,recovery,'calibration')
    c.begin_td_with_optimizer(a.learner,preserve_optimizer=True)
    assert a.learner.update_count==0 and all(int(v['step'])==1 for v in a.learner.optimizer.state.values())
    c.prepare_update(a,batch,returns,recovery,'td')
    assert a.learner.update_count==1 and all(int(v['step'])==2 for v in a.learner.optimizer.state.values())
    assert all(p.grad is None for p in a.learner.target.parameters())


def test_complete_snapshot_resumes_same_update(tmp_path):
    a,e,rng,state=fresh();e.close()
    data,batch,returns,recovery,_=c.old_data(state['environment_config'])
    c.prepare_update(a,batch,returns,recovery,'supervised')
    r.snapshot(tmp_path,a,state['environment_config'],rng,dict(total_updates=1))
    r.check_snapshot(tmp_path)
    b,other,_,_=c.restore(tmp_path/'state.pt',c.p.CertifiedSceneCache(1));other.close()
    first=c.prepare_update(a,batch,returns,recovery,'supervised')
    second=c.prepare_update(b,batch,returns,recovery,'supervised')
    assert first==second
    for item in ('online','target'):c.m.equal(getattr(a.learner,item).state_dict(),getattr(b.learner,item).state_dict())
    c.m.equal(a.learner.optimizer.state_dict(),b.learner.optimizer.state_dict())


def test_followup_objectives_sync_and_resume(tmp_path):
    a,e,rng,state=fresh();e.close()
    data,_,returns,recovery,_=c.old_data(state['environment_config'])
    data={k:v[:16] for k,v in data.items()};data['returns']=returns[:16];w=torch.full((16,),1/16)
    a.learner.update_count=199
    prior={k:v.clone() for k,v in a.learner.target.state_dict().items()}
    stats=c.followup_update(a,data,w,recovery,'margin')
    assert stats['mc']==stats['td']==0
    c.m.equal(a.learner.target.state_dict(),prior)
    r.snapshot(tmp_path,a,state['environment_config'],rng,dict(total_updates=1))
    b,env,_,_=c.restore(tmp_path/'state.pt',c.p.CertifiedSceneCache(1));env.close()
    a.learner.update_count=b.learner.update_count=399
    left=c.followup_update(a,data,w,recovery,'joint');right=c.followup_update(b,data,w,recovery,'joint')
    assert left==right and np.isfinite([v for k,v in left.items() if k!='phase']).all()
    c.m.equal(a.learner.online.state_dict(),a.learner.target.state_dict())
    c.m.equal(a.learner.online.state_dict(),b.learner.online.state_dict())
    c.m.equal(a.learner.optimizer.state_dict(),b.learner.optimizer.state_dict())
