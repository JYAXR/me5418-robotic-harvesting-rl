"""Reconstructed historical objectives, isolated from production code."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('retention_existing_penalty',ROOT/'experiments/fresh_penalty_coordinates/run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
f,p=m.f,m.p
from me5418_harvesting.agents.dqn_learning import TransitionBatch,td_loss
from me5418_harvesting.agents.return_calibration import discounted_demonstration_returns
from me5418_harvesting.agents.optimizer_transition import begin_td_with_optimizer
from me5418_harvesting.agents.dqn_agent import ObservationEncoder
from me5418_harvesting.planners.collision_free import collision_free_actions

RECOVERY=ROOT/'docs/experiments/027_step6_supervision/evidence/recovery_sample.json'
INITIAL=ROOT/'experiments/fresh_coordinate_dqn/results/run01/relative/initial_state.pt'
PHASES=(('supervised',20000),('calibration',20000),('td',10000),('targeted',2000))
NEW=list(range(1170000,1170067))
DEV=list(range(1181000,1181016))
TEST=list(range(1182000,1182032))
ARMS=('full_joint','full_staged')
BOUNDARIES=(3000,5000,6000)
HISTORICAL_HASHES={
    'prepared':'b849bacf5ed5261c8bffeca6aab5982f5690c11297810f05335b22980a5e8ef2',
    'full_joint_selected':'caca5ed74fb71a3e2a106da6856c1ccf876b91c2077af708900c0db633b7fba7',
    'full_staged_selected':'5bc1df2d00088abc8da3366a729c1a97188a9528b55c06405cf4d4210c2ba830',
}


def runtime():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    p.save(temporary,value);temporary.replace(path)


def arrays(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def old_data(config):
    meta=read(p.DATA/'dataset.json')
    assert p.sha(p.DATA/'expanded_demonstrations.npz')==meta['npz_sha256']
    raw=arrays(p.DATA/'expanded_demonstrations.npz')
    assert not m.demonstration_stagnation(raw['observations'],raw['next_observations'],config).any()
    data={k:torch.from_numpy(f.policy.transform_encoded(v,True) if k in ('observations','next_observations') else v.copy()) for k,v in raw.items()}
    batch=TransitionBatch(**data);returns=torch.empty_like(batch.rewards)
    for e in meta['episodes']:
        sl=slice(e['start'],e['stop'])
        part=TransitionBatch(**{k:v[sl] for k,v in data.items()})
        returns[sl]=discounted_demonstration_returns(part,.99)
        rewards=part.rewards.double().numpy()
        independent=[np.dot(rewards[i:],.99**np.arange(len(rewards)-i)) for i in range(len(rewards))]
        assert np.max(np.abs(returns[sl].numpy()-independent))<.0002
    sample=read(RECOVERY)
    assert sample['seed']==10001 and sample['action']==4 and sample['supervision_only']
    recovery=torch.from_numpy(f.policy.transform_encoded(np.array([sample['observation']],dtype=np.float32),True))
    groups={e['seed']:list(range(e['start'],e['stop'])) for e in meta['episodes']}
    return data,batch,returns,recovery,groups


def restore(path,cache):
    state=torch.load(path,weights_only=True)
    env=f.HarvestingEnv(state['environment_config'],scene_cache=cache)
    a=f.policy.CoordinateAgent(env,f.CONFIG,f.SEED,True)
    a.learner.online.load_state_dict(state['online']);a.learner.target.load_state_dict(state['target'])
    a.learner.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    a.learner.update_count=state['update_count'];a.rng.bit_generator.state=copy.deepcopy(state['agent_rng_state'])
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(state['sampler_rng_state'])
    torch.set_rng_state(state['torch_rng_state'])
    return a,env,rng,state


def weighted_margin(values,actions,targeted=False):
    offsets=torch.full_like(values,.8);offsets.scatter_(1,actions[:,None],0.)
    terms=(values+offsets).max(1).values-values.gather(1,actions[:,None]).squeeze(1)
    weights=torch.ones_like(terms);weights[40]=weights[55]=2.
    if targeted:weights[[98,102,103]]=2.
    return (terms*weights).mean()


def prepare_update(agent,batch,returns,recovery,phase):
    """Arithmetic/order reproduced from the lost restored_stages/run.py."""
    l=agent.learner;l.online.train();l.optimizer.zero_grad(set_to_none=True)
    td=torch.zeros(())
    if phase in ('td','targeted'):td,_,_=td_loss(l.online,l.target,batch,l.config.gamma)
    values=l.online(batch.observations)
    margin=weighted_margin(values,batch.actions,phase=='targeted')
    rq=l.online(recovery);offsets=torch.full_like(rq,.8);offsets[:,4]=0.
    extra=((rq+offsets).max(1).values-rq[:,4]).sum()/117
    actual=F.smooth_l1_loss(values.gather(1,batch.actions[:,None]).squeeze(1),returns) if phase=='calibration' else torch.zeros(())
    loss=.5*td+margin+extra+actual;assert torch.isfinite(loss)
    loss.backward();torch.nn.utils.clip_grad_norm_(l.online.parameters(),10.,error_if_nonfinite=True)
    l.optimizer.step();l.update_count+=1
    sync=phase in ('td','targeted') and l.update_count%200==0
    if sync:l.target.load_state_dict(l.online.state_dict())
    return dict(loss=float(loss.detach()),td=float(td.detach()),margin=float(margin.detach()),
        recovery=float(extra.detach()),calibration=float(actual.detach()),target_synchronized=sync)


def phase_for(arm,step):
    if arm=='full_joint':return 'joint'
    return 'margin' if step<=3000 else 'calibration' if step<=5000 else 'td'


def margin_terms(q,actions):
    offsets=torch.full_like(q,.8);offsets.scatter_(1,actions[:,None],0.)
    return (q+offsets).max(1).values-q.gather(1,actions[:,None]).squeeze(1)


def scene_weights(groups,n,old_seeds):
    w=torch.zeros(n,dtype=torch.float32)
    for seeds in (list(old_seeds),[s for s in groups if s not in old_seeds]):
        for seed in seeds:
            ids=groups[seed];w[ids]=.5/len(seeds)/len(ids)
    assert torch.isclose(w.sum(),torch.tensor(1.)) and torch.all(w>0)
    return w


def followup_update(agent,data,weights,recovery,phase):
    l=agent.learner;q=l.online(data['observations'])
    chosen=q.gather(1,data['actions'][:,None]).squeeze(1)
    margin=(margin_terms(q,data['actions'])*weights).sum()
    rq=l.online(recovery);extra=margin_terms(rq,torch.tensor([4])).sum()/117
    mc=(F.smooth_l1_loss(chosen,data['returns'],reduction='none')*weights).sum() if phase in ('joint','calibration') else torch.zeros(())
    td=torch.zeros(())
    if phase in ('joint','td'):
        with torch.no_grad():
            nxt=l.target(data['next_observations']).max(1).values
            target=data['rewards']+.99*torch.where(data['terminated'],torch.zeros_like(nxt),nxt)
        td=(F.smooth_l1_loss(chosen,target,reduction='none')*weights).sum()
    loss=margin+extra+mc+.5*td;assert torch.isfinite(loss)
    l.optimizer.zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(l.online.parameters(),10.,error_if_nonfinite=True)
    l.optimizer.step();l.update_count+=1
    if phase in ('joint','td') and l.update_count%200==0:l.target.load_state_dict(l.online.state_dict())
    return dict(loss=float(loss.detach()),margin=float(margin.detach()),recovery=float(extra.detach()),
        mc=float(mc.detach()),td=float(td.detach()),gradient_norm=float(norm),phase=phase)


def probe(agent,data):
    with torch.no_grad():q=agent.learner.online(data['observations'])
    chosen=q.gather(1,data['actions'][:,None]).squeeze(1)
    return dict(rows=len(q),action_agreement=int((q.argmax(1)==data['actions']).sum()),
        margin_loss=float(p.demonstration_margin_loss(q,data['actions'],.8)),
        chosen_q_mean=float(chosen.mean()),return_mae=float((chosen-data['returns']).abs().mean()))
