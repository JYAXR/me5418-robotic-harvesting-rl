"""Rebuild lost prepared model, then rerun the two fixed retention arms.

This is a reproduction on previously used seeds, NOT a new blind experiment.
No historical outcome is used to alter the fixed objectives/budgets/selection.
"""
import argparse
import copy
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
import yaml

spec=importlib.util.spec_from_file_location('retention_rebuild_common',Path(__file__).with_name('common.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def verify(out):
    for path,digest in c.read(out/'protocol.json')['hashes'].items():assert c.p.sha(c.ROOT/path)==digest,path


def initialize(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():verify(out);return
    config=yaml.safe_load((c.ROOT/'configs/environment_multiroute.yaml').read_text())
    old,_,_,_,groups=c.old_data(config)
    sources=[Path(__file__),Path(c.__file__),Path(c.m.__file__),Path(c.f.__file__),Path(c.f.policy.__file__),Path(c.p.__file__),
        c.INITIAL,c.RECOVERY,c.ROOT/'configs/environment_multiroute.yaml',c.p.DATA/'dataset.json',c.p.DATA/'expanded_demonstrations.npz']
    sources+=list((c.ROOT/'src').rglob('*.py'))
    assert len(set(list(groups)+c.NEW+c.DEV+c.TEST))==148
    c.save(out/'protocol.json',dict(environment=config,old_train=list(groups),new_train=c.NEW,dev=c.DEV,test=c.TEST,
        seed=c.f.SEED,preparation=c.PHASES,arms=c.ARMS,updates_per_arm=6000,checkpoints=c.BOUNDARIES,
        purpose='Reproduce lost 2026-11-18 retention follow-up on the SAME known evaluation seeds. Not an independent new generalization result.',
        reconstruction='Prepared model lost; recreate random seed20261118 then full1440 supervised20000, calibration20000, TD10000, targeted2000. Adam reset before supervised/calibration, preserved into TD/targeted.',
        followup='Full4225 success transitions; old33/new67 each 50%, scene-balanced; original recovery action4/117. Same mature Adam, lr1e-4. Joint6000:margin+MC+.5TD; staged3000margin+2000marginMC+1000marginTD. Copy target at staged TD entry.',
        selection='Success, fewer damage, fewer total steps; earlier checkpoint ties; baseline eligible. All choices locked before rerun test.',
        online_interactions=0,historical_hashes=c.HISTORICAL_HASHES,
        limitations='Same seeds and single initialization; no fresh validation of generalization. Do not tune to match historical counts.',
        runtime=dict(torch=torch.__version__,numpy=np.__version__),hashes={str(p.relative_to(c.ROOT)):c.p.sha(p) for p in sources}))


def snapshot(folder,a,config,rng,extra):
    folder.mkdir(parents=True,exist_ok=True)
    c.f.snapshot(folder/'state.pt',a,config,rng,0)
    c.save(folder/'progress.json',extra)
    names=[name for name in ('state.pt','progress.json','updates.json','model.pt') if (folder/name).exists()]
    c.save(folder/'complete.json',{name:c.p.sha(folder/name) for name in names})


def check_snapshot(folder):
    for name,sha in c.read(folder/'complete.json').items():assert c.p.sha(folder/name)==sha


def prepare(out):
    verify(out);folder=out/'preparation';folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():
        done=c.read(folder/'done.json')
        assert c.p.sha(folder/'prepared.pt')==done['model_sha256']
        assert c.p.sha(folder/'prepared_state.pt')==done['state_sha256'];return
    config=c.read(out/'protocol.json')['environment'];data,batch,returns,recovery,_=c.old_data(config)
    completed=sorted(p for p in folder.glob('checkpoint_*') if (p/'complete.json').exists())
    if completed:
        latest=completed[-1];check_snapshot(latest)
        a,env,rng,_=c.restore(latest/'state.pt',c.p.CertifiedSceneCache(1));env.close()
        previous=c.read(latest/'progress.json');total=previous['total_updates']
    else:
        torch.manual_seed(c.f.SEED);rng=np.random.default_rng(c.f.SEED)
        env=c.f.HarvestingEnv(config);a=c.f.policy.CoordinateAgent(env,c.f.CONFIG,c.f.SEED,True);env.close()
        old_initial=torch.load(c.INITIAL,weights_only=True)
        c.m.equal(a.learner.online.state_dict(),old_initial['online'])
        c.m.equal(a.learner.target.state_dict(),old_initial['target'])
        c.m.equal(torch.get_rng_state(),old_initial['torch_rng_state'])
        assert not a.learner.optimizer.state and a.learner.update_count==0
        c.save(folder/'initial_check.json',dict(equal_retained_initial_weights_and_torch_rng=True,seed=c.f.SEED))
        total=0;previous={}
    cumulative=0;started=time.perf_counter()
    for phase,budget in c.PHASES:
        phase_start=cumulative;cumulative+=budget
        if total>=cumulative:continue
        start=max(0,total-phase_start)
        if start==0 and phase!='targeted':
            transition=c.begin_td_with_optimizer(a.learner,preserve_optimizer=phase=='td')
            c.save(folder/f'{phase}_transition.json',transition)
        logs=[]
        for i in range(start,budget):
            logs.append(dict(update=i+1,**c.prepare_update(a,batch,returns,recovery,phase)))
            total+=1
            if total%1000==0:
                status=dict(stage='preparation',phase=phase,phase_update=i+1,total_updates=total,total_budget=52000,process_seconds=time.perf_counter()-started)
                checkpoint=folder/f'checkpoint_{total:06d}'
                checkpoint.mkdir(exist_ok=True);c.save(checkpoint/'updates.json',logs);logs=[]
                snapshot(checkpoint,a,config,rng,status)
                c.save(checkpoint/'loss_hash.json',dict(sha256=c.p.sha(checkpoint/'updates.json')))
                c.save(out/'status.json',status);print('PREP',phase,i+1,total,'/52000',flush=True)
    assert total==52000 and a.learner.update_count==12000
    assert all(int(s['step'])==32000 for s in a.learner.optimizer.state.values())
    a.save(folder/'prepared.pt',config,dict(stage='prepared',online_steps=0))
    c.f.snapshot(folder/'prepared_state.pt',a,config,rng,0)
    data['returns']=returns
    result=dict(model_sha256=c.p.sha(folder/'prepared.pt'),state_sha256=c.p.sha(folder/'prepared_state.pt'),
        historical_model_sha256=c.HISTORICAL_HASHES['prepared'],training_probe=c.probe(a,data),gradient_updates=52000)
    result['historical_model_exact_file_match']=result['model_sha256']==result['historical_model_sha256']
    c.save(folder/'done.json',result);print('PREPARED',result,flush=True)


def dump_rows(path,rows):
    result={}
    for k in rows[0]:
        dtype=np.float32 if k in ('observations','next_observations','rewards','returns','raw_reward') else None
        result[k]=np.asarray([row[k] for row in rows],dtype=dtype)
    np.savez_compressed(path,**result)


def collect(out):
    verify(out);folder=out/'data';folder.mkdir(exist_ok=True)
    if (folder/'frozen.json').exists():
        for name,sha in c.read(folder/'frozen.json').items():assert c.p.sha(folder/name)==sha
        return
    config=c.read(out/'protocol.json')['environment'];env=c.f.HarvestingEnv(config,scene_cache=c.p.CertifiedSceneCache(80))
    encoder=c.ObservationEncoder(env);all_rows=[];records=[]
    for seed in c.NEW:
        dest=folder/f'new_{seed}';dest.mkdir(exist_ok=True)
        if not (dest/'complete.json').exists():
            obs,_=env.reset(seed=seed);plan=c.collision_free_actions(env,include_current_shape=True);rows=[]
            for action in plan:
                before=[b.current_nodes.copy() for b in env.scene.branches]
                nxt,raw,term,trunc,info=env.step(int(action))
                change=max((float(np.max(np.abs(b.current_nodes-old))) for b,old in zip(env.scene.branches,before)),default=0.)
                stall=bool(np.linalg.norm(info['executed_motion'])<1e-6 and change<1e-6)
                rows.append(dict(observations=encoder(obs),actions=int(action),rewards=float(raw-.3*stall),
                    next_observations=encoder(nxt),terminated=bool(term),truncated=bool(trunc),
                    raw_reward=float(raw),stagnant=stall,seed=int(seed)))
                obs=nxt
                if term or trunc:break
            value=0.
            for row in reversed(rows):value=row['rewards']+.99*value;row['returns']=value
            dump_rows(dest/'rows.npz',rows)
            record=dict(seed=seed,success=bool(info['success']),damage=bool(info['damage']),steps=len(rows))
            c.save(dest/'complete.json',record)
        rec=c.read(dest/'complete.json');records.append(rec)
        if rec['success'] and not rec['damage']:
            a=c.arrays(dest/'rows.npz');all_rows.extend({k:v[i] for k,v in a.items()} for i in range(len(a['actions'])))
        c.save(out/'collection_status.json',dict(completed=len(records),total=len(c.NEW)))
        print('COLLECT',len(records),'/67',rec,flush=True)
    env.close();dump_rows(folder/'new_success.npz',all_rows);c.save(folder/'collection.json',records)
    c.save(folder/'frozen.json',{name:c.p.sha(folder/name) for name in ('new_success.npz','collection.json')})
    # Reproduction gate: do not silently substitute a different dataset.
    assert all(r['success'] and not r['damage'] for r in records) and len(all_rows)==2785


def full_data(out):
    config=c.read(out/'protocol.json')['environment'];data,_,returns,recovery,groups=c.old_data(config)
    old_groups=copy.deepcopy(groups);data['returns']=returns
    raw=c.arrays(out/'data/new_success.npz');offset=len(data['actions'])
    assert not raw['stagnant'].any()
    for k in list(data):
        a=raw[k];a=c.f.policy.transform_encoded(a,True) if k in ('observations','next_observations') else a
        data[k]=torch.cat([data[k],torch.from_numpy(a).to(data[k].dtype)])
    for seed in np.unique(raw['seed']):groups[int(seed)]=(np.flatnonzero(raw['seed']==seed)+offset).tolist()
    assert len(groups)==100 and len(data['actions'])==4225
    return data,recovery,groups,old_groups


def evaluate(path,config,seeds,cache,folder):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        stamp=c.read(folder/'complete.json')
        assert stamp['model_sha256']==c.p.sha(path) and stamp['seeds']==seeds
        assert stamp['metrics_sha256']==c.p.sha(folder/'metrics.json')
        return c.read(folder/'metrics.json')
    a,e,_=c.f.policy.CoordinateAgent.load(path);e.close();rows=[]
    for seed in seeds:
        rows+=c.p.evaluate(a,config,[seed],cache,folder)
        c.save(folder/'metrics.json',rows)
        print('EVAL',folder.parent.name,folder.name,len(rows),'/',len(seeds),'success',sum(z['success'] for z in rows),flush=True)
    c.save(folder/'complete.json',dict(model_sha256=c.p.sha(path),seeds=seeds,metrics_sha256=c.p.sha(folder/'metrics.json')))
    return rows


def trials(out):
    verify(out);data,recovery,groups,old_groups=full_data(out);weights=c.scene_weights(groups,len(data['actions']),old_groups)
    config=c.read(out/'protocol.json')['environment'];cache=c.p.CertifiedSceneCache(128)
    baseline_path=out/'preparation/prepared.pt';source=out/'preparation/prepared_state.pt'
    base_rows=evaluate(baseline_path,config,c.DEV,cache,out/'baseline_validation')
    baseline=dict(label='baseline',path=str(baseline_path.relative_to(c.ROOT)),score=c.p.score(base_rows),new_td_updates=0)
    selections={}
    for arm in c.ARMS:
        folder=out/arm;folder.mkdir(exist_ok=True)
        if (folder/'selection.json').exists():selections[arm]=c.read(folder/'selection.json');continue
        completed=sorted(p for p in folder.glob('checkpoint_*') if (p/'complete.json').exists())
        last=int(completed[-1].name.split('_')[1]) if completed else 0
        if completed:check_snapshot(completed[-1])
        a,env,rng,_=c.restore(completed[-1]/'state.pt' if completed else source,cache);env.close()
        history=c.read(completed[-1]/'updates.json') if completed else []
        candidates=[baseline]
        for n in c.BOUNDARIES:
            if n>last:continue
            dest=folder/f'checkpoint_{n:06d}'
            rows=evaluate(dest/'model.pt',config,c.DEV,cache,dest/'validation')
            candidates.append(dict(label=f'checkpoint_{n:06d}',path=str((dest/'model.pt').relative_to(c.ROOT)),score=c.p.score(rows),new_td_updates=n if arm=='full_joint' else max(0,n-5000)))
        for step in range(last+1,6001):
            phase=c.phase_for(arm,step)
            if arm=='full_staged' and step==5001:a.learner.target.load_state_dict(a.learner.online.state_dict())
            history.append(dict(update=step,**c.followup_update(a,data,weights,recovery,phase)))
            if step%100==0:
                c.save(out/'status.json',dict(stage='followup',arm=arm,update=step,budget=6000));print('TRAIN',arm,step,'/6000',flush=True)
            if step%1000==0:
                dest=folder/f'checkpoint_{step:06d}';dest.mkdir(exist_ok=True)
                a.save(dest/'model.pt',config,dict(arm=arm,updates=step,phase=phase))
                c.save(dest/'updates.json',history)
                snapshot(dest,a,config,rng,dict(update=step,phase=phase))
                c.save(dest/'files.json',{name:c.p.sha(dest/name) for name in ('model.pt','updates.json')})
                if step in c.BOUNDARIES:
                    probes={name:c.probe(a,{k:v[sl] for k,v in data.items()}) for name,sl in [('old',slice(None,1440)),('new',slice(1440,None))]}
                    c.save(dest/'training_probes.json',probes)
                    rows=evaluate(dest/'model.pt',config,c.DEV,cache,dest/'validation')
                    candidates.append(dict(label=f'checkpoint_{step:06d}',path=str((dest/'model.pt').relative_to(c.ROOT)),score=c.p.score(rows),new_td_updates=step if arm=='full_joint' else max(0,step-5000)))
                    print('VALIDATION',arm,step,c.p.score(rows),probes,flush=True)
        selection=dict(selected=max(candidates,key=lambda z:tuple(z['score'])),candidates=candidates)
        selections[arm]=selection;c.save(folder/'selection.json',selection)
    frozen={'prepared':dict(path=baseline['path'],score=baseline['score'],new_td_updates=0)}
    for arm,s in selections.items():frozen[arm]=s['selected'].copy()
    for item in frozen.values():item['sha256']=c.p.sha(c.ROOT/item['path'])
    winner=max(frozen,key=lambda k:tuple(frozen[k]['score']))
    lock=dict(models=frozen,validation_recommendation=winner,locked_unix=time.time(),
        evaluation_role='Reproduction of known test seeds, not a new blind test')
    if (out/'selection_lock.json').exists():assert c.read(out/'selection_lock.json')['models']==frozen
    else:
        assert not list((out/'evaluation').glob('*/trace_*.json'))
        c.save(out/'selection_lock.json',lock)
    c.save(out/'status.json',dict(stage='final_evaluation'));results={}
    for name,item in frozen.items():
        path=c.ROOT/item['path'];assert c.p.sha(path)==item['sha256']
        results[name]=evaluate(path,config,list(old_groups)+c.TEST,cache,out/'evaluation'/name)
        c.save(out/'summary.json',results)
        print('FINAL',name,'old',c.p.score(results[name][:33]),'test',c.p.score(results[name][33:]),flush=True)
    comparisons={name:dict(actual=frozen[name]['sha256'],historical=c.HISTORICAL_HASHES['prepared' if name=='prepared' else name+'_selected'],
        exact_file_match=frozen[name]['sha256']==c.HISTORICAL_HASHES['prepared' if name=='prepared' else name+'_selected']) for name in frozen}
    c.save(out/'historical_hash_comparison.json',comparisons);verify(out)
    c.save(out/'status.json',dict(stage='completed'));print('HISTORICAL_HASH_COMPARISON',comparisons,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','collect','trials'],required=True)
    ap.add_argument('--output',type=Path,default=c.HERE/'results/run01');args=ap.parse_args()
    c.runtime();out=args.output.resolve();initialize(out)
    {'prepare':prepare,'collect':collect,'trials':trials}[args.phase](out)
