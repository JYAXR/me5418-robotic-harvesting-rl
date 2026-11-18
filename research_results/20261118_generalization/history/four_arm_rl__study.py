"""Bounded four-arm online DQN continuation; no new expert demonstrations."""
import argparse
from collections import deque, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import importlib.util
import multiprocessing as mp
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'experiments/final_comparison_20261118'))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
b=module('four_eval_base',ROOT/'experiments/final_comparison_20261118/run.py')
c=module('four_preparation',ROOT/'experiments/retention_rebuild_20261118/common.py')
SOURCE=ROOT/'experiments/retention_rebuild_20261118/results/run01/preparation/prepared_state.pt'
PREPARED=SOURCE.with_name('prepared.pt')
SOURCE_SHA='1e37ec748d5244acbc394f5d6cf86068583eb18d242dfbc203c3c418021dd94f'
OLD=[10000,10001,10002]+list(range(290000,290030))
NEWTRAIN=list(range(2026113000,2026113033))
TEST=list(range(2026114000,2026114100))
ARMS={'control':(0.,0.,0.),'stagnation':(.3,0.,0.),'damage':(0.,50.,0.),'noise':(0.,0.,.1)}
TOTAL=13200
SEED=20261118


def schedule():return [int(s) for s in np.random.default_rng(SEED+51).permutation(OLD+NEWTRAIN)]
def epsilon(step):return .2-.15*min(step/(TOTAL-1),1.)
def reward_for(raw,stagnant,damage,arm):
    stall,extra_damage,_=ARMS[arm]
    return float(raw-stall*bool(stagnant)-extra_damage*bool(damage))


def freeze():
    if (OUT/'protocol.json').exists():return verify()
    assert b.sha(PREPARED)==SOURCE_SHA
    state=torch.load(SOURCE,weights_only=True)
    inference=torch.load(PREPARED,weights_only=True)
    assert all(torch.equal(v,inference['online'][k]) for k,v in state['online'].items())
    assert state['update_count']==12000
    assert len(set(OLD+NEWTRAIN+TEST))==166
    pattern=re.compile(r'(?<!\d)202611(?:30|40)\d{2}(?!\d)')
    prior=[]
    names=subprocess.check_output(['rg','--files','--hidden','-g','!.git/**','-g','!.venv/**'],cwd=ROOT,text=True).splitlines()
    for name in names:
        path=ROOT/name
        if HERE in path.parents or path.suffix not in ('.py','.json','.jsonl','.csv','.md','.yaml','.yml','.txt'):continue
        if pattern.search(path.read_text(errors='replace')):prior.append(name)
    assert not prior,prior
    sources=list((ROOT/'src').rglob('*.py'))+list(HERE.glob('*.py'))+[Path(b.__file__),Path(b.coordinates.__file__),Path(c.__file__),Path(c.f.__file__),Path(c.m.__file__),Path(c.p.__file__),c.RECOVERY,c.p.DATA/'dataset.json',c.p.DATA/'expanded_demonstrations.npz',SOURCE,PREPARED]
    p=dict(created_unix=time.time(),source=str(SOURCE.relative_to(ROOT)),source_sha256=b.sha(SOURCE),prepared_sha256=SOURCE_SHA,
        arms=ARMS,train_seeds=OLD+NEWTRAIN,old_seeds=OLD,test_seeds=TEST,train_schedule=schedule(),steps_per_scene=200,
        interactions_per_arm=TOTAL,updates_per_arm=3285,checkpoint_steps=[4400,8800,13200],seed=SEED,
        environment=state['environment_config'],dqn_config=state['dqn_config'],
        exploration='Common pre-generated epsilon-greedy choices for all arms; epsilon .20 to .05 over 13200 interactions.',
        update='Every4 interactions after64 warmup:48 online TD (32 scene-balanced ordinary+16 hard)+16 scene-balanced original demo TD; .5 TD Huber +64 scene-balanced original demo margins (existing five focused rows weight2) +original recovery margin/117. No new expert labels.',
        state='Same full prepared online/target/Adam state, lr1e-4, target sync200, gamma.99, clip10; empty online replay, same sampling RNG.',
        noise='Uniform additive displacement independently per x/y bounded by .1*commanded step size. Reset certification always noiseless to keep layouts matched; then set noise and independent execution RNG.',
        reset_rng='SeedSequence([20261118, layout_seed, scene_slot, reset_count]); common across arms; certification RNG unaffected.',
        evaluation='Fixed final endpoints plus prepared, all33 old+100 new without noise. Additionally first20 new layouts x3 execution RNG repeats x5 policies with noise .1; greedy actions. Do not select/retrain on test; preserve baseline.',
        scope='Single common training RNG, finite pilot. Historical supervision retained, not pure-from-scratch RL. No new planner labels, rewards/physics during final evaluation unchanged. Noise condition exploratory, 60 runs are20 distinct layouts.',
        hashes={str(p.relative_to(ROOT)):b.sha(p) for p in sources})
    b.save(OUT/'protocol.json',p);return verify()


def verify():
    p=b.read(OUT/'protocol.json')
    for name,digest in p['hashes'].items():assert b.sha(ROOT/name)==digest,name
    return p


def reset_env(env,seed,slot,reset_count,noise):
    env.noise_fraction=0.
    obs,info=env.reset(seed=seed)
    env.noise_fraction=noise
    env.np_random=np.random.default_rng(np.random.SeedSequence([SEED,int(seed),int(slot),int(reset_count)]))
    return obs,info


def sample_ids(rows,n,rng):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['seed']].append(i)
    return c.p.balanced_indices(groups,n,rng)


def update(agent,rows,hard,demos,groups,recovery,rng,arm):
    ids=sample_ids(rows,32,rng)
    pool=list(hard) if hard else list(range(len(rows)))
    hardrows=[rows[i] for i in pool]
    ids += [pool[i] for i in sample_ids(hardrows,16,rng)]
    chosen=[rows[i] for i in ids]
    dids=c.p.balanced_indices(groups,16,rng)
    batch=c.TransitionBatch(
        torch.cat([torch.from_numpy(np.stack([r['obs'] for r in chosen])),demos['observations'][dids]]),
        torch.cat([torch.tensor([r['action'] for r in chosen]),demos['actions'][dids]]),
        torch.cat([torch.tensor([reward_for(r['raw_reward'],r['stagnant'],r['damage'],arm) for r in chosen],dtype=torch.float32),demos['rewards'][dids]]),
        torch.cat([torch.from_numpy(np.stack([r['next'] for r in chosen])),demos['next_observations'][dids]]),
        torch.cat([torch.tensor([r['done'] for r in chosen]),demos['terminated'][dids]]),
        torch.cat([torch.tensor([r['truncated'] for r in chosen]),demos['truncated'][dids]]))
    l=agent.learner;l.online.train();l.optimizer.zero_grad(set_to_none=True)
    td,_,_=c.td_loss(l.online,l.target,batch,l.config.gamma)
    mids=c.p.balanced_indices(groups,64,rng)
    values=l.online(demos['observations'][mids]);actions=demos['actions'][mids]
    offsets=torch.full_like(values,.8);offsets.scatter_(1,actions[:,None],0.)
    margins=(values+offsets).max(1).values-values.gather(1,actions[:,None]).squeeze(1)
    weights=torch.tensor([2. if i in (40,55,98,102,103) else 1. for i in mids])
    margin=(weights*margins).mean()
    rq=l.online(recovery);off=torch.full_like(rq,.8);off[:,4]=0.
    recovery_loss=((rq+off).max(1).values-rq[:,4]).sum()/117
    loss=.5*td+margin+recovery_loss
    assert torch.isfinite(loss)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(l.online.parameters(),10.,error_if_nonfinite=True)
    l.optimizer.step();l.update_count+=1
    if l.update_count%200==0:l.target.load_state_dict(l.online.state_dict())
    return dict(loss=float(loss.detach()),td=float(td.detach()),margin=float(margin.detach()),recovery=float(recovery_loss.detach()),gradient_norm=float(norm))


def pack_rows(path,rows):
    np.savez_compressed(path,**{key:np.asarray([r[key] for r in rows]) for key in rows[0]})
def unpack_rows(path):
    data=c.arrays(path);return [{k:v[i].item() if v[i].ndim==0 else v[i] for k,v in data.items()} for i in range(len(data['action']))]


def train(arm):
    b.runtime();protocol=verify();folder=OUT/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'done.json').exists():
        done=b.read(folder/'done.json');assert b.sha(folder/'final.pt')==done['model_sha256'];return done
    cache=b.CertifiedSceneCache(70)
    if (folder/'latest.json').exists():
        latest=b.read(folder/'latest.json');checkpoint=folder/latest['folder']
        for name,digest in latest['hashes'].items():assert b.sha(checkpoint/name)==digest
        agent,env,rng,state=c.restore(checkpoint/'state.pt',cache)
        progress=b.read(checkpoint/'progress.json');start=progress['next_slot']
        rows=unpack_rows(checkpoint/'replay.npz');hard=deque(progress['hard'],maxlen=5000)
        losses=progress['losses'];logs=progress['logs']
    else:
        agent,env,_,state=c.restore(SOURCE,cache);rng=np.random.default_rng(SEED)
        start=0;rows=[];hard=deque(maxlen=5000);losses=[];logs=[]
    demos,_,_,recovery,groups=c.old_data(protocol['environment'])
    assert not set(groups)&set(TEST)
    random=np.random.default_rng(SEED+9);coins=random.random(TOTAL);random_actions=random.integers(8,size=TOTAL)
    started=time.perf_counter();noise=ARMS[arm][2]
    for slot in range(start,len(schedule())):
        seed=schedule()[slot];reset_count=0;obs,info=reset_env(env,seed,slot,reset_count,noise)
        first,first_hash=b.initial_record(env,obs,info)
        b.save(folder/'initials'/f'{slot:03d}.json',dict(seed=seed,initial=first,initial_sha256=first_hash))
        recent=deque(maxlen=8);slotrows=[];success=damage=timeout=0;random_count=0
        for local in range(200):
            step=slot*200+local;random_choice=coins[step]<epsilon(step)
            action=int(random_actions[step]) if random_choice else agent.act(obs)
            random_count+=int(random_choice)
            before=np.stack([branch.current_nodes.copy() for branch in env.scene.branches])
            nxt,raw,term,trunc,info=env.step(action)
            changed=float(np.max(np.abs(np.stack([branch.current_nodes for branch in env.scene.branches])-before)))
            stagnant=bool(np.linalg.norm(info['executed_motion'])<1e-6 and changed<1e-6)
            row=dict(obs=agent.encoder(obs),next=agent.encoder(nxt),action=action,raw_reward=raw,
                training_reward=reward_for(raw,stagnant,info['damage'],arm),done=bool(term),truncated=bool(trunc),
                seed=seed,slot=slot,reset_count=reset_count,blocked=bool(info['collision_blocked']),stagnant=stagnant,
                damage=bool(info['damage']),success=bool(info['success']),budget=bool(info['budget_exhausted']),
                random_action=bool(random_choice),executed_motion=info['executed_motion'].copy())
            rows.append(row);slotrows.append(row);recent.append(len(rows)-1)
            if row['damage'] or row['blocked']:hard.extend(recent)
            obs=nxt
            if len(rows)>=64 and (step+1)%4==0:losses.append(dict(step=step+1,**update(agent,rows,hard,demos,groups,recovery,rng,arm)))
            if term or trunc:
                success+=int(info['success']);damage+=int(info['damage']);timeout+=int(info['budget_exhausted'] and not info['success'] and not info['damage'])
                if local<199:
                    reset_count+=1;obs,_=reset_env(env,seed,slot,reset_count,noise);recent.clear()
        pack_rows(folder/f'transitions_{slot:03d}.npz',slotrows)
        logs.append(dict(slot=slot,seed=seed,steps=(slot+1)*200,success=success,damage=damage,timeout=timeout,random_actions=random_count))
        b.save(folder/'status.json',dict(stage='training',steps=len(rows),total=TOTAL,updates=len(losses)))
        print('TRAIN',arm,(slot+1)*200,'/',TOTAL,flush=True)
        if (slot+1)%22==0:
            checkpoint=folder/f'checkpoint_{len(rows):06d}';checkpoint.mkdir(exist_ok=True)
            c.f.snapshot(checkpoint/'state.pt',agent,protocol['environment'],rng,len(rows))
            agent.save(checkpoint/'model.pt',protocol['environment'],dict(arm=arm,training_noise=noise,steps=len(rows),online_updates=len(losses)))
            pack_rows(checkpoint/'replay.npz',rows)
            b.save(checkpoint/'progress.json',dict(next_slot=slot+1,hard=list(hard),losses=losses,logs=logs))
            b.save(folder/'latest.json',dict(folder=checkpoint.name,hashes={p.name:b.sha(p) for p in checkpoint.iterdir() if p.is_file()}))
    assert len(rows)==TOTAL and len(losses)==3285 and agent.learner.update_count==15285
    agent.save(folder/'final.pt',protocol['environment'],dict(arm=arm,training_noise=noise,steps=TOTAL,online_updates=len(losses)))
    b.save(folder/'updates.json',losses);b.save(folder/'training.json',logs)
    result=dict(arm=arm,model_sha256=b.sha(folder/'final.pt'),interactions=len(rows),updates=len(losses),total_td_counter=agent.learner.update_count,
        random_actions=sum(r['random_action'] for r in rows),training_success_episodes=sum(r['success'] for r in rows),
        training_damage_episodes=sum(r['damage'] for r in rows),stagnant_transitions=sum(r['stagnant'] for r in rows),seconds_this_process=time.perf_counter()-started)
    env.close();verify();b.save(folder/'done.json',result);print('TRAIN DONE',result,flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['freeze','train'],required=True);args=parser.parse_args()
    freeze()
    if args.phase=='train':
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(train,arm) for arm in ARMS]):future.result()
        b.save(OUT/'training_complete.json',{arm:b.read(OUT/arm/'done.json') for arm in ARMS})
