"""Paired, original-reward evaluation of fixed four-arm endpoints and baseline."""
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import time
import study as s

b=s.b
OUT=s.OUT/'evaluation'
MODELS={'prepared':s.PREPARED,**{arm:s.OUT/arm/'final.pt' for arm in s.ARMS}}


class TrialEnv(b.HarvestingEnv):
    def __init__(self,config,noise,repeat,cache=None):
        super().__init__(config,scene_cache=cache)
        self.trial_noise=noise;self.repeat=repeat
    def reset(self,*,seed=None,options=None):
        self.noise_fraction=0.
        obs,info=super().reset(seed=seed,options=options)
        self.noise_fraction=self.trial_noise
        if self.trial_noise:
            self.np_random=b.np.random.default_rng(b.np.random.SeedSequence([s.SEED,int(seed),700,self.repeat]))
        return obs,info


def freeze():
    s.verify()
    if (OUT/'protocol.json').exists():return verify()
    for arm in s.ARMS:
        done=b.read(s.OUT/arm/'done.json')
        assert done['updates']==3285 and done['interactions']==s.TOTAL
        assert b.sha(MODELS[arm])==done['model_sha256']
    # Same initial scene and exploration counts in all training arms.
    for slot in range(66):
        fingerprints=[b.read(s.OUT/a/'initials'/f'{slot:03d}.json')['initial_sha256'] for a in s.ARMS]
        assert len(set(fingerprints))==1
    assert len({b.read(s.OUT/a/'done.json')['random_actions'] for a in s.ARMS})==1
    b.save(OUT/'protocol.json',dict(created_unix=time.time(),training_protocol_sha256=b.sha(s.OUT/'protocol.json'),
        models={a:dict(path=str(p),sha256=b.sha(p)) for a,p in MODELS.items()},
        clean=s.OLD+s.TEST,noisy_layouts=s.TEST[:20],noisy_repeats=[0,1,2],noise_fraction=.1,
        definition='All predeclared final endpoints, no selection; original rewards, greedy actions; each of 20 noisy layouts repeated3 times is not60 independent layouts.'))
    return verify()


def verify():
    s.verify();p=b.read(OUT/'protocol.json')
    assert p['training_protocol_sha256']==b.sha(s.OUT/'protocol.json')
    for a,entry in p['models'].items():assert b.sha(MODELS[a])==entry['sha256']
    return p


def init():
    global AGENTS,CONFIG,CACHE
    b.runtime();verify();AGENTS={}
    CONFIG=s.verify()['environment'];CACHE=b.CertifiedSceneCache(2)
    for name,path in MODELS.items():
        AGENTS[name],config=b.load_policy(path)
        assert b.canonical(config)==b.canonical(CONFIG)


def task(key):
    condition,seed,repeat=key
    folder=OUT/condition/f'{seed}_{repeat}'
    if (folder/'complete.json').exists():
        stamp=b.read(folder/'complete.json')
        assert stamp['protocol_sha256']==b.sha(OUT/'protocol.json')
        for name,digest in stamp['hashes'].items():assert b.sha(folder/name)==digest
        return key
    first=None
    for name in MODELS:
        env=TrialEnv(CONFIG,.1 if condition=='noisy' else 0.,repeat,CACHE)
        try:
            record,initial=b.rollout(AGENTS[name],env,seed)
            fingerprint=record['metrics']['initial_sha256']
            if first is None:first=fingerprint;b.save(folder/'initial.json',initial)
            assert first==fingerprint,(key,name)
        except Exception as exc:
            b.save(folder/f'{name}.error.json',dict(type=type(exc).__name__,message=str(exc)))
            raise
        finally:env.close()
        record.update(condition=condition,repeat=repeat,model=name,model_sha256=b.sha(MODELS[name]))
        b.save(folder/f'{name}.json',record)
    files=['initial.json']+[f'{name}.json' for name in MODELS]
    b.save(folder/'complete.json',dict(protocol_sha256=b.sha(OUT/'protocol.json'),hashes={name:b.sha(folder/name) for name in files}))
    return key


def summary():
    verify();groups={'old':[], 'new_clean':[], 'new_noisy':[]}
    raw=[]
    for condition,seeds,repeats,label in [('clean',s.OLD,[0],'old'),('clean',s.TEST,[0],'new_clean'),('noisy',s.TEST[:20],[0,1,2],'new_noisy')]:
        result={}
        for name in MODELS:
            rows=[]
            for seed in seeds:
                for repeat in repeats:
                    folder=OUT/condition/f'{seed}_{repeat}';stamp=b.read(folder/'complete.json')
                    assert b.sha(folder/f'{name}.json')==stamp['hashes'][f'{name}.json']
                    record=b.read(folder/f'{name}.json');r=record['metrics']
                    assert record['model_sha256']==b.sha(MODELS[name])
                    assert r['steps']==len(record['trace']) and not r['error']
                    assert sum(bool(r[k]) for k in ('success','damage','timeout'))==1
                    rows.append(r);raw.append(dict(group=label,model=name,repeat=repeat,**r))
            result[name]=dict(n=len(rows),**{k:sum(bool(r[k]) for r in rows) for k in ('success','damage','timeout')},
                success_rate=sum(r['success'] for r in rows)/len(rows),
                mean_success_steps=sum(r['steps'] for r in rows if r['success'])/sum(r['success'] for r in rows) if any(r['success'] for r in rows) else None)
        groups[label]=result
    pairs={}
    for name in s.ARMS:
        pairs[name]={}
        for reference in ('prepared','control'):
            if reference==name:continue
            rows=[(b.read(OUT/'clean'/f'{seed}_0'/f'{reference}.json')['metrics'],b.read(OUT/'clean'/f'{seed}_0'/f'{name}.json')['metrics']) for seed in s.TEST]
            pairs[name][reference]=dict(gained=[a['seed'] for a,z in rows if not a['success'] and z['success']],
                lost=[a['seed'] for a,z in rows if a['success'] and not z['success']],both=[a['seed'] for a,z in rows if a['success'] and z['success']])
    # Repeat one clean and one noisy episode for each of five native reloaded policies.
    checks=[];config=s.verify()['environment']
    for name,path in MODELS.items():
        agent,_=b.load_policy(path)
        for condition,seed,repeat in [('clean',s.TEST[0],0),('noisy',s.TEST[0],0)]:
            env=TrialEnv(config,.1 if condition=='noisy' else 0.,repeat)
            try:r,_=b.rollout(agent,env,seed)
            finally:env.close()
            old=b.read(OUT/condition/f'{seed}_{repeat}'/f'{name}.json')
            assert b.canonical(r['metrics'])==b.canonical(old['metrics'])
            assert b.canonical(r['trace'])==b.canonical(old['trace'])
            checks.append(dict(model=name,condition=condition,seed=seed,repeat=repeat,exact=True))
            b.save(OUT/'reload_checks'/f'{name}_{condition}.json',r)
    verify()
    b.save(OUT/'summary.json',dict(groups=groups,paired_new_clean=pairs,reload_checks=checks,episodes=len(raw)))
    with (OUT/'per_episode.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    b.save(OUT/'status.json',dict(stage='completed',episodes=len(raw),reloads=len(checks)))
    print('FINAL',groups,flush=True)


if __name__=='__main__':
    freeze()
    tasks=[('clean',seed,0) for seed in s.OLD+s.TEST]+[('noisy',seed,rep) for seed in s.TEST[:20] for rep in range(3)]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=init) as pool:
        for i,future in enumerate(as_completed([pool.submit(task,key) for key in tasks]),1):
            key=future.result();b.save(OUT/'status.json',dict(stage='evaluating',completed_tasks=i,total_tasks=len(tasks)))
            print('EVAL',i,'/',len(tasks),key,flush=True)
    b.runtime();summary()
