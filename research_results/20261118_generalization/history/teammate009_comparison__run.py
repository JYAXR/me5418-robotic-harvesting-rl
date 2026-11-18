"""Frozen donor009 on the exact prepared-policy confirmation bank, no learning."""
import csv
import importlib.util
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DONOR=ROOT.parent/'teammate009-evaluation'
COMMIT='9f82828f9d00fa12098a94e96939647c4fbcbdd7'
MODEL=DONOR/'models/your_method_009_geometry.pt'
EXPECTED='7a4100d51d1df74f25081593407683e6d637a0654ebeba82925d5ae60326d969'
PRIOR=ROOT/'experiments/prepared_confirmation_20261118/results'
SEEDS=list(range(2026112000,2026112100))
OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'experiments/final_comparison_20261118'))
spec=importlib.util.spec_from_file_location('paired_base',ROOT/'experiments/final_comparison_20261118/run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
recovery=base.module('donor_recovery_policy',DONOR/'src/me5418_harvesting/agents/recovery_policy.py')


def load():
    agent,env,_=recovery.RecoveryAgent.load(MODEL)
    env.close()
    return agent


def verify():
    p=base.read(OUT/'protocol.json')
    assert base.sha(MODEL)==EXPECTED==p['model_sha256']
    for path,digest in p['hashes'].items(): assert base.sha(path)==digest,path
    assert p['seeds']==SEEDS
    return p


def freeze():
    if (OUT/'protocol.json').exists(): return verify()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=DONOR,text=True).strip()==COMMIT
    assert base.sha(MODEL)==EXPECTED
    # Every inherited source file is byte-identical, including environment dynamics.
    for path in (ROOT/'src').rglob('*.py'):
        assert base.sha(path)==base.sha(DONOR/path.relative_to(ROOT)),path
    prior=base.read(PRIOR/'protocol.json')
    assert prior['seeds']==SEEDS
    payload=base.torch.load(MODEL,map_location='cpu',weights_only=True)
    assert base.canonical(payload['environment_config'])==base.canonical(prior['environment'])
    # These scene IDs were not present in the donor's tracked text records.
    names=subprocess.check_output(['git','ls-files'],cwd=DONOR,text=True).splitlines()
    for name in names:
        p=DONOR/name
        if p.suffix in ('.py','.json','.jsonl','.csv','.md','.yaml','.yml','.txt'):
            content=p.read_text(errors='replace')
            assert not any(str(s) in content for s in SEEDS),name
    files=list((ROOT/'src').rglob('*.py'))+[Path(__file__),Path(base.__file__),Path(recovery.__file__),MODEL,
        PRIOR/'protocol.json',PRIOR/'summary.json',DONOR/'docs/experiments/your_method/009_geometry_retention/cli_verification.json']
    files += [PRIOR/'episodes'/f'{s}.json' for s in SEEDS]
    base.save(OUT/'protocol.json',dict(created_unix=time.time(),donor_commit=COMMIT,model=str(MODEL),model_sha256=EXPECTED,
        seeds=SEEDS,environment=prior['environment'],hashes={str(p):base.sha(p) for p in files},
        purpose='Evaluate the preidentified donor009 checkpoint on the exact 100-scene prepared-policy confirmation bank. No training, model selection or replaced seeds.',
        known_results='Prepared-policy outcomes already observed (24/100). Donor candidate independently fixed in its prior repository; this is a retrospective paired comparison, not a jointly preregistered blind test.',
        rules='Original physics, 200 steps, zero execution noise, greedy model actions, no overrides; retain errors and all failures.',
        limitations='Different training histories and representations; endpoint comparison, not isolated causal effect or multi-initialization stability.'))
    return verify()


def smoke():
    agent=load();config=verify()['environment']
    expected=base.read(DONOR/'docs/experiments/your_method/009_geometry_retention/cli_verification.json')['records']
    records=[]
    for reference in expected:
        env=base.HarvestingEnv(config)
        try: result,_=base.rollout(agent,env,reference['seed'])
        finally:env.close()
        r=result['metrics']
        for target,source in [('success','success'),('damage','damage'),('steps','steps'),('raw_return','return'),('travel','travel_distance'),('peak_deformation','peak_deformation')]:
            assert r[target]==reference[source],(reference['seed'],target,r[target],reference[source])
        records.append(result)
    base.save(OUT/'known_three_reproduction.json',records)
    print('KNOWN THREE EXACT MATCH',[(r['metrics']['seed'],r['metrics']['steps']) for r in records],flush=True)


def initialize():
    global AGENT,CONFIG
    base.runtime();CONFIG=verify()['environment'];AGENT=load()


def evaluate(seed):
    path=OUT/'episodes'/f'{seed}.json'
    if path.exists():
        r=base.read(path)
        assert r['model_sha256']==EXPECTED and r['protocol_sha256']==base.sha(OUT/'protocol.json')
        return r['metrics']
    env=base.HarvestingEnv(CONFIG)
    try:
        r,initial=base.rollout(AGENT,env,seed)
        prior=base.read(PRIOR/'episodes'/f'{seed}.json')
        assert r['metrics']['initial_sha256']==prior['metrics']['initial_sha256']
        r['initial']=initial
    except Exception as exc:
        r=dict(metrics=dict(seed=seed,success=False,damage=False,timeout=False,error=f'{type(exc).__name__}: {exc}',steps=0))
    finally:env.close()
    r.update(model_sha256=EXPECTED,protocol_sha256=base.sha(OUT/'protocol.json'))
    base.save(path,r)
    return r['metrics']


def summarize():
    config=verify()['environment'];rows=[];checks=[]
    for seed in SEEDS:
        record=base.read(OUT/'episodes'/f'{seed}.json');r=record['metrics']
        assert record['model_sha256']==EXPECTED and record['protocol_sha256']==base.sha(OUT/'protocol.json')
        assert r['seed']==seed
        if not r['error']:
            assert len(record['trace'])==r['steps']
            assert r['initial_sha256']==base.read(PRIOR/'episodes'/f'{seed}.json')['metrics']['initial_sha256']
            assert sum(bool(r[k]) for k in ('success','damage','timeout'))==1
        rows.append(r)
    agent=load()
    for key in ('success','damage','timeout'):
        example=next((r for r in rows if r[key]),None)
        if example is None:continue
        env=base.HarvestingEnv(config)
        try: repeated,_=base.rollout(agent,env,example['seed'])
        finally:env.close()
        saved=base.read(OUT/'episodes'/f"{example['seed']}.json")
        assert base.canonical(repeated['metrics'])==base.canonical(example)
        assert base.canonical(repeated['trace'])==base.canonical(saved['trace'])
        base.save(OUT/'reload_checks'/f"{example['seed']}.json",repeated);checks.append(example['seed'])
    counts={k:sum(bool(r[k]) for r in rows) for k in ('success','damage','timeout','error')}
    assert sum(counts.values())==100
    pairs=[]
    for r in rows:
        p=base.read(PRIOR/'episodes'/f"{r['seed']}.json")['metrics']
        pairs.append(dict(seed=r['seed'],prepared_success=p['success'],donor_success=r['success'],
            prepared_damage=p['damage'],donor_damage=r['damage'],prepared_timeout=p['timeout'],donor_timeout=r['timeout'],
            prepared_steps=p['steps'],donor_steps=r['steps'],donor_error=r['error']))
    result=dict(requested=100,**counts,success_rate=counts['success']/100,
        prepared_success=sum(p['prepared_success'] for p in pairs),
        both_success=[p['seed'] for p in pairs if p['prepared_success'] and p['donor_success']],
        prepared_only_success=[p['seed'] for p in pairs if p['prepared_success'] and not p['donor_success']],
        donor_only_success=[p['seed'] for p in pairs if p['donor_success'] and not p['prepared_success']],
        reload_exact_match_seeds=checks,model_unchanged=True,
        episode_hashes={str(s):base.sha(OUT/'episodes'/f'{s}.json') for s in SEEDS})
    verify();base.save(OUT/'summary.json',result)
    with (OUT/'paired_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    base.save(OUT/'status.json',dict(stage='completed',completed=100))
    print('SUMMARY',counts,'prepared',result['prepared_success'],'paired both/prepared only/donor only',
        len(result['both_success']),len(result['prepared_only_success']),len(result['donor_only_success']),flush=True)


if __name__=='__main__':
    base.runtime();freeze();smoke()
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        for count,future in enumerate(as_completed([pool.submit(evaluate,s) for s in SEEDS]),1):
            r=future.result();base.save(OUT/'status.json',dict(stage='running',completed=count,total=100))
            print('COMPLETE',count,'/100',r['seed'],r['success'],flush=True)
    summarize()
