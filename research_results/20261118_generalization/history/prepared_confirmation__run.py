"""A second 100-scene frozen confirmation of the prepared policy; no training."""
import csv
import hashlib
import importlib.util
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/final_comparison_20261118'))
spec = importlib.util.spec_from_file_location('frozen_comparison', ROOT/'experiments/final_comparison_20261118/run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
MODEL = ROOT/'experiments/retention_rebuild_20261118/results/run01/preparation/prepared.pt'
EXPECTED = '1e37ec748d5244acbc394f5d6cf86068583eb18d242dfbc203c3c418021dd94f'
SEEDS = list(range(2026112000, 2026112100))
OUT = HERE/'results'


def freeze():
    assert base.sha(MODEL) == EXPECTED
    if (OUT/'protocol.json').exists():
        return verify()
    # Search all existing tracked or untracked experiment text, excluding this new folder.
    candidates = subprocess.check_output(['rg', '--files', '--hidden', '-g', '!.git/**', '-g', '!.venv/**'], cwd=ROOT, text=True).splitlines()
    pattern = re.compile(r'(?<!\d)20261120\d{2}(?!\d)')
    scanned = 0
    for name in candidates:
        path = ROOT/name
        if HERE in path.parents or path.suffix not in ('.py','.json','.jsonl','.csv','.md','.yaml','.yml','.txt'): continue
        scanned += 1
        assert not pattern.search(path.read_text(errors='replace')), f'Previously used seed range: {name}'
    agent, config = base.load_policy(MODEL)
    assert config['max_steps'] == 200 and config['execution_noise_fraction'] == 0
    sources = list((ROOT/'src').rglob('*.py')) + [Path(__file__), Path(base.__file__), Path(base.coordinates.__file__)]
    base.save(OUT/'protocol.json', dict(created_unix=time.time(),model=str(MODEL.relative_to(ROOT)),model_sha256=EXPECTED,
        seeds=SEEDS,environment=config,scanned_text_files=scanned,
        purpose='Confirm one preselected prepared policy on a second disjoint bank of 100 new seeds. No training, tuning, model selection, outcome-based replacement or added noise.',
        prior_result='Same prepared checkpoint: 25/100 on seeds 2026111800--2026111899; 33/33 on original training scenes.',
        scope='Same-generator unseen layouts; not changed plant physics or real-world generalization. Practical 100-scene budget.',
        source_hashes={str(p.relative_to(ROOT)):base.sha(p) for p in sources}))
    return verify()


def verify():
    p = base.read(OUT/'protocol.json')
    assert p['seeds'] == SEEDS and base.sha(MODEL) == p['model_sha256'] == EXPECTED
    for path, digest in p['source_hashes'].items(): assert base.sha(ROOT/path) == digest, path
    return p


def initialize():
    global AGENT, CONFIG
    base.runtime()
    p = verify()
    AGENT, CONFIG = base.load_policy(MODEL)
    assert base.canonical(CONFIG) == base.canonical(p['environment'])


def evaluate(seed):
    path = OUT/'episodes'/f'{seed}.json'
    if path.exists():
        r = base.read(path)
        assert r['model_sha256'] == EXPECTED and r['protocol_sha256'] == base.sha(OUT/'protocol.json')
        return r['metrics']
    env = base.HarvestingEnv(CONFIG)
    try:
        record, initial = base.rollout(AGENT, env, seed)
        record['initial'] = initial
    except Exception as exc:
        record = dict(metrics=dict(seed=seed,success=False,damage=False,timeout=False,error=f'{type(exc).__name__}: {exc}',steps=0))
    finally:
        env.close()
    record.update(model_sha256=EXPECTED,protocol_sha256=base.sha(OUT/'protocol.json'))
    base.save(path,record)
    return record['metrics']


def summarize():
    verify()
    records = [base.read(OUT/'episodes'/f'{s}.json') for s in SEEDS]
    rows = [r['metrics'] for r in records]
    for s,r in zip(SEEDS, records):
        assert r['metrics']['seed'] == s and r['model_sha256'] == EXPECTED
        assert r['protocol_sha256'] == base.sha(OUT/'protocol.json')
        if not r['metrics']['error']:
            assert len(r['trace']) == r['metrics']['steps']
            assert sum(bool(r['metrics'][k]) for k in ('success','damage','timeout')) == 1
            digest = hashlib.sha256(base.canonical(r['initial']).encode()).hexdigest()
            assert digest == r['metrics']['initial_sha256']
    counts = {k:sum(bool(r[k]) for r in rows) for k in ('success','damage','timeout','error')}
    assert sum(counts.values()) == 100
    success = counts['success']; p = success/100; z = 1.95996398454
    center = (p+z*z/200)/(1+z*z/100)
    half = z*((p*(1-p)/100+z*z/40000)**.5)/(1+z*z/100)
    # Fresh model reload and uncached repeat of one seed per observed outcome.
    agent, config = base.load_policy(MODEL)
    checked = []
    for key in ('success','damage','timeout'):
        selected = next((r for r in rows if r[key]), None)
        if selected is None: continue
        env = base.HarvestingEnv(config)
        try: repeated, _ = base.rollout(agent, env, selected['seed'])
        finally: env.close()
        original = base.read(OUT/'episodes'/f"{selected['seed']}.json")
        assert base.canonical(repeated['metrics']) == base.canonical(selected)
        assert base.canonical(repeated['trace']) == base.canonical(original['trace'])
        checked.append(selected['seed'])
        base.save(OUT/'reload_checks'/f"{selected['seed']}.json", repeated)
    verify()
    result = dict(requested=100,**counts,success_rate=p,wilson95=[center-half,center+half],
        mean_success_steps=sum(r['steps'] for r in rows if r['success'])/success if success else None,
        reload_exact_match_seeds=checked,model_unchanged=True,
        episode_hashes={str(s):base.sha(OUT/'episodes'/f'{s}.json') for s in SEEDS})
    base.save(OUT/'summary.json', result)
    with (OUT/'per_episode.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=sorted(set().union(*(r.keys() for r in rows)))
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    base.save(OUT/'status.json',dict(stage='completed',completed=100,summary=result))
    print('SUMMARY',counts,'success_rate',p,'reload checks',checked,flush=True)


if __name__ == '__main__':
    freeze()
    done = 0
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        for future in as_completed([pool.submit(evaluate,s) for s in SEEDS]):
            row=future.result();done+=1
            base.save(OUT/'status.json',dict(stage='running',completed=done,total=100))
            print('COMPLETE',done,'/100',row['seed'],row['success'],flush=True)
    base.runtime()
    summarize()
