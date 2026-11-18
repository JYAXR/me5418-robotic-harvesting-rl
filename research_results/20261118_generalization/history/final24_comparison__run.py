"""Frozen, paired evaluation only; never trains or changes existing models."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np
import torch

from catalogue import ROOT, HERE, MODELS, OLD, NEW, EXCLUDED
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.envs.harvesting_env import HarvestingEnv
from me5418_harvesting.envs.scene_cache import CertifiedSceneCache

OUT = HERE / 'results/run01'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


coordinates = module('comparison_coordinates', ROOT/'experiments/fresh_coordinate_dqn/policy.py')
imitation = module('comparison_imitation', ROOT/'experiments/imitation_routes/run.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, default=encode)


def read(path): return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=encode))
    temp.replace(path)


def runtime():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def load_policy(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('method') == 'behavior_cloning':
        agent = imitation.Policy(payload['environment_config'], payload['relative'])
        agent.net.load_state_dict(payload['network']); agent.net.eval(); agent.env.close()
    elif payload.get('format_version') == 'fresh-coordinate-dqn-v1':
        agent, env, _ = coordinates.CoordinateAgent.load(path); env.close()
    else:
        agent, env, _ = DQNAgent.load(path); env.close()
    return agent, payload['environment_config']


def verify():
    p = read(OUT/'protocol.json')
    for name, digest in p['hashes'].items():
        assert sha(ROOT/name) == digest, name
    return p


def freeze():
    if (OUT/'protocol.json').exists(): return verify()
    assert len(MODELS) == 24 and len(set(OLD+NEW)) == 133
    configs = []
    for _, _, _, name in MODELS:
        configs.append(torch.load(ROOT/name, weights_only=True, map_location='cpu')['environment_config'])
    assert all(canonical(c) == canonical(configs[0]) for c in configs)
    assert configs[0]['max_steps'] == 200 and configs[0]['execution_noise_fraction'] == 0
    left = torch.load(ROOT/MODELS[3][3], weights_only=True)
    right = torch.load(ROOT/'experiments/recovery_comparison/results/run01/control/final.pt', weights_only=True)
    assert left['online'].keys() == right['online'].keys()
    assert all(torch.equal(left['online'][k], right['online'][k]) for k in left['online'])
    assert canonical(left['environment_config']) == canonical(right['environment_config'])
    # Check previously tracked text records BEFORE creating this protocol.
    tracked = subprocess.check_output(['git','ls-files','-z'], cwd=ROOT).decode().split('\0')
    matches = []; scanned = 0
    pattern = re.compile(r'(?<!\d)20261118\d{2}(?!\d)')
    for rel in tracked:
        file = ROOT/rel
        if file.suffix not in ('.py','.json','.md','.yaml','.yml','.txt') or HERE in file.parents: continue
        if not file.is_file(): continue
        scanned += 1
        found = {int(x) for x in pattern.findall(file.read_text(errors='replace'))} & set(NEW)
        if found: matches.append(dict(path=rel,seeds=sorted(found)))
    assert not matches, matches
    sources = [HERE/'catalogue.py', Path(__file__), Path(coordinates.__file__), Path(imitation.__file__),
               ROOT/'experiments/online_pilot/run.py'] + list((ROOT/'src').rglob('*.py'))
    files = sources + [ROOT/row[3] for row in MODELS] + [ROOT/'experiments/recovery_comparison/results/run01/control/final.pt']
    protocol = dict(created_unix=time.time(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        purpose='Existing frozen endpoints compared on one new paired bank. No training, tuning, budget changes, seed replacement, or checkpoint re-selection.',
        models=[dict(id=i,family=f,label=l,path=p) for i,f,l,p in MODELS],old_seeds=OLD,new_seeds=NEW,
        environment=configs[0],excluded=EXCLUDED,prior_text_records_scanned=scanned,prior_new_seed_matches=matches,
        sample_size='100 new scenes is a predeclared practical evaluation budget, not a course requirement or a statistical power calculation.',
        metrics='Success, damage, timeout and errors on all 133 seeds; success-only steps/travel clearly conditional; paired common-success efficiency separately. Retain every failed trajectory.',
        limitations='Same deterministic generator and fixed capacity, no new physical/noise distribution. Unequal historical data/training budgets: cross-family comparisons are descriptive, not single-factor causality. No claim that post-test best seed is independently validated.',
        original_model='models/supervision_034_targeted.pt; not the earlier model associated with historical13/100.',
        selected_before_new_test='rebuilt_joint6000 and rebuilt_staged3000 are from prior validation selection; other variants use frozen final endpoints. Pure imitation kept separate.',
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,threads_per_worker=1),
        hashes={str(file.relative_to(ROOT)):sha(file) for file in files})
    save(OUT/'protocol.json',protocol)
    return protocol


def initial_record(env, observation, reset_info):
    state = dict(scene=asdict(env.scene),rng=env.np_random.bit_generator.state,observation=observation,reset_info=reset_info)
    return state, hashlib.sha256(canonical(state).encode()).hexdigest()


def rollout(agent, env, seed):
    obs, reset_info = env.reset(seed=seed)
    initial, fingerprint = initial_record(env, obs, reset_info)
    trace = []; total = 0.; blocked = 0
    for step in range(env.max_steps):
        action = int(agent.act(obs))
        assert env.action_space.contains(action)
        obs, reward, term, trunc, info = env.step(action)
        total += reward; blocked += int(info['collision_blocked'])
        trace.append(dict(step=step+1,action=action,reward=reward,success=info['success'],damage=info['damage'],
            distance=info['distance_to_target'],position=env.scene.end_effector.copy(),
            travel=info['travel_distance'],deformation=info['max_normalized_deformation'],
            peak_deformation=info['peak_normalized_deformation'],blocked=bool(info['collision_blocked']),
            contact_steps=info['contact_steps'],terminated=bool(term),truncated=bool(trunc)))
        if term or trunc: break
    success, damage = bool(info['success']), bool(info['damage'])
    row = dict(seed=seed,success=success,damage=damage,timeout=not success and not damage,
        error=None,steps=len(trace),raw_return=total,blocked=blocked,travel=info['travel_distance'],
        peak_deformation=info['peak_normalized_deformation'],contact_steps=info['contact_steps'],
        initial_sha256=fingerprint)
    assert not (success and damage)
    if row['timeout']: assert len(trace) == env.max_steps
    return dict(metrics=row,trace=trace), initial


def init_worker():
    global POLICIES, CONFIG, CACHE, PROTOCOL_SHA
    runtime(); protocol = verify(); PROTOCOL_SHA = sha(OUT/'protocol.json')
    CONFIG = protocol['environment']; CACHE = CertifiedSceneCache(2); POLICIES = {}
    for name, _, _, path in MODELS:
        POLICIES[name], config = load_policy(ROOT/path)
        assert canonical(config) == canonical(CONFIG)


def check_seed(seed):
    folder=OUT/'episodes'/str(seed); stamp=read(folder/'complete.json')
    assert stamp['protocol_sha256'] == sha(OUT/'protocol.json')
    for name,digest in stamp['files'].items(): assert sha(folder/name)==digest
    return stamp


def run_seed(seed):
    folder = OUT/'episodes'/str(seed)
    if (folder/'complete.json').exists(): return check_seed(seed)
    started=time.perf_counter(); first=None; metrics={}
    for name, _, _, _ in MODELS:
        env=HarvestingEnv(CONFIG,scene_cache=CACHE)
        try:
            record,initial=rollout(POLICIES[name],env,seed)
        except Exception as exc:
            # Do not silently replace/drop a difficult scene or missing outcome.
            save(folder/f'{name}.error.json',dict(seed=seed,model=name,type=type(exc).__name__,message=str(exc)))
            raise
        finally: env.close()
        fingerprint=record['metrics']['initial_sha256']
        if first is None:
            first=fingerprint; save(folder/'initial.json',initial)
        assert fingerprint==first, (seed,name,'different initial scene')
        # Hash is already frozen in protocol; only paths belonging to this seed are written.
        record.update(model=name,model_sha256=sha(ROOT/next(x[3] for x in MODELS if x[0]==name)),protocol_sha256=PROTOCOL_SHA)
        save(folder/f'{name}.json',record); metrics[name]=record['metrics']
    names=['initial.json']+[f'{x[0]}.json' for x in MODELS]
    stamp=dict(seed=seed,protocol_sha256=PROTOCOL_SHA,initial_sha256=first,
        files={name:sha(folder/name) for name in names},seconds=time.perf_counter()-started)
    save(folder/'complete.json',stamp)
    return stamp


def run(workers):
    verify(); seeds=OLD+NEW
    completed=[s for s in seeds if (OUT/'episodes'/str(s)/'complete.json').exists()]
    for seed in completed: check_seed(seed)
    pending=[s for s in seeds if s not in completed]
    save(OUT/'status.json',dict(stage='running',completed=len(completed),total=len(seeds),workers=workers))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        jobs={pool.submit(run_seed,s):s for s in pending}
        for future in as_completed(jobs):
            stamp=future.result(); completed.append(stamp['seed'])
            save(OUT/'status.json',dict(stage='running',completed=len(completed),total=len(seeds),workers=workers,last_seed=stamp['seed']))
            print('COMPLETE SCENES',len(completed),'/',len(seeds),'episodes',24*len(completed),'/',24*len(seeds),flush=True)
    verify(); save(OUT/'status.json',dict(stage='completed',scenes=len(seeds),episodes=len(seeds)*len(MODELS)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['freeze','run'],required=True)
    parser.add_argument('--workers',type=int,default=8); args=parser.parse_args()
    if not 1<=args.workers<=8: parser.error('Use one to eight workers')
    if args.phase=='freeze': runtime(); freeze(); print('FROZEN 24 models,33 old +100 new scenes')
    else: run(args.workers)
