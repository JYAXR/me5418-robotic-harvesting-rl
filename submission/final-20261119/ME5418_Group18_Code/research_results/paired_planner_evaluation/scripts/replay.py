"""Replay stored actions using the actual environment, with no planner/transition cache."""
import json,hashlib,argparse
from pathlib import Path
import yaml
from run_episode import root,torch
from me5418_harvesting import HarvestingEnv

def replay(seed,files):
    cfg=yaml.safe_load((root/'configs/environment_multiroute.yaml').read_text())
    for f in files:
        r=json.loads(f.read_text())
        env=HarvestingEnv(cfg);obs,info=env.reset(seed=seed)
        assert hashlib.sha256(obs.tobytes()).hexdigest()==r['initial_observation_sha256']
        for row in r['trace']:
            _,reward,term,trunc,info=env.step(row['action'])
            assert reward==row['reward'] and row['terminal']==bool(term or trunc)
            assert env.scene.end_effector.tolist()==row['xy']
            assert env.total_travel==row['travel'] and env.peak_deformation==row['peak']
            assert env.contact_steps==row['contact_steps']
        assert all(r[k]==bool(info[n]) for k,n in [('damage','damage'),('success','success')])
        assert r['timeout']==bool(info['budget_exhausted'] and not r['success'] and not r['damage'])
        print(f'PASS {f.name} {len(r["trace"])} actual transitions',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seed',type=int);p.add_argument('--episodes-dir',type=Path,default=Path(__file__).resolve().parents[1]/'data/episodes')
    a=p.parse_args();files=sorted(a.episodes_dir.glob(f'{a.seed}-*.json'));assert files
    replay(a.seed,files)
