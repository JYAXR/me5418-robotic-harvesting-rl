"""Regenerate results in a new directory. Never edit supplied evidence or retrain."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

scripts = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--workers', type=int, default=1)
args = p.parse_args()
if args.workers < 1 or not __debug__:
    p.error('Use positive workers and do not disable assertions.')
repo, out = args.repo.resolve(), args.out.resolve()
model = repo / 'models/compact_guided_generalization.pt'
assert hashlib.sha256(model.read_bytes()).hexdigest() == '75d7d7ca728b1c85c8ba6fa226076f85f4f72bdce3cfe4c6c0177946e5fe51e6'
for line in (scripts.parent/'audits/source-sha256.txt').read_text().splitlines():
    checksum, relpath = line.split(None,1)
    assert hashlib.sha256((repo/relpath).read_bytes()).hexdigest()==checksum, relpath
out.mkdir(parents=True, exist_ok=False)
data = out / 'data'
episodes = data / 'episodes'
episodes.mkdir(parents=True)
procenv = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(repo/'src') + os.pathsep + os.environ.get('PYTHONPATH',''))
def call(*cmd):
    subprocess.run([sys.executable, *map(str,cmd)], cwd=repo, env=procenv, check=True)
call(scripts/'reference.py', data/'astar.json')
call(repo/'scripts/evaluate_coordinate_models.py', '--model','compact=models/compact_guided_generalization.pt',
     '--seed',5420000,'--episodes',100,'--known-dataset','--workers',args.workers,'--output',data/'dqn_133.json')
def episode(item):
    seed,h,beam=item
    suffix = '3b32' if beam else str(h)
    call(scripts/'run_episode.py','--seed',seed,'--depth',h,'--out',episodes/f'{seed}-{suffix}.json',
         *(['--beam','32'] if beam else []))
jobs = [(s,h,b) for s in range(5420000,5420010) for h,b in [(0,False),(1,False),(3,False),(5,False),(7,False),(3,True)]]
with ThreadPoolExecutor(args.workers) as pool:
    list(pool.map(episode,jobs))
call(scripts/'tables.py','--data',data,'--output',out/'tables')
for s in range(5420000,5420010):
    call(scripts/'replay.py',s,'--episodes-dir',episodes)
