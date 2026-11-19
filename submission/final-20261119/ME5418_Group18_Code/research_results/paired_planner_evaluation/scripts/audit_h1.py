"""Independent complete H1 episode: no reuse of any transition or decision."""
import json
from copy import deepcopy
from pathlib import Path
from run_episode import CompactGuidedAgent, root, torch

torch.set_num_threads(1)
_,env,_=CompactGuidedAgent.load(root/'models/compact_guided_generalization.pt')
env.reset(seed=5420000)
gold=json.loads((Path(__file__).resolve().parents[1]/'data/episodes/5420000-1.json').read_text())
for record in gold['trace']:
    values=[deepcopy(env).step(action)[1] for action in range(8)]
    action=max(range(8),key=values.__getitem__)
    _,reward,done,trunc,info=env.step(action)
    assert action==record['action'] and reward==record['reward']
    assert env.scene.end_effector.tolist()==record['xy']
    assert env.total_travel==record['travel'] and env.peak_deformation==record['peak']
    assert bool(done or trunc)==record['terminal']
assert env.step_count==200 and info['budget_exhausted'] and not info['success']
print('PASS seed 5420000 H1: all 200 actions, rewards, positions, path lengths, peaks and termination flags match.')
