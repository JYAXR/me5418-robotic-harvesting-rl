from pathlib import Path
import sys,itertools
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_episode as pilot
import numpy as np
from copy import deepcopy
from functools import partial
import json
pilot.torch.set_num_threads(1)
seed=5420000
agent,base,_=pilot.CompactGuidedAgent.load(pilot.root/'models/compact_guided_generalization.pt')
base.reset(seed=seed)
original=pilot.RecedingHorizonPlanner.plan
def exact(e,d):
    best=-float('inf')
    for a in range(8):
        c=deepcopy(e)
        _,r,x,y,_=c.step(a)
        v=r+(exact(c,d-1) if d>1 and not(x or y) else 0)
        best=max(best,v)
    return best
checks=0
def scrutinize(planner,state,transition):
    global checks
    plan=original(planner,state,transition)
    k,n=state
    if n==3 and (checks==0 or (checks==1 and float(np.frombuffer(k[0],np.float32)[0])>.23)):
        env=deepcopy(base); env.scene.end_effector=np.frombuffer(k[0],np.float32).copy()
        for b,data,flag in zip(env.scene.branches,k[1],k[2]):
            b.current_nodes=np.frombuffer(data,np.float32).reshape(b.rest_nodes.shape).copy();b.in_contact=flag
        vals={}
        for a in range(8):
            c=deepcopy(env);_,r,x,y,_=c.step(a)
            vals[a]=r+(exact(c,2) if not(x or y) else 0)
        assert abs(plan.predicted_return-max(vals.values()))<1e-10 and abs(vals[plan.action]-max(vals.values()))<1e-10
        print('independent check',checks,'xy',env.scene.end_effector,'best',plan.action,'score',plan.predicted_return,'all root values',vals,flush=True)
        checks+=1
    return plan
pilot.RecedingHorizonPlanner.plan=scrutinize
row=pilot.run(seed,3)
assert checks==2
old=json.loads((Path(__file__).resolve().parents[1]/'data/episodes/5420000-3.json').read_text())
assert all(old[k]==row[k] for k in ['steps','success','damage','timeout','travel','peak','reward','contacts'])
print('PASS: independent exhaustive returns at both states and complete episode outcome agree.')
