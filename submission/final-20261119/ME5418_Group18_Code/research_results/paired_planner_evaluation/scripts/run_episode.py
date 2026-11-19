"""Clean, same-layout pilot. Run with the compact branch root as working directory."""
from pathlib import Path
from copy import deepcopy
import json, sys, time, argparse, hashlib
from functools import lru_cache
import numpy as np
root=Path.cwd()
sys.path[:0]=[str(root/'src'),str(root/'scripts'),str(root/'research_results/20261118_generalization')]
import torch
from guided_coordinate_policy import CompactGuidedAgent
from me5418_harvesting.planners.receding_horizon import RecedingHorizonPlanner

def key(env):
    return (env.scene.end_effector.tobytes(),tuple(b.current_nodes.tobytes() for b in env.scene.branches),
            tuple(b.in_contact for b in env.scene.branches))

def run(seed,h,progress=None,beam_override=None):
    agent,env,_=CompactGuidedAgent.load(root/'models/compact_guided_generalization.pt')
    obs,reset=env.reset(seed=seed)
    initial=hashlib.sha256(obs.tobytes()).hexdigest()
    assert env.noise_fraction==0 and env.max_steps==200
    sim=deepcopy(env)
    # Deterministic transition cache, local to one seed and method. Historical peak
    # below threshold has no effect on rewards/physics; budget termination added below.
    @lru_cache(maxsize=24000)
    def physical(k,a):
        p,shapes,flags=k
        sim.scene.end_effector=np.frombuffer(p,np.float32).copy()
        for b,data,flag in zip(sim.scene.branches,shapes,flags):
            b.current_nodes=np.frombuffer(data,np.float32).reshape(b.rest_nodes.shape).copy()
            b.in_contact=flag
        sim._initialize_episode()
        _,r,done,tr,_=sim.step(a)
        return key(sim),r,bool(done or tr)
    def transition(state,a):
        k,n=state
        dest,r,done=physical(k,a)
        return (dest,n-1),r,done or n==1
    beam=beam_override if beam_override is not None else (32 if h>3 else None)
    planner=RecedingHorizonPlanner(depth=h,beam_width=beam) if h else None
    @lru_cache(maxsize=256)
    def decision(k,n):
        return planner.plan((k,n),transition)
    times=[]; cpu=[]; expanded=0; ret=0; trace=[]
    for t in range(env.max_steps):
        start=time.perf_counter(); tc=time.process_time()
        if h:
            k=key(env)
            misses=decision.cache_info().misses
            result=decision(k,min(h,env.max_steps-t))
            if misses!=decision.cache_info().misses: expanded+=result.model_evaluations
            a=result.action
        else:
            a=agent.act(obs,epsilon=0)
        times.append(time.perf_counter()-start); cpu.append(time.process_time()-tc)
        obs,r,done,tr,info=env.step(a); ret+=r
        if h:
            dest,rr,terminal=physical(k,a)
            assert dest==key(env) and rr==r and (terminal or t+1==env.max_steps)==(done or tr)
        trace.append(dict(step=t+1,action=int(a),xy=env.scene.end_effector.tolist(),reward=float(r),
                     travel=float(env.total_travel),peak=float(env.peak_deformation),contact_steps=int(env.contact_steps),
                     wall_s=times[-1],cpu_s=cpu[-1],terminal=bool(done or tr),model_calls=physical.cache_info().misses))
        if progress is not None and (t%10==0 or done or tr):
            print(seed,h,t+1,physical.cache_info().misses,flush=True)
        if done or tr: break
    row=dict(seed=seed,method='DQN' if not h else ('H3-B32' if h==3 and beam==32 else 'H'+str(h)),beam=beam,
         steps=t+1,success=bool(info['success']),damage=bool(info['damage']),timeout=bool(info['budget_exhausted'] and not info['success'] and not info['damage']),
         travel=env.total_travel,peak=float(env.peak_deformation),reward=ret,contacts=env.contact_steps,
         wall_ms=1000*sum(times)/(t+1),cpu_ms=1000*sum(cpu)/(t+1),expansions=expanded,
         model_calls=physical.cache_info().misses,plan_cache_hits=decision.cache_info().hits)
    row.update(initial_observation_sha256=initial,reference=reset['reference_route'],trace=trace)
    return row
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--depth',type=int,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--beam',type=int,choices=[32])
    args=ap.parse_args();torch.set_num_threads(1)
    if args.beam and args.depth!=3: ap.error('--beam is for the supplementary H3 test')
    if args.out.exists(): ap.error('Output already exists; use a new path.')
    with torch.inference_mode(): row=run(args.seed,args.depth,True,args.beam)
    args.out.write_text(json.dumps(row,indent=2))
