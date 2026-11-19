"""Render first layout in the protocol, using original geometry and recorded positions."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from run_episode import root
from me5418_harvesting import HarvestingEnv

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',required=True,type=Path)
parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data')
args=parser.parse_args()
seed=5420000
env=HarvestingEnv(yaml.safe_load((root/'configs/environment_multiroute.yaml').read_text()))
env.reset(seed=seed)
fig,axes=plt.subplots(1,2,figsize=(9,4.4),constrained_layout=True)
for ax,h,title,color in zip(axes,(0,7),('Greedy compact DQN','H7 planner (beam 32)'),('#007489','#b34e24')):
    r=json.loads((args.data/'episodes'/f'{seed}-{h}.json').read_text())
    xy=np.asarray([env.scene.end_effector.tolist()]+[p['xy'] for p in r['trace']])
    for b in env.scene.branches:
        nodes=b.rest_nodes
        ax.plot(nodes[:,0],nodes[:,1],color='#626e63',lw=2,zorder=1)
    ax.add_patch(plt.Circle(env.scene.target,env.target_radius,color='#edbf53',alpha=.7,label='goal region'))
    ax.plot(xy[:,0],xy[:,1],color=color,lw=1.8,marker='.',ms=2.5,zorder=3,label='executed path')
    ax.scatter(*xy[0],marker='s',s=35,color='#15263c',zorder=4,label='start')
    ax.scatter(*xy[-1],marker='x',s=48,color=color,zorder=5,label='last position')
    ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel='Normalized x',ylabel='Normalized y')
    ax.set_title(f"{title}\n{'Safe success' if r['success'] else 'Budget exhausted'} · {r['steps']} actions",fontsize=11)
axes[0].legend(loc='lower left',fontsize=7,framealpha=.92)
fig.suptitle('Same initial branches, different routes | seed 5420000',fontsize=13)
fig.savefig(args.out,dpi=170)
