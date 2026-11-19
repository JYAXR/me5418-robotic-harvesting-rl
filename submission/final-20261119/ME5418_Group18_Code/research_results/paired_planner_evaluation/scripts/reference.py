"""Recover the reference feasibility witness on exactly the pilot layouts."""
import json,sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path.cwd()/'src'))
from me5418_harvesting import HarvestingEnv
env=HarvestingEnv(yaml.safe_load(Path('configs/environment_multiroute.yaml').read_text()))
out=[]
for seed in range(5420000,5420010):
    _,info=env.reset(seed=seed)
    out.append(dict(seed=seed,**info['reference_route']))
Path(sys.argv[1]).write_text(json.dumps(out,indent=2))
