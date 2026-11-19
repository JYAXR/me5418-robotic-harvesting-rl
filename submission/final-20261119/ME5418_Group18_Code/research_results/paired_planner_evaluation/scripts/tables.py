"""Validate every complete raw trace; regenerate CSVs without training or PyTorch."""
import argparse,csv,hashlib,json,math,statistics as st
from pathlib import Path

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    data=args.data; seeds=set(range(5420000,5420010)); main_methods=['DQN','H1','H3','H5','H7']
    methods=main_methods+['H3-B32']; a={v['seed']:v for v in json.loads((data/'astar.json').read_text())}
    official=json.loads((data/'dqn_133.json').read_text())
    assert official['models']['compact']['sha256']=='75d7d7ca728b1c85c8ba6fa226076f85f4f72bdce3cfe4c6c0177946e5fe51e6'
    assert official['environment_config']['execution_noise_fraction']==0
    d={v['seed']:v for v in official['records'] if v['group']=='range'}
    assert len(d)==100 and len(a)==10 and set(a)==seeds
    episodes={};init={};rows=[]
    for path in sorted((data/'episodes').glob('*.json')):
        row=json.loads(path.read_text());s=row['seed'];m=row['method']; tr=row.pop('trace')
        assert s in seeds and m in methods and (s,m) not in episodes
        assert row['beam']==(32 if m in ('H5','H7','H3-B32') else None)
        assert len(tr)==row['steps']<=200 and [p['step'] for p in tr]==list(range(1,len(tr)+1))
        assert all(0<=p['action']<8 for p in tr)
        assert tr[-1]['terminal'] and not any(p['terminal'] for p in tr[:-1])
        assert sum(row[k] for k in ['success','damage','timeout'])==1
        assert not row['timeout'] or len(tr)==200
        assert row['damage']==(row['peak']>1)
        cumulative=0.0
        for p in tr: cumulative+=p['reward']
        assert cumulative==row['reward']
        for metric,field in [('travel','travel'),('peak','peak'),('contacts','contact_steps'),('model_calls','model_calls')]:
            assert row[metric]==tr[-1][field]
        assert math.isclose(row['wall_ms'],1000*sum(p['wall_s'] for p in tr)/len(tr))
        assert row['initial_observation_sha256']==init.setdefault(s,row['initial_observation_sha256'])
        assert row.pop('reference')=={k:v for k,v in a[s].items() if k!='seed'}
        if m=='DQN':
            for key,k2 in [('steps','steps'),('travel','travel_distance'),('success','success'),('contacts','contact_steps'),('peak','peak_deformation'),('reward','return'),('damage','damage'),('timeout','timeout')]:
                assert row[key]==d[s][k2]
        episodes[s,m]=row;rows.append(row)
    write(args.output/'episodes.csv',rows)
    coverage=[dict(seed=s,**{m:'complete' if (s,m) in episodes else 'not_available' for m in methods}) for s in sorted(seeds)]
    write(args.output/'coverage.csv',coverage)
    common={s for s in seeds if all((s,m) in episodes for m in main_methods)}
    def summarize(ms,ss):
        results=[]
        for m in ms:
            vs=[episodes[s,m] for s in sorted(ss) if (s,m) in episodes];n=sum(v['steps'] for v in vs)
            if not vs: continue
            result=dict(method=m,episodes=len(vs),success=sum(v['success'] for v in vs),damage=sum(v['damage'] for v in vs),timeout=sum(v['timeout'] for v in vs))
            for k in ('steps','travel','peak','reward','contacts'):
                result['mean_'+k]=st.mean(v[k] for v in vs)
            for k in ('cpu_ms','wall_ms'):
                result['per_action_'+k]=sum(v['steps']*v[k] for v in vs)/n
            for k in ('expansions','model_calls','plan_cache_hits'):
                result['per_action_'+k]=sum(v[k] for v in vs)/n
            results.append(result)
        return results
    write(args.output/'primary_paired.csv',summarize(main_methods,common))
    write(args.output/'all_completed.csv',summarize(methods,seeds))
    beam_common={s for s in seeds if all((s,m) in episodes for m in ('DQN','H1','H3-B32','H5','H7'))}
    write(args.output/'fixed_beam_paired.csv',summarize(['DQN','H1','H3-B32','H5','H7'],beam_common))
    pair=[]
    for s in sorted(seeds):
        v=episodes[s,'DQN'];w=a[s];assert w['success'] and w['contact_steps']==0
        pair.append(dict(seed=s,both_success=v['success'] and w['success'],dqn_travel=v['travel'],astar_travel=w['travel_distance'],dqn_minus_astar=v['travel']-w['travel_distance'],dqn_steps=v['steps'],astar_steps=w['steps'],dqn_contact_steps=v['contacts'],astar_contact_steps=w['contact_steps']))
    write(args.output/'astar_pairs.csv',pair)
    print(f'PASS: {len(episodes)} complete method-layout rollouts; primary matched seeds {sorted(common)}; fixed-beam matched seeds {sorted(beam_common)}.')
if __name__=='__main__':
    main()
