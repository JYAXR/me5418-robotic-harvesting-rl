"""Verify reconstruction accounting, frozen selection, trajectories and reloads."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

spec=importlib.util.spec_from_file_location('rebuild_audit_run',Path(__file__).with_name('run.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
c=r.c


def records(folder,seeds):
    rows=c.read(folder/'metrics.json');assert [z['seed'] for z in rows]==seeds
    for row in rows:
        assert 'reset_error' not in row
        trace=c.read(folder/f"trace_{row['seed']}.json")
        assert len(trace)==row['steps']
        assert [x['step'] for x in trace]==list(range(1,len(trace)+1))
        assert abs(sum(x['reward'] for x in trace)-row['raw_return'])<1e-9
        assert sum(x['blocked'] for x in trace)==row['blocked']
        assert not (row['success'] and row['damage'])
        if not row['success'] and not row['damage']:assert row['steps']==200
    return rows


def main():
    c.runtime();out=c.HERE/'results/run01';r.verify(out)
    assert c.read(out/'status.json')['stage']=='completed'
    protocol=c.read(out/'protocol.json');prep=out/'preparation'
    completed=sorted(prep.glob('checkpoint_*'))
    assert len(completed)==52
    phase_counts={name:0 for name,_ in c.PHASES}
    for i,folder in enumerate(completed,1):
        r.check_snapshot(folder)
        progress=c.read(folder/'progress.json');assert progress['total_updates']==1000*i
        logs=c.read(folder/'updates.json');assert len(logs)==1000
        assert c.p.sha(folder/'updates.json')==c.read(folder/'loss_hash.json')['sha256']
        phase=progress['phase'];phase_counts[phase]+=len(logs)
        assert [z['update'] for z in logs]==list(range(progress['phase_update']-999,progress['phase_update']+1))
        state=torch.load(folder/'state.pt',weights_only=True)
        assert state['online_steps']==0
        expected_clock=progress['phase_update']+(10000 if phase=='targeted' else 0)
        expected_adam=progress['phase_update']+({'td':20000,'targeted':30000}.get(phase,0))
        assert state['update_count']==expected_clock
        assert all(int(z['step'])==expected_adam for z in state['optimizer']['state'].values())
        for z in logs:
            assert np.isfinite([v for v in z.values() if not isinstance(v,bool)]).all()
            if phase in ('supervised','calibration'):assert z['td']==0
            if phase!='calibration':assert z['calibration']==0
    assert phase_counts==dict(c.PHASES)
    done=c.read(prep/'done.json');assert c.p.sha(prep/'prepared.pt')==done['model_sha256']
    assert c.p.sha(prep/'prepared_state.pt')==done['state_sha256']
    for name,digest in c.read(out/'data/frozen.json').items():assert c.p.sha(out/'data'/name)==digest
    raw=c.arrays(out/'data/new_success.npz')
    assert len(raw['actions'])==2785 and set(raw['seed'])==set(c.NEW) and not raw['stagnant'].any()
    stops=np.flatnonzero(raw['terminated'])+1;assert len(stops)==67 and stops[-1]==2785
    start=0
    for stop in stops:
        assert np.array_equal(raw['next_observations'][start:stop-1],raw['observations'][start+1:stop])
        value=0.
        for i in range(stop-1,start-1,-1):
            value=float(raw['rewards'][i])+.99*value
            assert abs(value-float(raw['returns'][i]))<.0003
        assert raw['rewards'][stop-1]>0;start=stop
    assert len(set(protocol['old_train']+c.NEW+c.DEV+c.TEST))==148
    base=records(out/'baseline_validation',c.DEV);selections={}
    for arm in c.ARMS:
        folder=out/arm;selection=c.read(folder/'selection.json')
        for n in range(1000,6001,1000):
            dest=folder/f'checkpoint_{n:06d}';r.check_snapshot(dest)
            for name,digest in c.read(dest/'files.json').items():assert c.p.sha(dest/name)==digest
            state=torch.load(dest/'state.pt',weights_only=True)
            model=torch.load(dest/'model.pt',weights_only=True)
            c.m.equal(state['online'],model['online'])
            assert state['online_steps']==0 and state['update_count']==12000+n
            assert all(int(z['step'])==32000+n for z in state['optimizer']['state'].values())
            logs=c.read(dest/'updates.json');assert len(logs)==n
            assert [z['update'] for z in logs]==list(range(1,n+1))
            assert [z['phase'] for z in logs]==[c.phase_for(arm,i) for i in range(1,n+1)]
            for z in logs:
                assert np.isfinite([v for k,v in z.items() if k!='phase']).all()
                assert abs(z['loss']-(z['margin']+z['recovery']+z['mc']+.5*z['td']))<1e-5
            if n in c.BOUNDARIES:
                rows=records(dest/'validation',c.DEV)
                item=next(z for z in selection['candidates'] if z['label']==dest.name)
                assert tuple(item['score'])==c.p.score(rows)
        assert tuple(selection['candidates'][0]['score'])==c.p.score(base)
        assert selection['selected']==max(selection['candidates'],key=lambda z:tuple(z['score']))
        selections[arm]=selection['selected']
    lock=c.read(out/'selection_lock.json');summary=c.read(out/'summary.json')
    assert lock['validation_recommendation']==max(lock['models'],key=lambda name:tuple(lock['models'][name]['score']))
    cache=c.p.CertifiedSceneCache(8);reloads=[]
    for name,item in lock['models'].items():
        assert c.p.sha(c.ROOT/item['path'])==item['sha256']
        if name in selections:
            for k,v in selections[name].items():assert item[k]==v
        folder=out/'evaluation'/name
        assert summary[name]==records(folder,protocol['old_train']+c.TEST)
        assert all((folder/f'trace_{s}.json').stat().st_mtime>=lock['locked_unix'] for s in c.TEST)
        if name!='prepared':
            dest=out/'reload_checks'/name
            rows=r.evaluate(c.ROOT/item['path'],protocol['environment'],[c.TEST[0]],cache,dest)
            assert rows==[next(z for z in summary[name] if z['seed']==c.TEST[0])]
            assert c.read(dest/f'trace_{c.TEST[0]}.json')==c.read(folder/f'trace_{c.TEST[0]}.json')
            reloads.append(dict(name=name,seed=c.TEST[0],exact_trace_match=True))
    comparison=c.read(out/'historical_hash_comparison.json')
    for name,row in comparison.items():
        assert row['actual']==lock['models'][name]['sha256']
        assert row['exact_file_match']==(row['actual']==row['historical'])
    r.verify(out)
    c.save(out/'audit.json',dict(passed=True,preparation_updates=52000,followup_updates=12000,
        total_new_gradient_updates=64000,final_rollouts=195,reload_checks=reloads,
        historical_model_hash_comparison=comparison,
        scope='Checks source/input/checkpoint hashes, data continuity/returns, clocks, recorded objectives, selection and trajectories; reloads two episodes. Does not independently rerun all training.',
        evaluation_role='Reproduction on previously used seeds, not new independent evidence'))
    print('AUDIT PASSED',comparison,flush=True)


if __name__=='__main__':main()
