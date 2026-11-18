"""Offline checks of all recorded online transitions, clocks and paired outcomes."""
import numpy as np
import torch
import study as s
import evaluate as e


def main():
    s.verify();e.verify();counts={};summaries=s.b.read(e.OUT/'summary.json')['groups']
    random=np.random.default_rng(s.SEED+9);coins=random.random(s.TOTAL);actions=random.integers(8,size=s.TOTAL)
    expected_random=coins<np.array([s.epsilon(i) for i in range(s.TOTAL)])
    source=torch.load(s.SOURCE,weights_only=True)
    for arm in s.ARMS:
        folder=s.OUT/arm;total=0;damage=stagnant=0
        for slot,seed in enumerate(s.schedule()):
            path=folder/f'transitions_{slot:03d}.npz';rows=s.c.arrays(path)
            n=len(rows['action']);assert n==200
            assert np.all(rows['seed']==seed) and np.all(rows['slot']==slot)
            assert rows['obs'].shape==(200,341) and rows['next'].shape==(200,341)
            assert np.isfinite(rows['obs']).all() and np.isfinite(rows['next']).all()
            valid=~(rows['done'][:-1]|rows['truncated'][:-1])
            np.testing.assert_array_equal(rows['next'][:-1][valid],rows['obs'][1:][valid])
            actual=np.array([s.reward_for(r,z,d,arm) for r,z,d in zip(rows['raw_reward'],rows['stagnant'],rows['damage'])])
            np.testing.assert_array_equal(actual,rows['training_reward'])
            assert np.all(rows['done']==(rows['success']|rows['damage']|rows['budget']))
            assert not np.any(rows['truncated'])
            indices=slice(slot*200,(slot+1)*200);mask=expected_random[indices]
            np.testing.assert_array_equal(rows['random_action'],mask)
            np.testing.assert_array_equal(rows['action'][mask],actions[indices][mask])
            total+=n;damage+=int(rows['damage'].sum());stagnant+=int(rows['stagnant'].sum())
        losses=s.b.read(folder/'updates.json')
        assert [x['step'] for x in losses]==list(range(64,s.TOTAL+1,4))
        assert all(np.isfinite([x['loss'],x['td'],x['margin'],x['recovery'],x['gradient_norm']]).all() for x in losses)
        last=folder/'checkpoint_013200';state=torch.load(last/'state.pt',weights_only=True)
        assert state['online_steps']==13200 and state['update_count']==15285
        assert all(int(v['step'])==35285 for v in state['optimizer']['state'].values())
        assert state['optimizer']['param_groups']==source['optimizer']['param_groups']
        endpoint=torch.load(folder/'final.pt',weights_only=True)
        assert all(torch.equal(v,endpoint['online'][k]) for k,v in state['online'].items())
        counts[arm]=dict(transitions=total,updates=len(losses),random_actions=int(expected_random.sum()),damage_transitions=damage,stagnant_transitions=stagnant)
    for condition,seeds,repeats,label in [('clean',s.OLD,[0],'old'),('clean',s.TEST,[0],'new_clean'),('noisy',s.TEST[:20],[0,1,2],'new_noisy')]:
        actual={name:dict(success=0,damage=0,timeout=0) for name in e.MODELS}
        for seed in seeds:
            for repeat in repeats:
                folder=e.OUT/condition/f'{seed}_{repeat}';stamp=s.b.read(folder/'complete.json')
                for name,digest in stamp['hashes'].items():assert s.b.sha(folder/name)==digest
                fingerprints=[]
                for name in e.MODELS:
                    r=s.b.read(folder/f'{name}.json');m=r['metrics'];fingerprints.append(m['initial_sha256'])
                    assert len(r['trace'])==m['steps'] and not m['error']
                    assert abs(sum(t['reward'] for t in r['trace'])-m['raw_return'])<1e-8
                    for outcome in actual[name]:actual[name][outcome]+=int(m[outcome])
                assert len(set(fingerprints))==1
        for name in e.MODELS:
            for outcome,value in actual[name].items():assert summaries[label][name][outcome]==value
    noise_pairing={}
    for name in e.MODELS:
        layout_rows=[]
        for seed in s.TEST[:20]:
            clean=s.b.read(e.OUT/'clean'/f'{seed}_0'/f'{name}.json')['metrics']
            noisy=[s.b.read(e.OUT/'noisy'/f'{seed}_{rep}'/f'{name}.json')['metrics'] for rep in range(3)]
            layout_rows.append(dict(seed=seed,clean_success=clean['success'],noisy_successes=sum(r['success'] for r in noisy),
                noisy_damage=sum(r['damage'] for r in noisy),noisy_timeout=sum(r['timeout'] for r in noisy)))
        noise_pairing[name]=dict(clean_success_20=sum(r['clean_success'] for r in layout_rows),
            noisy_success_60=sum(r['noisy_successes'] for r in layout_rows),
            rescued_at_least_once=sum(not r['clean_success'] and r['noisy_successes']>0 for r in layout_rows),
            lost_at_least_once=sum(r['clean_success'] and r['noisy_successes']<3 for r in layout_rows),
            all_three_noisy_success=sum(r['noisy_successes']==3 for r in layout_rows),layouts=layout_rows)
    s.b.save(s.OUT/'noise_pairing.json',noise_pairing)
    left=torch.load(s.OUT/'control/final.pt',weights_only=True)
    right=torch.load(s.OUT/'damage/final.pt',weights_only=True)
    weights_equal=all(torch.equal(v,right['online'][k]) for k,v in left['online'].items())
    damage_rows=[]
    for path in (s.OUT/'control').glob('transitions_*.npz'):
        data=s.c.arrays(path)
        for i in np.flatnonzero(data['damage']):damage_rows.append((data['obs'][i],int(data['action'][i]),float(data['raw_reward'][i])))
    agent,_=s.b.load_policy(s.OUT/'control/final.pt')
    equal_gradients=0
    if damage_rows:
        with torch.no_grad():
            q=agent.learner.online(torch.from_numpy(np.stack([r[0] for r in damage_rows]))).gather(1,torch.tensor([r[1] for r in damage_rows])[:,None]).squeeze(1)
            targets=torch.tensor([r[2] for r in damage_rows],dtype=torch.float32)
            equal_gradients=int(((q-targets).clamp(-1,1)==(q-(targets-50)).clamp(-1,1)).sum())
    s.b.save(s.OUT/'damage_penalty_diagnostic.json',dict(final_network_exact_equal=weights_equal,
        control_damage_transitions=len(damage_rows),final_network_huber_derivatives_equal_on_control_damage_rows=equal_gradients,
        interpretation='Post-hoc derivative check on final-network terminal damage predictions. Huber derivative is capped outside absolute TD error1. This does not independently replay every historical training gradient.'))
    s.b.save(s.OUT/'audit.json',dict(passed=True,training=counts,evaluation_episodes=965,paired_initial_states=True,
        frozen_inputs=True,trajectory_and_reward_accounting=True,original_optimizer_preserved_and_advanced=True,
        limitation='Offline audit plus ten fresh evaluation reloads. Does not independently replay every training optimizer step or every training physics transition.'))
    print('AUDIT PASSED',counts,flush=True)


if __name__=='__main__':main()
