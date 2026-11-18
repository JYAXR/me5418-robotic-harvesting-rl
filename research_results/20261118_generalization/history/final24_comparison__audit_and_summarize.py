"""Audit every paired rollout; produce complete data tables without hiding losers."""
import csv
import json
from pathlib import Path
import statistics

import numpy as np

import run as r


def mean(values): return statistics.mean(values) if values else None


def summary(rows):
    good=[x for x in rows if x['success']]
    return dict(n=len(rows),success=len(good),damage=sum(x['damage'] for x in rows),
        timeout=sum(x['timeout'] for x in rows),errors=sum(x['error'] is not None for x in rows),
        success_rate=len(good)/len(rows),successful_mean_steps=mean([x['steps'] for x in good]),
        successful_mean_travel=mean([x['travel'] for x in good]),
        all_mean_peak_deformation=mean([x['peak_deformation'] for x in rows]),
        all_mean_return=mean([x['raw_return'] for x in rows]),total_steps=sum(x['steps'] for x in rows))


def paired(left, right):
    assert [x['seed'] for x in left]==[x['seed'] for x in right]
    gained=[b['seed'] for a,b in zip(left,right) if not a['success'] and b['success']]
    lost=[b['seed'] for a,b in zip(left,right) if a['success'] and not b['success']]
    common=[(a,b) for a,b in zip(left,right) if a['success'] and b['success']]
    return dict(gained=gained,lost=lost,net_success=len(gained)-len(lost),common_success=len(common),
        mean_step_difference_right_minus_left=mean([b['steps']-a['steps'] for a,b in common]),
        mean_travel_difference_right_minus_left=mean([b['travel']-a['travel'] for a,b in common]))


PAIRS=[('pilot_control','pilot_penalty'),('lr_1e4','lr_3e5'),('lr_1e4','lr_1e5'),
       ('lr_1e4','stagnation'),('lr_1e4','recovery'),
       ('fresh_abs','fresh_abs_penalty'),('fresh_rel','fresh_rel_penalty'),
       ('fresh_abs','fresh_rel'),('fresh_abs_penalty','fresh_rel_penalty'),
       ('rebuilt_prepared','rebuilt_joint'),('rebuilt_prepared','rebuilt_staged'),('rebuilt_joint','rebuilt_staged'),
       ('bc33','bc97'),('bc97','bc_rel4k'),('bc_rel4k','bc_rel20k'),
       ('bc_no_mirror','bc_repeat_A'),('bc_no_relative','bc_repeat_A')]


def audit():
    protocol=r.verify()
    assert r.read(r.OUT/'status.json')['stage']=='completed'
    rows={name:[] for name,_,_,_ in r.MODELS}; new_geometry=set()
    for seed in r.OLD+r.NEW:
        folder=r.OUT/'episodes'/str(seed); stamp=r.check_seed(seed)
        initial=r.read(folder/'initial.json')
        assert r.hashlib.sha256(r.canonical(initial).encode()).hexdigest()==stamp['initial_sha256']
        assert initial['reset_info']['seed']==seed
        assert initial['reset_info']['reference_route']['success']
        if seed in r.NEW:
            new_geometry.add(r.hashlib.sha256(r.canonical(initial['scene']).encode()).hexdigest())
        for name,_,_,path in r.MODELS:
            assert (folder/f'{name}.json').stat().st_mtime>=protocol['created_unix']
            record=r.read(folder/f'{name}.json'); row=record['metrics']; trace=record['trace']
            assert record['model']==name and record['model_sha256']==protocol['hashes'][path]
            assert record['protocol_sha256']==r.sha(r.OUT/'protocol.json')
            assert row['seed']==seed and row['initial_sha256']==stamp['initial_sha256']
            assert row['error'] is None and sum((row['success'],row['damage'],row['timeout']))==1
            assert len(trace)==row['steps'] and 1<=row['steps']<=200
            assert [x['step'] for x in trace]==list(range(1,len(trace)+1))
            assert all(0<=x['action']<8 for x in trace)
            assert not any(x['terminated'] or x['truncated'] for x in trace[:-1])
            assert trace[-1]['terminated'] and not trace[-1]['truncated']
            assert abs(sum(x['reward'] for x in trace)-row['raw_return'])<1e-9
            assert sum(x['blocked'] for x in trace)==row['blocked']
            assert trace[-1]['travel']==row['travel']
            assert trace[-1]['peak_deformation']==row['peak_deformation']
            assert trace[-1]['contact_steps']==row['contact_steps']
            assert trace[-1]['success']==row['success'] and trace[-1]['damage']==row['damage']
            assert all(b['travel']>=a['travel'] and b['peak_deformation']>=a['peak_deformation'] for a,b in zip(trace,trace[1:]))
            if row['success']: assert trace[-1]['distance']<=protocol['environment']['target_radius'] and row['peak_deformation']<=1
            if row['damage']: assert row['peak_deformation']>1
            if row['timeout']: assert row['steps']==200
            assert np.isfinite([row[k] for k in ('steps','raw_return','travel','peak_deformation')]).all()
            rows[name].append(row)
    assert len(new_geometry)==100
    # Reload representative formats and rerun old/new cases outside worker cache.
    checks=[]
    for name in ('original034','fresh_rel_penalty','rebuilt_staged','bc_repeat_A'):
        path=next(z[3] for z in r.MODELS if z[0]==name)
        agent,config=r.load_policy(r.ROOT/path)
        for seed in (r.OLD[0],r.NEW[0]):
            env=r.HarvestingEnv(config)
            actual,initial=r.rollout(agent,env,seed);env.close()
            previous=r.read(r.OUT/'episodes'/str(seed)/f'{name}.json')
            assert r.canonical(actual['metrics'])==r.canonical(previous['metrics'])
            assert r.canonical(actual['trace'])==r.canonical(previous['trace'])
            r.save(r.OUT/'reload_checks'/f'{name}_{seed}.json',actual)
            checks.append(dict(model=name,seed=seed,exact_trajectory_match=True))
    r.verify()
    r.save(r.OUT/'audit.json',dict(passed=True,models=24,old_scenes=33,new_scenes=100,
        episodes=3192,initial_state_checks=3192,reload_checks=checks,
        scope='All records and frozen hashes checked; eight uncached reload rollouts repeated, not a second run of all3192.',
        no_training=True))
    return rows


def export(rows):
    summaries={}
    flat=[]
    for name,family,label,path in r.MODELS:
        summaries[name]=dict(family=family,label=label,path=path,old=summary(rows[name][:33]),new=summary(rows[name][33:]))
        flat.extend(dict(model=name,family=family,split='old33' if a['seed'] in r.OLD else 'new100',**a) for a in rows[name])
    r.save(r.OUT/'summary.json',summaries)
    r.save(r.OUT/'paired_vs_original034.json',{name:dict(old=paired(rows['original034'][:33],a[:33]),new=paired(rows['original034'][33:],a[33:])) for name,a in rows.items()})
    comparisons={f'{a}__{b}':dict(left=a,right=b,old=paired(rows[a][:33],rows[b][:33]),new=paired(rows[a][33:],rows[b][33:])) for a,b in PAIRS}
    r.save(r.OUT/'controlled_comparisons.json',comparisons)
    for filename,values in [('per_episode.csv',flat),('comparison.csv',[dict(model=name,family=z['family'],label=z['label'],old_success=z['old']['success'],
            new_success=z['new']['success'],new_damage=z['new']['damage'],new_timeout=z['new']['timeout'],
            successful_mean_steps=z['new']['successful_mean_steps'],successful_mean_travel=z['new']['successful_mean_travel'],
            all_mean_peak_deformation=z['new']['all_mean_peak_deformation']) for name,z in summaries.items()])]:
        with (r.OUT/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    def table(family):
        lines=['| 版本/变化 | 原33成功 | 新100成功 | 新100损伤 | 新100超时 | 成功场景平均步数 |',
               '|---|---:|---:|---:|---:|---:|']
        for name,_,label,_ in r.MODELS:
            a=summaries[name]
            if a['family']!=family:continue
            o,n=a['old'],a['new'];steps='—' if n['successful_mean_steps'] is None else f"{n['successful_mean_steps']:.1f}"
            lines.append(f"| {label} (`{name}`) | {o['success']}/33 | {n['success']}/100 | {n['damage']} | {n['timeout']} | {steps} |")
        return '\n'.join(lines)
    output='''# 统一实际效果：完整结果表

本次只评估现有模型，没有训练或调参。24个模型使用相同33个原训练场景与100个预先固定的新场景，
共3192次闭环运行，全部成功/失败均计入。100是实际评估预算，不是课程标准。
原模型为034版本，不是曾获得13/100的更早版本；不能把不同版本、不同考题的旧分数拼进本表。

## DQN及示范辅助DQN（15个）

'''+table('dqn')+'''

## 纯模仿学习（9个，不能作为DQN训练效果宣称）

'''+table('imitation')+'''

## 如何读表

- 原33成功：是否仍会做原来的训练场景；新100成功：相同生成器下未参与训练/既往选点的新布局表现。
- 损伤：变形超过安全限制；超时：200步内没有完成；成功必须在预算内到达目标且无损伤。
- 成功场景平均步数只统计各自成功的场景，集合可能不同，不能单凭这个平均数宣称某个模型更快。
- `controlled_comparisons.json`另列共同成功场景上的步数/距离差，以及新增/丢失成功的具体种子。
- 距离使用仿真坐标单位，不是实测米；归一化变形超过1触发模拟损伤终止。
- `comparison.csv`为简表，`per_episode.csv`为3192条逐回合数据，`episodes/`保存全部逐步轨迹与初始场景。

## 解释限制

历史训练预算、数据量和初始化不同，因此统一考题的排名是最终效果比较，不是所有改动的严格单变量归因。
应在原本匹配的学习率、奖励/恢复、坐标×惩罚四组、模仿消融内部解释对应变化。
`rebuilt_staged`是此前验证组选中的3000次追加监督检查点，后续校准/TD检查点未被选中，不能归功于后续TD。
`rebuilt_prepared`未逐字节恢复更早的准备模型；两者不能混用。
纯模仿A/B/C是3个训练初始化，同一批100场景不是300个独立场景，也不能事后只挑最高种子声称稳定达到其分数。

没有新增规划器深度对照，也没有测试新的执行噪声、枝条数量范围、植物物理模型或真实机器人。本表是冻结候选的一次统一有限测试，
不是证明某方法普遍优越。如果根据本表继续调参或选择新赢家，仍需另留数据作独立确认。

## 完整性

模型名单与哈希在测试前冻结；初始场景、随机数状态对所有模型一致；不替换失败场景。
最终核对3192次记录，并对4种代表策略各重载重跑1个旧场景和1个新场景，共8次，轨迹完全一致。
原`src/`、`configs/`、`models/`和`docs/`未修改，没有推送GitHub。
'''
    (r.HERE/'RESULTS.md').write_text(output)
    print('AUDIT PASSED; EXPORTED ALL24 MODELS /3192 EPISODES',flush=True)
    for name,a in summaries.items(): print(name,a['old']['success'],a['new']['success'],a['new']['damage'],a['new']['timeout'],flush=True)


if __name__=='__main__':
    r.runtime();export(audit())
