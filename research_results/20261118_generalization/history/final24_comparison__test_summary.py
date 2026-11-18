"""Check outcome totals and paired efficiency use the same successful scenes."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).parent;sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('comparison_summary',HERE/'audit_and_summarize.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def row(seed,success,damage=False,steps=200):
    return dict(seed=seed,success=success,damage=damage,timeout=not success and not damage,
                error=None,steps=steps,travel=steps/10,peak_deformation=1.1 if damage else .1,raw_return=0.)


def test_summary_no_success_has_no_conditional_average():
    result=a.summary([row(1,False),row(2,False,True,10)])
    assert result['success']==0 and result['damage']==result['timeout']==1
    assert result['successful_mean_steps'] is None and result['successful_mean_travel'] is None


def test_paired_counts_and_common_success_efficiency():
    left=[row(1,True,steps=20),row(2,False),row(3,True,steps=40)]
    right=[row(1,True,steps=30),row(2,True,steps=1),row(3,False)]
    result=a.paired(left,right)
    assert result['gained']==[2] and result['lost']==[3] and result['net_success']==0
    assert result['common_success']==1 and result['mean_step_difference_right_minus_left']==10
    assert result['mean_travel_difference_right_minus_left']==1


def test_summary_partition():
    result=a.summary([row(1,True,steps=20),row(2,False),row(3,False,True,5)])
    assert result['n']==result['success']+result['damage']+result['timeout']+result['errors']==3
    assert result['successful_mean_steps']==20 and result['total_steps']==225
