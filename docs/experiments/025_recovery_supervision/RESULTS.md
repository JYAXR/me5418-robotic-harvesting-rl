# Attempt 025: the recovery action is learned, but early instability remains

Completed on 15 November 2026. The [protocol](../025_recovery_supervision.md),
training implementation and tests were committed before execution in
`ee4806f`. This experiment adds one actual off-reference observation with a
`north_large` label. It improves sampled three-scene retention from 63/69 to
67/69 checkpoints and rescues all five targeted historical failures through
learned decisions. It does not achieve uninterrupted retention: TD 2004 and
2009 fail scene 10001 at an earlier decision, despite retaining recovery ability.

## What changed

The recovery input is the encoded observation immediately before step 25 on
the frozen TD-2031 failure trajectory, after executing its first 24 actions
on scene 10001. Every collected observation and transition is checked against
attempt 024. Its float32 SHA-256 is
`8e27f43ffd9ba9234be3a6c894dff2d62fae2c9a2efe4e5c8b75541cffdca8d8`.
The new action is `north_large` (4), established by the successful intervention;
it is not the original demonstration's time-index-25 action. The observation
is absent from the original 117 demonstration rows.

The treatment resumes the same complete TD-2000 snapshot as attempt 022 and
performs 8000 updates through absolute TD 10000. Its objective is:

```text
0.5 * mean_TD_Huber(original 117 transitions)
+ weighted_original_hinge_sum / 117
+ recovery_state_hinge(north_large, margin=0.8) / 117
```

Original row 55 (scene 10001 step 21) retains weight 2, and every other original
row retains weight 1. The original 117 TD transitions, TD targets/reduction,
learning rate 0.0001, gamma 0.99, target interval 200, clipping 10, Adam moments
and clocks remain unchanged at the comparison boundary. There is no MC term,
new pretraining, online data, fabricated recovery transition, or early stopping.
The denominator remains 117 so adding the sample does not also dilute existing
supervision. Subsequent weights and TD targets naturally evolve under the new
objective. Production defaults are unchanged.

## Autonomous full episodes

The historical control combines attempt 022's 50 checkpoints with attempt 023's
19 neighboring checkpoints. Those immutable results are reused, not counted as
new baseline simulation. Treatment uses exactly the same 69 checkpoints and
three seeds, with every action selected by the network from reset to termination.

| Matched measurement | Historical control | Recovery supervision |
| --- | ---: | ---: |
| Successful full episodes | 201/207 | 205/207 |
| Checkpoints with all three scenes successful | 63/69 | 67/69 |
| All-scene successes at the five targeted checkpoints | 0/5 | 5/5 |
| All-scene successes at TD 2001--2010 | 9/10 | 8/10 |
| All-scene successes at TD 9946--9966 | 21/21 | 21/21 |
| Final scene successes, TD 10000 | 3/3 | 3/3 |

The five targeted checkpoints are TD 2031, 2226, 2344, 2345 and 2346. Their
scene-10001 outcomes change from 200-step budget failures to success in
41, 42, 41, 41 and 41 steps respectively. Scenes 10000 and 10002 remain
successful at every sampled treatment checkpoint.

Treatment scene 10001 fails at TD 2004 and TD 2009. Both first depart from the
reference at step 6, selecting `east_small` (2) instead of `south_small` (1).
Both failures are preserved and independently reloaded. The earlier control
failed at TD 2003 and at the five targeted later checkpoints; treatment
removes those observed failures but introduces the two early failures above.
Thus the early ten-update window worsens, even though overall sampled retention
improves. An interim progress message missed TD 2009; the complete trace audit
and all final statistics include it.

All 59 sampled checkpoints from TD 2010 through 10000 succeed on all three
scenes. This is a statement about the declared checkpoints, not every intervening
model. The final model agrees with 117/117 original labels; agreement is perfect
at 7031/8000 updates and falls as low as 115/117 elsewhere.

## Learned recovery versus avoiding the state

At every treatment checkpoint, a separate conditional probe executes the
same recorded 24-action prefix, then relinquishes all control to the neural
policy. No action is forced at step 25 or later. These episodes are not pooled
with autonomous full-episode success rates.

| Conditional measurement | Result |
| --- | ---: |
| Initial TD-2000 model | Fails; chooses east_large |
| Treatment checkpoints after learning begins | 68/68 successful; all choose north_large |
| Seven matched baseline probes, including TD 2000 | 1/7 successful |
| Same seven treatment probes, including TD 2000 | 6/7 successful |
| Recovery-state greedy north_large after each update | 8000/8000 |
| Minimum recovery Q gap against every other action | +0.171905518 |
| Final recovery Q gap against every other action | +1.260749817 |

The seven baseline probes are TD 2000, 2031, 2226, 2344, 2345, 2346 and 10000.
Only the final baseline model succeeds. Treatment learns the recovery action
by TD 2001 and retains a positive recovery gap through TD 10000; all 68 later
conditional probes finish in 42 steps.

There is also direct evidence in fully autonomous episodes. At TD 2033, 2200,
2201, 2225, 2226 and 3000, the network itself chooses the complete historical
24-action prefix and reaches the same recovery input. It then independently
chooses `north_large` and succeeds in 42 steps. Exact prefix actions and paths
match the conditional probe; no scripted action is used in these full episodes.
Both early failed models also pass their conditional recovery tests. Their
remaining weakness is upstream of the recovery state.

## Neighboring decisions and limitations

Reference-state ranking reversals remain:

| Reference step | Negative-gap updates, control | Negative-gap updates, treatment |
| --- | ---: | ---: |
| 6 | 9 | 10 |
| 18 | 0 | 0 |
| 20 | 0 | 0 |
| 21 | 0 | 0 |
| 22 | 690 | 663 |
| 23 | 0 | 0 |
| 24 | 2 | 8 |

The learned recovery makes some deviations survivable; it does not eliminate
all upstream ranking changes. The increase at steps 6 and 24 is retained as
an adverse observation. Reference gaps alone do not establish physical failure
for an untested model, and this attempt does not isolate the gradient cause
of those reversals.

This is one inherited initialization, one recovery observation and three known
training scenes. The 69-checkpoint schedule intentionally includes previously
problematic intervals and is not an unbiased success-rate estimate. There is
no held-out generalization test or proof of success at all 8000 intermediate
models. A useful next diagnostic is the new TD-2004/2009 step-6 failure; no
additional training remedy was applied in this attempt.

## Validation and work accounting

A disposable zero-addition update exactly matches the historical control's
metrics, full parameters, target and Adam state. Independent treatment math
matches the added-loss update exactly, while original pre-update TD and
supervision terms remain identical to the control. The four gate updates do
not enter treatment.

Independent computation replays updates 2001--2201 and 9801--10000 (401 total),
matching every update record and applicable saved full snapshots. Every saved
target lineage and Adam clock is audited. Twenty-eight predeclared reload
episodes reproduce all three full trajectories and the conditional trajectory
at seven models. Eight additional, post-hoc reload episodes reproduce the two
new failed models and their successful conditional recoveries. The latter use
the same conditional adapter as the main run; the predeclared conditional
reloads use a separately written adapter. No optimizer updates occur in reloads.
All model state comparisons and reset-certification checks pass.

| New simulator work | Resets | Executed actions | Copied certification actions |
| --- | ---: | ---: | ---: |
| Recovery-state collection | 1 | 24 | 41 |
| Seven baseline conditional probes | 7 | 1242 | 287 |
| Treatment autonomous episodes | 207 | 8397 | 8073 |
| Treatment conditional probes | 69 | 3056 | 2829 |
| Predeclared full-episode reloads | 21 | 820 | 819 |
| Predeclared conditional reloads | 7 | 452 | 287 |
| Post-hoc failed-model full reloads | 6 | 552 | 234 |
| Post-hoc failed-model conditional reloads | 2 | 84 | 82 |
| Total | 320 | 14627 | 12652 |

Total simulator work is 27,279 actions. The executed-action column includes
2,040 scripted-prefix actions in conditional probes; there are zero forced
recovery actions. Optimizer work is 8000 treatment + 4 gate + 401 independent
updates = 8405. Reused historical control episodes and unit-test work are
excluded from these new-work counts. No held-out scene is accessed.

The primary run completed once without failure, from 05:54:30.000665 to
06:15:43.458192 UTC. The supplemental failed-model verification completed once.
Forty-nine regression tests passed in 9.69 s; the final 11 focused tests,
including conditional-probe guards, passed in 6.24 s. Ruff passed. Every
inference checkpoint's online weights match its complete training snapshot.

## Reproduction and retained evidence

From the repository root, with the verified historical artifacts available:

```powershell
.venv/Scripts/python.exe -u docs/experiments/025_recovery_supervision/run_recovery_supervision.py
.venv/Scripts/python.exe -u docs/experiments/025_recovery_supervision/verify_failure_reloads.py
.venv/Scripts/python.exe docs/experiments/025_recovery_supervision/analyze_recovery_supervision.py
```

The scripts refuse to overwrite their run, supplemental verification or archive.
Local models and full raw logs are under
`results/training/recovery_supervision_20261115`. The final inference model is
`models/update_10000.pt`, and its complete training snapshot is
`models/training_state_10000.pt`. Both have recorded SHA-256 hashes.

The [archive manifest](evidence/archive_manifest.json) binds the
[recovery input](evidence/recovery_sample.json), [comparison and failure analysis](evidence/analysis.json),
[all full episodes](evidence/observations.json), [all conditional probes](evidence/conditional.json),
[every update's compact Q diagnostics](evidence/dense_compact.json),
[verification](evidence/verification.json), [failed-model reloads](evidence/failure_reload_verification.json),
[work counts](evidence/accounting.json), provenance, model hashes and validation.
Full raw-log hashes and sampled update records are retained; model binaries
and full raw logs remain local. The immutable pre-run protocol remains
unchanged. All code and comments are English. No report was edited.
