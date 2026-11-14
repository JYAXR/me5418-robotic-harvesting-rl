# Attempt 022: local supervision with TD coefficient 0.5

## Protocol declared before execution

Change only the supervised coefficient on original demonstration row 55
(scene 10001, one-based step 21, action 0 north_small) from 1 to 2.
Use 0.5 * mean(TD Huber) + mean(weighted demonstration hinges), with all
other 116 coefficients fixed at 1, hinge margin 0.8, and denominator 117.
Do not renormalize by weight sum, alter margin, duplicate data, add MC loss,
force actions, reset Adam, change learning rate, or modify production defaults.

Restore the same complete TD-2000 snapshot as attempt 019 and retain online,
target, Adam moments/steps, RNG and absolute clock. Train 8000 updates through
TD 10000 with learning rate 0.0001, gamma 0.99, clipping 10 and target period
200. No early stopping or checkpoint selection. Reuse attempt 019 as the
historical coefficient-one control, augmented by its attempt-020 physical
checks at every update 9946--9966. Audit both committed source archives and
all referenced snapshot/data hashes before starting.

Evaluate the same 29 original checkpoints plus all 21 updates 9946--9966:
50 physical checkpoints, three known scenes each, 150 autonomous episodes.
Record complete trajectories, failure, blocking, return and demonstration fit.
Probe all seven existing scene-10001 reference states after every update;
record full-batch action agreement on all 117 reference states as an additional
monitor. Full-batch predictions are not physical outcomes and may differ from
batch-one inference at numerical ties. Retain every failure and ranking flip.

Check initial behavior/Q equality. Save full snapshots at every evaluation and
every target synchronization. Test exact equivalence to attempt 019 when the
selected coefficient is 1, and independently compute the coefficient-2 update.
Independently replay TD 2001--2201 (201 updates), including target sync, and
compare logs, Q probes and saved full states. Independently reload TD 2000,
2009, 2200, 2201, 9955, 9956 and 10000 on all three scenes (21 episodes),
requiring exact trajectory and reset-certification equality. Audit later state
hashes, counters and target lineage without claiming full numerical replay.

Reset certification remains enabled and its simulated actions are counted
separately. Historical control and inherited training are reused, not rerun.
No held-out evaluation or new demonstration collection. Stable sampled success
and stable reference ranking are separate criteria; neither proves performance
at unsampled physical states. Record English code, all evidence and results in
Git. The user reports remain unchanged. Results pending at protocol commit.

## Completed results: 14 November 2026

The predeclared coefficient-2 treatment completed all 8000 updates. One formal
training run and one analysis/archive pass completed without execution failure.
The selected step-21 action remained correctly ranked at every new update, and
the two known late failures disappeared under autonomous policy execution.
This is an improvement on the tested continuation, **not complete stable
retention across the three scenes**: the early step-6 failure remained and a
neighboring reference decision became less stable.

| Matched measure | Historical local weight 1 | New local weight 2 |
| --- | ---: | ---: |
| All-scene successful checks, full matched schedule | 47/50 | 49/50 |
| Successful individual episodes | 147/150 | 149/150 |
| All-scene successful checks, original schedule | 28/29 | 28/29 |
| All-scene successful checks, TD 9946--9966 | 19/21 | 21/21 |
| Scene-10001 failures in the combined schedule | 2003, 9955, 9956 | 2003 |
| Final autonomous success | 3/3 | 3/3 |
| Final demonstration agreement | 117/117 | 117/117 |
| Final actual-return RMSE | 0.660312 | 0.787949 |

Both arms in this table use TD coefficient 0.5; the comparison is not against
the earlier coefficient-1 TD experiments. The historical 29-checkpoint run and
21-checkpoint late diagnostic are reused, not new control training/evaluation.
The late window was selected after observing past failures, so the combined
counts are a matched targeted comparison, not an unbiased success-rate estimate.

At TD 9955 and 9956, the new policy completed scene 10001 autonomously in 41
steps, with no action override. All 63 episodes in the late window succeeded.
The final model also completed scenes 10000, 10001 and 10002 in 35, 41 and 41
steps. The successful step-21 behavior is now produced by the learned network,
whereas attempt 021 required an external correction at that step.

The remaining TD-2003 failure first deviated at scene 10001 step 6, choosing
`east_small` instead of `south_small`, and exhausted the 200-step budget.
Its return was -104.71076339534699. The next sampled checkpoint, TD 2004,
succeeded again. Both arms therefore still have only 9/10 all-scene successes
in the first ten new updates. The new method does not fix that early failure.

## Dense ranking and neighboring decisions

The counts below cover TD 2001--10000, excluding the shared initial boundary.
They are batch-one reference-state measurements, not executed failed episodes.

| Reference step | Negative-gap updates, weight 1 | Negative-gap updates, weight 2 | Final gap, weight 2 |
| --- | ---: | ---: | ---: |
| 6 | 18 | 9 | 0.479561 |
| 18 | 0 | 0 | 0.783653 |
| 20 | 0 | 0 | 1.713211 |
| 21 | 153 | 0 | 0.800568 |
| 22 | 0 | 690 | 0.223755 |
| 23 | 0 | 0 | 0.754616 |
| 24 | 5 | 2 | 0.761894 |

Step 21 had **zero** negative gaps across all 8000 new updates. Its minimum gap
was +0.186546 and final gap +0.800568. This supports stable selection of
`north_small` on this specific recorded input throughout the tested continuation.
It does not establish correct decisions on unseen or off-demonstration states.

The improvement has a measurable trade-off: step 22 acquired **690** negative
gaps, first at TD 2031 and last at TD 9847, with minimum gap -0.312080.
Its historical control had no reversals. None of these reference-state records
alone establishes a failed physical episode; the scheduled scene evaluations
had no newly observed late failure. A complete-episode diagnostic at selected
step-22 reversal models is needed before treating those flips as harmless or
as confirmed failures. The stronger local coefficient should not be promoted
as a general stability fix on the present evidence.

Full-batch inference on all 117 demonstration states gave perfect agreement
on 7032/8000 new updates, with a minimum of 114/117 and final 117/117. The
batch-one sentinel checks remain the authoritative seven-state ranking probes;
full-batch predictions can differ near numerical ties. Successful reference
action deviations were seen on scene 10002 step 37 at TD 3000, 6000 and 7000,
without episode failure. The restored TD-2000 boundary already had 116/117
agreement while all three scenes succeeded. Task success and exact imitation
must therefore remain separate measures.

Actual-return RMSE worsened despite stronger action retention at the selected
state. There was no MC objective; actual discounted demonstration returns were
used only to report calibration error. The supervised loss in update logs is
weighted for the treatment, so it is not directly an unweighted comparison
against the historical supervised loss. The denominator remains 117, not 118.

## Verification, tests and actual work

Before treatment, a three-update real-model gate verified exact historical
coefficient-one behavior, complete-state equality, and the independently
computed first coefficient-two objective. Those gate updates were performed on
separate restored copies and were not inherited by training.

An independent implementation then reproduced TD 2001--2201 exactly, including
loss records, reference Q vectors, available full states and target sync at
2200. All later saved full-state counters and target lineage were audited.
There were 40 target synchronizations, with Adam parameter clocks continuing
from 22000 to 30000. The analyzer verified hashes for all 131 model files
(81 full snapshots plus 50 inference checkpoints) and matching online weights
between each inference checkpoint and full snapshot.

Twenty-one reload episodes at seven declared checkpoints reproduced complete
trajectories and reset certification exactly. All physical evaluations and
reloads preserved full training state. No action was forced. Later training
updates were not all numerically replayed, and unsampled physical outcomes
remain unknown.

| New formal work | Count |
| --- | ---: |
| Main training updates | 8000 |
| Independent replay updates | 201 |
| Preflight gate updates | 3 |
| Main autonomous episodes | 150 |
| Independent reload episodes | 21 |
| Main policy actions | 6009 |
| Main copied-simulator certification actions | 5850 |
| Reload policy actions | 819 |
| Reload copied-simulator certification actions | 819 |
| Total simulator actions | 13497 |

The 8204 optimizer updates above exclude inherited training and historical
control work. Reset certification remained enabled; counters distinguish its
simulation actions from policy actions. There was no new demonstration
collection, online exploration or held-out evaluation. Unit-test execution is
separate from these formal work counts. The focused tests passed 46/46 in
10.64 seconds; the full suite passed **846 tests in 145.89 seconds**. Ruff passed.

## Reproduction and retained artifacts

Run from the repository root with the original local source experiments:

```powershell
.venv/Scripts/python.exe -u docs/experiments/022_half_td_local_supervision/run_local_supervision.py
.venv/Scripts/python.exe docs/experiments/022_half_td_local_supervision/analyze_local_supervision.py
```

The runner and archiver refuse existing destinations. The local directory
`results/training/half_td_local_supervision_20261114/models` retains all models,
including `training_state_10000.pt` and the loadable `update_10000.pt`.
Full raw logs remain local. The [evidence manifest](022_half_td_local_supervision/evidence/archive_manifest.json)
indexes compact records, with all 8001 update states in the paired dense series,
complete physical outcomes, model/source hashes and sampled raw update/probe
logs. The [analysis](022_half_td_local_supervision/evidence/analysis.json)
contains every observed failure and all negative-gap update IDs; the
[verification](022_half_td_local_supervision/evidence/verification.json) records
exact replay/reload checks and work accounting.

The trial fixes the selected late decision on the tested states, while leaving
early failure and neighboring-state instability unresolved. All code and
comments are English. Reports and production defaults were not changed.
