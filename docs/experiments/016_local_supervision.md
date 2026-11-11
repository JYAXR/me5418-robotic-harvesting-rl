# Attempt 016: double supervision for one fragile decision during continued TD

## Protocol recorded before execution

Attempt 014 kept learning rate 0.0001 and Adam state after return calibration,
but scene 10001 failed at absolute TD updates 2009 and 10000. Attempt 015
reproduced those failures and rescued both by replacing only step 21 with
`north_small`. The models then completed the route autonomously in 41 actions.
This motivates a controlled training test of the supervision at that state;
the diagnostic intervention itself did not improve either learned policy.

Create two fresh branches from the **same complete attempt 013 treatment
snapshot at absolute TD update 2000**. Restore online and target weights,
Adam moments and internal parameter steps, the TD update clock and both RNG
states exactly. Do not reset Adam, resynchronize the target or rerun the
supervised/return-calibration stages. The inherited branch has 42000 updates:
20000 supervision, 20000 return calibration and 2000 TD. Adam's internal
parameter step is 22000.

## The only training factor

Use the same ordered 117 demonstration transitions and original returns from
training seeds 10000, 10001 and 10002. Scene 10001 occupies zero-based rows
35--75. Its **one-based step 21 is zero-based row 55**, demonstrated action
`north_small (0)`. Check the row identity, encoded state hash and action
against the committed source evidence before training.

Let `h_i` denote the existing action-margin hinge loss for demonstration row
`i`, with margin 0.8. The objectives are:

- **Baseline:** `L_TD + sum(h_i) / 117`.
- **Treatment:** `L_TD + (sum(h_i) + h_55) / 117`.

Equivalently, the treatment gives supervised row 55 coefficient 2 and every
other row coefficient 1. The denominator remains **117**, not the sum of
weights 118. Thus each other supervised coefficient remains unchanged.
The global supervision multiplier remains 1. Do not duplicate transitions,
upweight that row's TD loss, change the sampling distribution or add examples.
At a shared parameter/target state, the only objective difference is
`h_55 / 117`; resulting later optimizer and target differences are consequences
of this factor.

Keep learning rate **0.0001**, gamma **0.99**, gradient clip **10**, target
synchronization interval **200**, network **341/256/256/8**, environment,
encoder, rewards, action definitions and full-batch training unchanged.
Implement the experimental weighted update separately from immutable production
and previous experiment code. Test its all-ones equivalence to the original
updater and its weighted objective/gradient against an independent formulation.

## Fixed budget and baseline reproduction

Train **8000 new TD updates per arm**, from absolute TD 2000 through 10000.
This is **16000 new main optimizer updates**, with the 42000-step shared history
reused once, not counted twice as new work. Each branch ends with 50000 total
updates and Adam parameter step 30000. Target synchronization remains on the
absolute clock at 2200, 2400, ..., 10000: 40 new synchronizations per arm.

Run the baseline first using the original updater. Require exact reproduction
of attempt 014's 8000 update records, 29 physical observations and complete
training snapshots before beginning the treatment. Require exact payload
equality for all 67 full/inference checkpoint files, including metadata,
without requiring identical serialization bytes. Preserve the original baseline evidence and record equivalence
in this new run. A failed equivalence check invalidates the comparison and must
be recorded; it is not a reason to change the treatment to obtain success.

For each arm, physically evaluate all three scenes at:

- 2000, confirming the restored initial behavior;
- every update 2001--2010, explicitly including the known failure at 2009;
- 2200 and 2201, the next ordinary target synchronization and following update;
- 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000,
  8500, 9000, 9500 and 10000.

This gives **29 checkpoints and 87 episodes per arm**, or **174 main episodes**.
All actions come from the greedy neural policy: no forced step-21 correction,
planner fallback or action override is used during evaluation. Retain every
failure and recovery. Continue through 10000 even if an intermediate model
fails; do not stop early, roll back or select a successful checkpoint as the
final result.

## Neighbor monitoring and retained diagnostics

After every update, plus the restored TD-2000 boundary, record batch-of-one Q
probes on scene 10001's fixed demonstration states at steps
**6, 18, 20, 21, 22, 23 and 24**. There are 8001 records per arm, 16002 total,
containing 112014 individual state forward passes. Steps 20--23 test the target
decision and its neighbors; steps 6, 18 and 24 retain earlier diagnostic/control
states. Save all eight Q values, demonstrated and greedy actions, and the signed
gap `Q(demonstrated) - max(Q(other))`. The probes must preserve weights, Adam,
RNG and module mode flags, and add no environment actions or optimizer updates.

The local intervention should not be judged only by step 21's gap. In particular,
check whether favoring `north_small` damages neighboring decisions that require
`north_large`, or whether failures move to another state or scene. Report each
scene's actual outcomes and first actual reference-action divergence, alongside
negative-gap intervals, minima, ties and action-rank changes in all seven probes.
Reference-state reversals, including unvisited states after a trajectory fork,
are **not measured failed episodes**.

Save inference and full training snapshots at the 29 physical checkpoints, plus
otherwise unsaved target-source states at 2400, 3400, ..., 9400. This retains
67 model files per arm, including the final full-state alias. Keep all 8000
update diagnostics and a separate supervision log per arm. Baseline update logs
retain their original schema for exact source comparison. Treatment
`demonstration_loss_before/after` denotes the weighted mean; supplementary logs
separately retain the unweighted mean hinge and selected-row hinge. These
quantities must not be conflated when comparing loss curves.

Record actual gradient norms, clipping, parameter changes, return-calibration
error and demonstration agreement. The update log's action accuracy is measured
before the update; final policy accuracy comes from its saved observation.
Keep the complete compact gap/action series and all physical outcomes in Git.
Raw full Q logs and model/data binaries stay locally with recorded hashes.

## Independent verification and accounting

Independently replay updates **2001--2201 for both arms**, reproducing weighted
updates through an independent objective implementation. This adds
**402 verification updates**, separately from 16000 main updates. Compare logs,
Q probes and saved online/target/Adam/RNG states across the first ten steps and
the target synchronization at 2200. Inspect all later model hashes, counters
and recorded target lineage without claiming a numerical replay of all 16000.

Reload policies at **2000, 2009, 2200, 2201 and 10000** from both arms, on all
three scenes: **30 additional verification episodes**, with their actual actions
counted separately. The selection is fixed regardless of which new checkpoints
fail. Read-only analysis and plots add no episodes or optimizer updates.
Reset feasibility probes are excluded from policy-action counts as before.

Commit and push the code/protocol before execution, then preserve all results,
failures, resolved settings, source/helper hashes and exact launch commit.
The code, comments and runtime messages remain English. Previous attempts,
production defaults, reports and submission archives are outside this change.

## Interpretation limits

This is a targeted mechanism test on three known training scenes from one
initialization. The target row was selected using earlier diagnostic failures;
it is not held-out validation. Improved target-state ranking alone does not
establish robust retention. All-scene success at every scheduled checkpoint is
the primary sampled-retention criterion, while dense probes describe additional
rank stability and neighbor tradeoffs. Neither establishes success at every
unobserved state or on unseen scenes. A failed treatment remains useful evidence
and will not be hidden by its final checkpoint or mean loss.

## Reproduction

Run from the repository root using the configured Python environment:

```powershell
python docs/experiments/016_local_supervision/run_local_supervision.py --reference-run results/training/td_extension_10000_20261112 --archive-dir docs/experiments/014_td_extension/evidence --output-dir results/training/local_supervision_20261112
python docs/experiments/016_local_supervision/verify_local_supervision.py --run-dir results/training/local_supervision_20261112 --output results/training/local_supervision_20261112/reload_checks.json
python docs/experiments/016_local_supervision/analyze_local_supervision.py --run-dir results/training/local_supervision_20261112 --output results/training/local_supervision_20261112/q_analysis.json
python docs/experiments/016_local_supervision/plot_local_supervision.py --run-dir results/training/local_supervision_20261112
python docs/experiments/016_local_supervision/archive_local_supervision.py --run-dir results/training/local_supervision_20261112 --destination docs/experiments/016_local_supervision/evidence
```

Local execution uses `.venv/Scripts/python.exe`. The original local binary
checkpoints and demonstration data are required; compact Git evidence does not
include those files. The runner refuses to overwrite an existing output
directory. The code and protocol used for execution were committed as
`597e92648beb70c3c99becf4edfdaa7a75397b40`.

## Completed results

Doubling the selected sample's supervision preserved the step-21 decision, but
**did not stably preserve all three scenes or their neighboring decisions**.
The treatment finished with three successful scenes and avoided the baseline's
two observed step-21 failures. It also introduced three observed failures whose
first reference-action divergence was at step 6, and new reference-state action
reversals at step 22. On the predeclared
sampled-retention criterion, the treatment was worse: 26/29 checkpoints retained
all three scenes, compared with 27/29 for the baseline.

The main run completed on 11 November 2026, from
20:32:16.904455 to 21:20:47.199432 UTC, taking about 48 minutes 30 seconds.
The pre-launch suite passed **578 tests in 199 seconds**, and Ruff passed.
The recorded dirty-worktree flag came from an existing untracked report;
the 52 recorded production/configuration source hashes remained unchanged.
Both arms restored the same full TD-2000 snapshot, and their initial Q vectors
matched the source on all 117 demonstration states. The selected observation
hash also matched the earlier diagnostic evidence.

Before the treatment started, the baseline passed the exact attempt-014
reproduction gate: **8000 update records, 29 observations, 67 complete
checkpoint payloads and 8001 inherited four-state probe records** matched.
The additional three probes did not alter its training trajectory.
The intervention remained exactly `h_55 / 117`: only supervised row 55 had
coefficient 2, the other 116 coefficients stayed 1, and the TD term was unchanged.
At the first shared state, the weighted supervision loss increased from
0.063221075 to 0.068521939, consistent with adding the selected hinge
0.620201111 divided by 117 within float32 rounding.

### Physical outcomes at the fixed checkpoints

| Measure | Baseline: selected weight 1 | Treatment: selected weight 2 |
| --- | ---: | ---: |
| All-three-scene successes, TD 2001--2010 | 9/10 | 10/10 |
| Individual scene successes, TD 2001--2010 | 29/30 | 30/30 |
| All-three-scene successes, all scheduled checkpoints | 27/29 | 26/29 |
| Individual scene successes, all scheduled checkpoints | 85/87 | 84/87 |
| Scene 10000 successes | 29/29 | 29/29 |
| Scene 10001 successes | 27/29 | 26/29 |
| Scene 10002 successes | 29/29 | 29/29 |
| Final TD-10000 scene successes | 2/3 | 3/3 |
| Main evaluation policy actions | 3711 | 3871 |

Both arms completed scenes 10000 and 10002 at every scheduled checkpoint.
The treatment repaired scene 10001 at TD 2009 and 10000, completing each in
41 actions. At its final checkpoint, the three episode lengths were 35, 41
and 41. However, it failed scene 10001 at TD 4500, 8000 and 8500, when the
baseline succeeded. Every failure exhausted the 200-action budget without
damage. All five observed failures are retained below; the gap is
`Q(demonstrated) - max(Q(other))` at the first actual reference-action divergence.

| Arm | Absolute TD update | First divergent step | Demonstrated action | Chosen action | Signed Q gap | Episode return |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| Baseline | 2009 | 21 | `north_small` | `north_large` | -0.016716 | 2.479857 |
| Baseline | 10000 | 21 | `north_small` | `north_large` | -0.059891 | 2.479857 |
| Treatment | 4500 | 6 | `south_small` | `east_small` | -0.013138 | -104.649724 |
| Treatment | 8000 | 6 | `south_small` | `east_small` | -0.052422 | -104.565207 |
| Treatment | 8500 | 6 | `south_small` | `east_small` | -0.042625 | -104.565207 |

The baseline recovered at the next scheduled check, TD 2010, after its
TD-2009 failure. The treatment recovered at TD 5000 after its TD-4500 failure,
and at TD 9000 after the failures at 8000 and 8500. These recoveries were
observed checkpoints; no unmeasured recovery time is inferred.

Each treatment failure contained 171 blocked actions and 170 stationary actions.
These were autonomous rollouts, with no correction or fallback. The earlier
step-6 divergence means that subsequent fixed demonstration states, including
steps 21 and 22, no longer describe the states visited on those failed routes.
The treatment's successful scene-10001 episode at TD 2008 took 42 actions;
success should therefore not be confused with exact demonstration reproduction.

### Target and neighboring action rankings

The following table covers all seven fixed scene-10001 reference states.
Negative counts use the **8000 new updates, TD 2001--10000**. Minimum gaps
include the restored TD-2000 boundary; final gaps are measured at TD 10000.
These are batch-of-one Q probes, not additional physical episodes.

| Reference step | Negative counts, baseline / treatment | Minimum gap, baseline | Minimum gap, treatment | Final gap, baseline | Final gap, treatment |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 394 / 600 | -0.323555 | -0.285713 | 0.255623 | 0.187805 |
| 18 | 0 / 0 | 0.105278 | 0.101379 | 0.700241 | 0.873474 |
| 20 | 0 / 0 | 0.520699 | 0.735703 | 0.689560 | 1.469482 |
| 21, reweighted | 627 / 0 | -0.223328 | 0.128944 | -0.059891 | 0.731766 |
| 22 | 0 / 63 | 0.224068 | -0.148010 | 0.807518 | 0.221893 |
| 23 | 0 / 0 | 0.186195 | 0.312515 | 0.922073 | 0.794777 |
| 24, control | 102 / 52 | -0.211525 | -0.176292 | 0.802681 | 0.730171 |

The intervention removed all **627** new-update ranking reversals at the
targeted step 21. Its lowest treatment gap was still positive, 0.128944 at
TD 2021. Steps 18, 20 and 23 retained the demonstrated greedy action at every
probe in both arms. There were no zero-gap ties at any of the seven states.

The neighboring step 22 did not retain that stability: it acquired **63 negative
gaps across 49 intervals**, starting at TD 2501 and reaching -0.148010 at
TD 3536. All 49 intervals fell entirely between scheduled physical evaluations.
The baseline had no reversal there. Each of these 63 paired regressions
coincided with a larger step-21 gap under the treatment, which documents a
tradeoff between the two decisions. It does not establish 63 failed episodes
or show that step 22 caused the three measured treatment failures.

At step 6, negative-gap counts increased from **394 to 600**, with 267 baseline
intervals and 385 treatment intervals. Its first treatment reversal occurred
at TD 2011, immediately after the ten successful early checkpoints; three
intervals included the measured failures at 4500, 8000 and 8500. Thus the clean
early result did not establish sustained retention. At step 24, the negative
counts decreased from 102 to 52. Including the initially negative TD-2000
boundary gives 103 and 53, respectively. This control state was already known
to admit successful deviations, reinforcing why ranking disagreement alone
must not be counted as rollout failure.

### Fit, optimizer behavior and interpretation

| Diagnostic | Shared TD-2000 start | Baseline TD 10000 | Treatment TD 10000 |
| --- | ---: | ---: | ---: |
| Full-batch demonstration action agreement | 116/117 | 115/117 | 117/117 |
| Unweighted demonstration hinge loss | 0.063221 | 0.044419 | 0.031453 |
| Demonstration discounted-return Q RMSE | 0.461398 | 0.499196 | 0.630023 |

The treatment ended with a lower unweighted hinge loss and complete full-batch
action agreement, yet its Q-value fit to the fixed demonstration discounted
returns was worse. These diagnostics describe different properties: improving
action separation did not preserve return calibration or continuous rollout
success. The treatment's final weighted hinge loss was 0.032036; it is a
different objective from the baseline loss and is not used as an unqualified
cross-arm performance measure. Final agreement above is measured after the
update, separately from the pre-update accuracy in the raw update log.

All **8000 updates in each arm changed the parameters**, and each arm performed
40 target synchronizations on the original absolute clock. The pre-clipping
gradient norm exceeded 10 on 7149/8000 updates in each arm. Mean norms were
17.855495 and 17.738506 for baseline and treatment; final parameter displacement
from the shared resume state was 2.266949 and 2.304578. The comparison therefore
did not preserve behavior by silently stopping learning or resetting Adam.

This single-factor result supports a narrow conclusion: extra supervision on
one diagnosed state can protect that state's preferred action, but does not
provide stable route retention. The treatment repaired the selected failure
while exposing earlier route failures and a neighboring ranking tradeoff.
It should not be promoted as a validated stability fix based on its final 3/3
success or its lower mean hinge loss. Any follow-up should test another
predeclared factor while retaining all-scene physical checks and dense
neighbor monitoring. No held-out scenes, additional initializations or
generalization claims are included in this attempt.

### Verification, accounting and retained evidence

The main experiment consumed **16000 new optimizer updates and 174 evaluation
episodes with 7582 policy actions**. The two branches reused the same 42000
inherited updates; that history was not rerun or counted twice. Each branch
ended at absolute TD 10000, 50000 cumulative updates and Adam parameter step
30000. The 16002 dense records contained 112014 state forward passes and
added no environment actions or optimizer updates. There were no new online
training actions, demonstration-collection episodes or held-out evaluations.

Read-only analysis checked all 16002 seven-state records, all 58 checkpoint
observations and 151 recorded file hashes, including 136 model/source files.
It confirmed nonzero parameter changes on all 16000 main updates. This audit
added no environment resets, actions or optimizer updates.

Independent numerical replay and policy-reload verification **passed**.
Each arm independently reproduced TD 2001--2201 using a separate implementation
of the objective, including the ordinary target synchronization at 2200.
Update logs, supervision diagnostics, Q probes and all saved full states in
that interval matched exactly, including online/target weights, Adam and both
RNG states. This added **402 verification updates**. Reloading the fixed
TD 2000, 2009, 2200, 2201 and 10000 policies from both arms reproduced all
**30 verification episodes, with 1488 policy actions**. These costs are
separate from the main experiment. Later logs, all 134 model files, counters
and target lineage were audited without numerically replaying all 16000 main
updates. See the [independent verification](016_local_supervision/evidence/reload_checks.json).

The retained evidence directory is
[`016_local_supervision/evidence`](016_local_supervision/evidence).
It contains 30 evidence files plus the archive manifest, totaling 9297102
bytes. Every archived hash, byte count and copied source file was checked.
Per arm, the archive retains 170 update samples, 170 supervision samples and
171 full-Q probe samples; the compact gap/action series retains every update.
The full raw logs have recorded hashes and coverage counts. Both figures
passed visual inspection, and the final Ruff and diff-whitespace checks passed.
A scoped Git attribute disables line-ending conversion for this evidence
directory, preserving the archived bytes and checksums on other platforms.
It contains the [summary](016_local_supervision/evidence/summary.json),
[baseline equivalence gate](016_local_supervision/evidence/baseline_equivalence.json),
[accounting](016_local_supervision/evidence/accounting.json),
[provenance](016_local_supervision/evidence/provenance.json),
[Q analysis](016_local_supervision/evidence/q_analysis.json), and complete
compact dense series for the
[baseline](016_local_supervision/evidence/q_analysis_baseline_dense.json) and
[treatment](016_local_supervision/evidence/q_analysis_treatment_dense.json).
The [outcome figure](016_local_supervision/evidence/local_supervision_outcomes.png)
and [neighbor-gap figure](016_local_supervision/evidence/neighbor_q_gaps.png)
show the entire prescribed update range, including every scheduled failure.
Raw Q logs, all update records and new checkpoint binaries remain under
`results/training/local_supervision_20261112`. The reused ordered 117-row
demonstration tensor remains in the attempt-011 source directory,
`results/training/multiscene_retention_20261112`, identified by
`input_manifest.original_data_run`. Their hashes are retained in the compact
archive; no duplicate demonstration binary was created for this attempt.
