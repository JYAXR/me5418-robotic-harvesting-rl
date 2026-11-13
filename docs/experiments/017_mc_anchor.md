# Attempt 017: retain actual-return calibration during TD learning

## Protocol recorded before execution

Attempt 016 protected scene 10001 step 21 by doubling its supervised sample
coefficient, but sampled three-scene retention fell from 27/29 to 26/29.
New failures first diverged at step 6, and neighboring step 22 acquired 63
negative Q gaps between physical checkpoints. Its final demonstration-return
RMSE also increased from the baseline's 0.499196 to 0.630023. These observations
motivate testing a global value constraint rather than choosing another
individual sample. They do not establish that value drift caused the failures.

The new treatment retains the existing TD objective and adds a persistent
constraint toward the actual discounted returns of the original demonstrations.
The comparison uses the verified **attempt-016 baseline**, where every
supervised coefficient is 1. The locally reweighted treatment from attempt 016
is background evidence, not the matched control for this single-factor trial.

## One additional objective term

For the same ordered 117 transitions, define:

- Historical control: `L_TD + L_margin`.
- New treatment: `L_TD + L_margin + 1.0 * L_MC`.
- `L_MC = mean(SmoothL1(Q(s_i, a_demo_i), G_demo_i))`.

Each term uses its original mean over 117 rows. The total is not divided by
three, and the TD and MC targets are not blended into one target. All 117
supervised margin coefficients remain 1, with margin 0.8; there is no selected
row weighting. The MC coefficient is fixed at 1 before execution and will not
be tuned in response to intermediate outcomes.

The targets `G_demo` are the complete actual discounted demonstration returns
already used for calibration in attempt 011, computed separately for scenes
10000, 10001 and 10002 with gamma 0.99. Reuse their exact recorded tensor order
and values. They remain detached and fixed throughout TD; do not regenerate
them from a target network or accumulate rewards across episode boundaries.
They represent returns of the demonstration policy, not proven optimal Q
values. An MC constraint could also resist improvements beyond that policy.

The general use of demonstration supervision alongside TD has precedent in
[Hester et al., Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732).
This experiment's fixed full-trajectory MC anchor is a project-specific
addition; it does not implement the full DQfD algorithm or establish the same
performance claims. No external implementation is copied.

## Exact continuation and historical control

Restore the same successful full TD-2000 snapshot used in attempts 014 and
016: online and target networks, Adam moments and parameter steps, both RNG
states and the absolute TD clock. The inherited history is 20000 supervision,
20000 return calibration and 2000 TD updates. Adam starts at parameter step
22000. Do not rerun either pretraining stage or resynchronize the target at
the boundary.

Keep learning rate 0.0001, gamma 0.99, gradient clipping at 10, target
synchronization interval 200, network 341/256/256/8, full-batch data order,
environment, rewards, encoder and action definitions unchanged. Updates occur
in the existing experimental framework; production and earlier experiment
implementations remain immutable.

The historical control is attempt 016's completed baseline at commit
`e51effb2f5e0e05fb70486088d38ab235a395106`. Before new training, verify its
committed archive, source chain, raw logs, complete model hashes, exact
TD-2000 state and the earlier exact-reproduction gate against attempt 014.
Label it explicitly as reused historical evidence. Do not report its 8000
updates, 87 episodes or 3711 policy actions as newly executed work.

Train the new treatment for exactly **8000 updates**, TD 2001 through 10000.
It ends with 50000 cumulative updates on its branch and Adam step 30000.
The ordinary target synchronizations remain at 2200, 2400, ..., 10000.
Do not stop early, roll back a failure, freeze the network, override actions
or replace the prescribed final model with an earlier successful checkpoint.

## Evaluation and diagnostic coverage

Use the same 29 physical checkpoints:

`2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
2200, 2201, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500,
7000, 7500, 8000, 8500, 9000, 9500, 10000`.

Evaluate all three known scenes at each checkpoint using only greedy network
actions, giving **87 new main episodes**. Check the restored initial behavior
against the source before updating. Record success, damage, budget exhaustion,
return, action count, blocked/stationary actions and the first actual
reference-action divergence. Retain all failures and subsequent recoveries.

After every update and at the restored boundary, record all eight Q values on
scene 10001's fixed demonstration states at steps **6, 18, 20, 21, 22, 23, 24**,
one input per forward pass. Save greedy/demo labels and the signed difference
`Q(demo) - max(Q(other))`, using the same numerical procedure as attempt 016.
There are 8001 new dense records and 56007 forward passes. These probes must
preserve module modes and training/RNG state and execute no environment actions.

Report every negative-gap interval, ties, minima, final gaps and paired
changes against the historical control. Inspect neighboring steps 20--23 and
the earlier fragile step 6 together. A fixed-state reversal does not constitute
a measured rollout failure, and later reference states may be unvisited after
an actual trajectory diverges.

Save the same 67 model files: 29 inference checkpoints, 29 complete training
snapshots, eight otherwise unsaved target-source snapshots, and the final
full-state alias. Keep all 8000 update diagnostics, separately identifying TD,
unweighted margin and MC losses, their sum, parameter change and pre-clipping
gradient norm. Before/after TD losses use identical frozen Bellman targets.
Also record the difference between those targets and fixed MC returns.
Final action accuracy must be measured after updating, not taken from the
raw log's pre-update accuracy field.

## Verification and accounting

Synthetic tests must establish exact original-updater equivalence at MC
coefficient 0, independent objective/gradient equivalence at coefficient 1,
proper target timing and Adam continuity, detached return targets and
validation before mutation.

After the run, independently implement and replay TD 2001--2201 for both the
historical control and the new treatment. Compare logs, probes and saved full
states, including the target synchronization at 2200. This is **402 additional
verification updates**, not a rerun of the historical 8000-step control or a
numerical replay of every main update. Audit all later snapshot hashes,
counters, finite diagnostics and target lineage.

Reload only the new treatment's policies at TD **2000, 2009, 2200, 2201 and
10000**, each on all three scenes: **15 verification episodes**, with actual
actions counted separately. Previous control reload evidence is reused and
must not be counted again. Read-only analysis and figures add no actions,
resets or optimizer updates. Reset feasibility probes are excluded from
policy-action counts, consistently with earlier attempts.

New main work is therefore 8000 optimizer updates and 87 physical episodes;
shared historical pretraining and historical control costs remain separate.
No new demonstration collection, online interaction for training or held-out
scene access occurs in this experiment.

## Decision rule and scope

The primary criterion is retaining success on all three scenes at **every**
scheduled checkpoint. Compare the whole trajectory, not only the final
checkpoint or the average loss. Dense neighboring ranks and return
calibration provide separate diagnostics; improved RMSE cannot substitute for
successful behavior.

This is one fixed-coefficient test on three known training scenes from one
initialization. Record a failed or mixed result without changing its budget
or selecting another coefficient within the attempt. Even if it passes,
multi-layout training and independent-scene evaluation remain necessary before
claiming generalization or promoting this as a broadly reliable policy.

## Reproduction

Run from the repository root in the configured environment:

```powershell
python docs/experiments/017_mc_anchor/run_mc_anchor.py --reference-run results/training/local_supervision_20261112 --archive-dir docs/experiments/016_local_supervision/evidence --output-dir results/training/mc_anchor_20261113
python docs/experiments/017_mc_anchor/verify_mc_anchor.py --run-dir results/training/mc_anchor_20261113 --output results/training/mc_anchor_20261113/reload_checks.json
python docs/experiments/017_mc_anchor/analyze_mc_anchor.py --run-dir results/training/mc_anchor_20261113 --output results/training/mc_anchor_20261113/q_analysis.json
python docs/experiments/017_mc_anchor/plot_mc_anchor.py --run-dir results/training/mc_anchor_20261113
python docs/experiments/017_mc_anchor/archive_mc_anchor.py --run-dir results/training/mc_anchor_20261113 --destination docs/experiments/017_mc_anchor/evidence
```

The local interpreter is `.venv/Scripts/python.exe`. Original local model and
data binaries are required; the Git evidence archive contains compact records
and hashes. Refuse to overwrite any earlier run. Commit the protocol and
implementation before execution, then record and push all outcomes.

Before execution, the full suite passed **655 tests in 77.40 seconds** and
Ruff passed. The 77 added focused tests cover update semantics, independent
verification and analysis. A read-only real-source preflight verified the
complete TD-2000 state, 117 transitions, fixed returns and historical control;
it performed no optimizer updates or environment actions.

The protocol and implementation used for execution were committed as
`22bab47f747baae667212b90912dba700b0adc39`.

## Completed results

Persistent return anchoring improved Q-value calibration but **did not improve
stable route retention**. The final model succeeded on all three scenes and
matched all 117 demonstration actions. Across the prescribed checkpoints,
however, all-scene success fell from the historical control's 27/29 to 23/29.
The treatment had six observed failures across two scenes, including five
scene/checkpoint pairs where the control succeeded. MC coefficient 1 is
therefore rejected as a stability improvement under this protocol and is not
promoted into the production defaults.

The new main run completed on 13 November 2026, from 14:54:37.648603 to
15:00:29.118208 UTC, taking about 5 minutes 51 seconds (22:54 to 23:00 in
Asia/Shanghai). The recorded dirty-worktree flag came from the existing
untracked report, not a production-code change. The 52 production/configuration
source hashes and inherited source chain remained unchanged. The initial full
state and all 117 initial Q vectors matched the shared TD-2000 source.

The control was reused from the verified attempt-016 baseline; its 8000 updates
and 87 episodes were not repeated as new main work. Its complete checkpoint
hashes, original attempt-014 reproduction and recorded inputs were checked
before and after the treatment. All supervised coefficients remained 1.
At the first shared state, TD loss was 0.079081811 and margin loss was
0.063221075; the sole added term was MC loss 0.099846750. The combined loss
increased by that same amount.

### Physical retention and failures

| Measure | Historical control, MC weight 0 | New treatment, MC weight 1 |
| --- | ---: | ---: |
| All-scene successes, TD 2001--2010 | 9/10 | 8/10 |
| Individual scene successes, TD 2001--2010 | 29/30 | 28/30 |
| All-scene successes, all scheduled checkpoints | 27/29 | 23/29 |
| Individual scene successes, all scheduled checkpoints | 85/87 | 81/87 |
| Scene 10000 successes | 29/29 | 29/29 |
| Scene 10001 successes | 27/29 | 24/29 |
| Scene 10002 successes | 29/29 | 28/29 |
| Final TD-10000 scene successes | 2/3 | 3/3 |
| Main policy actions | 3711, reused | 4347, new |

The treatment repaired scene 10001 at TD 10000, while both policies failed
that scene at TD 2009. Five newly regressed scene/checkpoint pairs occurred
at 2008, 2201, 3000 and 4500 in scene 10001, and at 8500 in scene 10002.
Every failure exhausted the 200-action budget without damage. All eight
compared failures are retained below. Q values come from the verified common
trajectory prefix at the first actual reference-action divergence; the gap is
`Q(demo) - max(Q(other))`.

| Arm | TD update | Scene | First divergent step | Demonstrated / chosen action | Q(demo) / Q(chosen) | Signed gap | Episode return |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Historical control | 2009 | 10001 | 21 | `north_small` / `north_large` | 84.930313 / 84.947029 | -0.016716 | 2.479857 |
| Historical control | 10000 | 10001 | 21 | `north_small` / `north_large` | 84.809303 / 84.869194 | -0.059891 | 2.479857 |
| Treatment | 2008 | 10001 | 21 | `north_small` / `north_large` | 85.207275 / 85.309448 | -0.102173 | 2.479857 |
| Treatment | 2009 | 10001 | 21 | `north_small` / `north_large` | 85.266975 / 85.340820 | -0.073845 | 2.479857 |
| Treatment | 2201 | 10001 | 6 | `south_small` / `east_small` | 75.509834 / 75.563530 | -0.053696 | -104.706068 |
| Treatment | 3000 | 10001 | 6 | `south_small` / `east_small` | 75.718117 / 75.737076 | -0.018959 | -104.640333 |
| Treatment | 4500 | 10001 | 6 | `south_small` / `east_small` | 75.635536 / 75.637680 | -0.002144 | -104.616856 |
| Treatment | 8500 | 10002 | 37 | `south_small` / `east_small` | 96.887321 / 96.916473 | -0.029152 | 7.516525 |

The two treatment failures first diverging at step 21 each had 178 blocked
and 177 stationary actions. The three first diverging at step 6 each had
171 blocked and 170 stationary actions. Scene 10002's TD-8500 failure had
no recorded blocked actions but 157 stationary actions. No action correction,
planner fallback or checkpoint rollback was used. First divergence locates
departure from the demonstrated trajectory; no isolated-action rescue was
performed to establish causality here. In particular, scene 10002's same
step-37 action deviation also occurs at successful checkpoints, so the first
deviation alone does not explain its failed continuation.

Scene 10001 recovered at the next sampled checkpoints TD 2010, 2500, 3500
and 5000 after its respective failure groups. Scene 10002 recovered at TD 9000.
These are measured recovery checkpoints, not inferred recovery times between
evaluations. The final treatment episode lengths were 35, 41 and 41 actions.

### Dense action-ranking diagnostics

All seven fixed probes refer to scene 10001. Negative counts below cover
the 8000 new update indices, TD 2001--10000. Minima include the restored
TD-2000 boundary, and final gaps are measured at TD 10000.

| Reference step | Negative counts, control / treatment | Minimum gap, control | Minimum gap, treatment | Final gap, control | Final gap, treatment |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 394 / 1441 | -0.323555 | -0.323463 | 0.255623 | 0.216263 |
| 18 | 0 / 3 | 0.105278 | -0.195732 | 0.700241 | 0.981148 |
| 20 | 0 / 0 | 0.520699 | 0.326401 | 0.689560 | 0.833679 |
| 21 | 627 / 244 | -0.223328 | -0.317703 | -0.059891 | 0.173325 |
| 22 | 0 / 0 | 0.224068 | 0.193466 | 0.807518 | 0.638245 |
| 23 | 0 / 0 | 0.186195 | 0.334831 | 0.922073 | 0.993439 |
| 24, control state | 102 / 29 | -0.211525 | -0.136131 | 0.802681 | 0.881287 |

Steps 20, 22 and 23 retained the demonstrated greedy action at every probe
in both arms. Step 21 improved in negative-update count but still reversed
244 times across 174 intervals. Its first reversal was at TD 2008 and its
minimum was -0.317703 at TD 2161. Of its 174 intervals, 173 fell entirely
between physical checks; the remaining interval included TD 2008 and 2009.
The historical control had 428 step-21 negative intervals, 426 entirely
between physical checks.

Step 6 became less stable: its negative-update count rose from 394 to 1441,
spanning 855 treatment intervals versus 267 control intervals. The first new
reversal occurred at TD 2018, with the minimum -0.323463 at TD 2158. Of the
855 intervals, 852 were entirely between physical checks; the others included
the failures at 2201, 3000 and 4500. Step 18 acquired three negative updates
in two intervals, both between checks, with minimum -0.195732 at TD 2152.

At step 24, 29 negative new-update gaps formed 21 treatment intervals;
including the initially negative boundary gives 30 records. The corresponding
control counts were 102 new-update gaps, 68 intervals and 103 records including
the boundary. There were no zero-gap ties at any of the seven states.
Every negative interval and the complete compact gap/action series are retained
in the analysis evidence.

These rankings do not represent additional failed episodes. Later reference
states may be unvisited after an actual trajectory diverges. The fixed
seven-state set does not monitor scene 10002's step 37 at every update;
its failure was captured by the three-scene physical evaluations and analyzed
at the saved checkpoint. The probe set was not expanded after observing it.

### Calibration, loss components and optimizer behavior

| Diagnostic | Shared TD-2000 start | Historical control TD 10000 | Treatment TD 10000 |
| --- | ---: | ---: | ---: |
| Full-batch demonstration agreement | 116/117 | 115/117 | 117/117 |
| Unweighted demonstration hinge loss | 0.063221 | 0.044419 | 0.045554 |
| Actual demonstration-return Q RMSE | 0.461398 | 0.499196 | 0.222843 |

The anchor achieved its direct value-fitting aim: final return RMSE decreased
to 0.222843. It did not achieve uninterrupted sampled retention. Its final
hinge loss was slightly higher than the control's despite complete final
action agreement. The agreement above is measured after the update, separate
from the raw log's pre-update accuracy.

Across 8000 updates, mean pre-update TD Huber loss decreased from the control's
0.075954 to 0.049896, while mean unweighted margin loss increased from 0.049829
to 0.061889. The treatment's mean MC Huber loss was 0.034616. Its final
post-update components were TD 0.042219, margin 0.045554 and MC 0.024830,
summing to 0.112603. Combined losses from the two objectives are not used as
directly comparable performance measures. The historical control did not
record per-update MC losses; missing values are not replaced with zero.

For the treatment, RMSE between frozen pre-update Bellman targets and fixed
MC targets decreased from 0.455334 at TD 2001 to 0.205072 at TD 10000,
with mean 0.261129 across updates. The corresponding signed mean difference
changed from -0.173887 to 0.022528, and maximum absolute difference from
1.888123 to 0.784584. These use the same pre-update targets as the recorded
TD losses and differ from post-update policy-to-return RMSE. Both target-network
and online-Q trajectories evolve after the shared starting state.

All 8000 new updates changed parameters, with 40 ordinary target
synchronizations on the original clock. Mean pre-clipping gradient norm
increased from 17.855495 to 23.791972. Clipping was active on 7888/8000
treatment updates, versus 7149/8000 control updates. Mean parameter-change
norm was 0.001858 for the treatment and 0.001959 for the control; final
displacement from the shared resume state was 2.107225 and 2.266949,
respectively. Lower return error did not arise from freezing the policy,
and the unchanged learning rate did not imply unchanged optimizer behavior.

This fixed-coefficient result shows that closer agreement with demonstration
returns need not preserve action ordering or successful execution throughout
TD learning. It does not establish that value drift caused the earlier
failures, that all MC coefficients must fail, or that the demonstrations are
optimal. A useful next diagnostic would compare TD, margin and MC gradient
directions and their effects on fragile action gaps, including clipping and
Adam state. That is a proposed follow-up, not an experiment performed here.
Additional layouts and held-out scenes remain necessary for generalization
claims; the current evidence is one initialization and three known scenes.

### Verification, accounting and retained evidence

New main work was **8000 optimizer updates and 87 evaluation episodes with
4347 policy actions**. The shared 42000 inherited updates were not rerun.
The historical control's 8000 updates, 87 episodes and 3711 actions are
separate reused evidence. The new branch ended at TD 10000, 50000 cumulative
updates and Adam step 30000. Its 8001 dense records contained 56007 state
forward passes with no environment actions. No new online training interactions,
demonstration collection or held-out scene evaluations were introduced.

Independent verification **passed**. Separately implemented control and
treatment objectives reproduced TD 2001--2201, adding **402 verification
updates**. Logs, probes and saved full states matched exactly across that
interval, including the target synchronization at 2200. Reloading the five
fixed treatment checkpoints reproduced all **15 verification episodes with
903 policy actions**; no control episodes were rerun. Later hashes, counters,
finite diagnostics and target lineage were audited without claiming numerical
replay of all main updates.

The independent read-only audit checked all 16002 compared seven-state dense
records, all 58 checkpoint records and 150 hashes: 136 model/source files
and 14 raw/configuration files. Historical records remain labeled as reused.
This audit added no resets, actions or optimizer updates. Both figures passed
visual inspection. The archive contains 21 evidence files plus its manifest,
22 files totaling 7471166 bytes. All archived hashes, byte counts and source
copies passed validation. It retains 170 update samples and 171 full-Q samples,
alongside complete paired compact dense series and every physical outcome.

Evidence is in [`017_mc_anchor/evidence`](017_mc_anchor/evidence), including
the [summary](017_mc_anchor/evidence/summary.json),
[accounting](017_mc_anchor/evidence/accounting.json),
[historical-control reference](017_mc_anchor/evidence/baseline_reference.json),
[provenance](017_mc_anchor/evidence/provenance.json),
[independent verification](017_mc_anchor/evidence/reload_checks.json),
[Q analysis](017_mc_anchor/evidence/q_analysis.json), and complete dense series
for the [control](017_mc_anchor/evidence/q_analysis_baseline_dense.json) and
[treatment](017_mc_anchor/evidence/q_analysis_treatment_dense.json).
The [outcome figure](017_mc_anchor/evidence/mc_anchor_outcomes.png) and
[neighbor-gap figure](017_mc_anchor/evidence/neighbor_q_gaps.png) show the
full prescribed comparison, including all sampled failures.

Full new logs and checkpoint binaries remain locally under
`results/training/mc_anchor_20261113`. Historical control binaries remain in
the referenced attempt-016 run, and the original ordered demonstration tensor
remains in attempt 011, identified by `input_manifest.original_data_run`.
The earlier [attempt-016 evidence](016_local_supervision/evidence) remains
unchanged; historical raw evidence is not duplicated in the new archive apart
from derived comparison products. Recorded hashes and coverage distinguish
reused evidence from newly executed work.
