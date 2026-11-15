# Attempt 029: broader data does not improve holdout success and loses old skills

Completed on 15 November 2026. Expanding the fixed demonstration set from three
to 33 scenes, while continuing the same TD procedure for the same number of
updates, produces **4/100 holdout successes in both arms**. Original-scene final
success falls from 3/3 in the matched control to 1/3 in the expanded arm. The
expanded arm succeeds on only 2/30 added training scenes. This attempt does not
meet the combined goal of improving generalization while retaining the three
original tasks, and the expanded candidate is not promoted over the prior policy.

The [protocol](../029_training_coverage.md), runner, objective and tests were
committed and pushed before execution in `ca63e5f`. Both arms start from the
same full attempt-027 TD-10000 snapshot, including inherited Adam state. This
is a test of adding demonstrations during TD continuation, not of rerunning
supervised pretraining and return calibration on the enlarged dataset.

## What changed and what stayed fixed

Control retains the original 117 transitions from seeds 10000--10002. Expanded
training appends 1323 physically executed, successful planner transitions from
30 predeclared seeds, 290000--290029: 1440 rows across 33 scenes. Every requested
collection succeeds; no seed is substituted. Original rows remain byte-equivalent
and in their original order. Privileged planner demonstrations are training data,
not autonomous policy successes.

Both arms perform 8000 updates, absolute TD 10001--18000, with architecture
341-256-256-8, learning rate 0.0001, gamma 0.99, TD coefficient 0.5, supervision
coefficient 1, margin 0.8, gradient clip 10 and target synchronization interval
200. Initial online/target parameters, Adam moments/settings/clocks and RNG
match exactly. The old selected rows 40 and 55 retain weight 2, all other rows
have weight 1, and the separate north_large recovery observation retains its
coefficient 1/117. No MC constraint, extra calibration, scheduler or early stop
is added. The loss remains:

```text
0.5 * mean_TD_Huber(N transitions)
+ sum(weighted_demonstration_margin_terms) / N
+ recovery_margin / 117
```

The full-data mean rule is unchanged, but N grows from 117 to 1440. Consequently,
original rows make up 8.125% of the expanded transition set and each old margin
term has 117/1440 of its former normalization coefficient. This is an explicit
consequence of the data expansion, not a hidden old-scene oversampling scheme.
The separate recovery coefficient is not diluted. The configured batch_size=64
remains unused in this full-data diagnostic. Budgets match optimizer updates,
not computation or total row exposures: control has 936000 transition exposures
and expanded training 11520000, excluding gates and independent verification.

## Paired, previously unused holdout scenes

Both final models are saved and hash-frozen before accessing seeds
390000--390099. These seeds are disjoint from every training scene and from
attempt 028's 280000--280099 test set. Both arms use the same embedded environment,
greedy inference, zero exploration, no correction and the same 200-action budget.
All 100 scenes initialize in both arms; their initial encoded states, geometry
and reset certifications match pairwise and differ from training starts.

| Final holdout measurement | Three-scene control | Expanded 33-scene arm |
| --- | ---: | ---: |
| Success | 4/100 | 4/100 |
| Damage termination | 49/100 | 53/100 |
| Budget failure | 47/100 | 43/100 |
| Persistent blocked-stall category | 42/100 | 42/100 |
| Other budget-failure category | 5/100 | 1/100 |
| Mean return | -50.852610 | -43.988525 |
| Mean episode length | 116.52 | 111.91 |
| Mean length among successful episodes | 38.75 | 49.25 |

Both descriptive 95% Wilson intervals for success are 1.57%--9.84%. Equal
observed rates do not prove population-level equivalence; this is one shared
initialization and a finite paired scene set. The less-negative expanded mean
return is retained as a favorable metric, but it is not accompanied by higher
task success or lower damage. The successful-episode length means concern
mostly different successful scenes, so they are not a paired efficiency test.

The paired table explains the unchanged success count:

| Paired outcome | Count | Successful scene seeds |
| --- | ---: | --- |
| Both succeed | 1 | 390063 |
| Control only | 3 | 390005, 390028, 390032 |
| Expanded only | 3 | 390062, 390071, 390087 |
| Neither succeeds | 93 | See complete per-scene records |

There are three gains and three losses, rather than a net success improvement.
Attempt 028's 1/100 result used different seeds and an earlier frozen model;
it is not the matched baseline for this experiment.

## Original-scene retention

| Relative continuation update | Control successes | Expanded successes |
| --- | ---: | ---: |
| 0 | 3/3 | 3/3 |
| 1 | 3/3 | 3/3 |
| 10 | 3/3 | 0/3 |
| 100 | 3/3 | 0/3 |
| 500 | 3/3 | 0/3 |
| 1000 | 3/3 | 1/3 |
| 2000 | 3/3 | 0/3 |
| 4000 | 3/3 | 0/3 |
| 6000 | 3/3 | 1/3 |
| 8000 | 3/3 | 1/3 |

Control succeeds in 30/30 sampled original-scene episodes; expanded training
succeeds in 9/30, including the common unchanged starting model. These are
checkpoint evaluations, not every intermediate model.

At the final expanded model, scene 10000 exhausts 200 actions with 152 blocked
and 150 stationary steps. Scene 10002 also exhausts 200 actions, with 194 blocked
and 193 stationary steps. Neither terminates with damage. Scene 10001 succeeds
in 45 steps with return 104.031448 and peak deformation 0.243103; the control
needs 41 steps, return 107.585050 and zero peak deformation. Thus even the
remaining successful original scene does not reproduce its former trajectory.

## Acquisition versus generalization

Only added training scenes 290010 and 290021 succeed under the final expanded
policy: 2/30, with 16 damage and 12 budget failures. Together with the one
retained original task, final training-scene success is 3/33. The problem is
therefore not restricted to held-out inputs.

Final original-label agreement is 117/117 for control and 85/117 for expanded
training. On the 1323 added labels it is 493/1323 and 910/1323, respectively;
the control only observes these labels for diagnostics, never for updates.
Expanded training improves new-label fit, but 995/1440 total agreement is not
sufficient for reliable closed-loop task completion. Both final models choose
north_large at the retained recovery input, with positive Q gaps 1.506073 and
1.784531. That local preference does not guarantee the full original tasks.

The data-mean normalization reduces old samples' relative contribution, while
new states enter TD without their own supervised/calibration stages. These are
plausible contributors to interference and incomplete acquisition. This attempt
does not isolate their individual causal effects and does not establish that
broader training data is inherently harmful. It establishes that this particular
fixed-budget continuation recipe is insufficient.

## Verification, resource accounting and artifacts

Four disposable gate updates verify exact equality to the old objective at
N=117 and to an independent expanded objective implementation. Independent
replays cover the first and last ten updates of each arm (40 total), matching
logs and applicable complete snapshots. All 16000 main updates have nonzero
parameter changes. Each arm has 40 target synchronizations and ends with Adam
internal step 38000; optimizer settings and DQN configuration are unchanged.

Eighteen final-model reload episodes through the normal evaluator reproduce
all selected original/holdout metrics and reset certifications. Complete state
checks ensure evaluation does not update weights, optimizer state or agent/Torch
RNG. All 290 main evaluation traces are independently audited. Thirteen focused
tests pass and Ruff passes. All bound source and frozen candidate hashes match.

| New work | Resets | Executed actions | Copied certification actions |
| --- | ---: | ---: | ---: |
| Added demonstration collection | 30 | 1323 | 1323 |
| Original-scene retention checks | 60 | 4451 | 2340 |
| Expanded-arm acquisition checks | 30 | 3199 | 1323 |
| Paired holdout evaluations | 200 | 22843 | 8310 |
| Normal-evaluator reloads | 18 | 2198 | 750 |
| Total | 338 | 34014 | 14046 |

Total simulator work is 48060 actions. Of the executed actions, 1323 are
planner-supplied collection actions; no policy evaluation action is forced.
There is no online exploratory training in this attempt. Optimizer work totals
16000 main + 4 gates + 40 independent = 16044 updates. Unit-test work is excluded.
The original 117 rows are reused data and are not counted as new collection.

The archive contains all update logs, the full expanded demonstration arrays,
all evaluation traces and failures, hashes, matched outcomes and accounting:
[analysis](evidence/analysis.json), [per-scene CSV](evidence/per_scene.csv),
[diagnoses](evidence/diagnoses.json), [dataset](evidence/dataset.json),
[accounting](evidence/accounting.json), [archive manifest](evidence/archive_manifest.json).
Full resume snapshots remain local; both predeclared final inference candidates
are tracked with outcome metadata. Neither replaces step6_repair_027.pt.

From the repository root in the configured environment:

```bash
python scripts/evaluate_dqn.py --checkpoint models/coverage_029_control.pt --episodes 100 --seed 390000
python scripts/evaluate_dqn.py --checkpoint models/coverage_029_expanded.pt --episodes 100 --seed 390000
python scripts/evaluate_dqn.py --checkpoint models/coverage_029_expanded.pt --episodes 3 --seed 10000
```

These repeat now-observed scenes, not a fresh blind assessment. The archival
training command was `python docs/experiments/029_training_coverage/run_coverage.py`,
followed by `python docs/experiments/029_training_coverage/analyze_coverage.py`.
It requires the hash-bound historical full snapshot and original demonstration
file, and refuses to overwrite prior outputs. After these results inform later
design choices, a future tuned candidate needs another disjoint test set.

This attempt leaves the original repaired policy intact and records the failed
retention/generalization objective rather than selecting a convenient checkpoint
or quietly changing training parameters.
