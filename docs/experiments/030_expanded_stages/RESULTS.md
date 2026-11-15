# Attempt 030: staged training across 33 scenes

## Outcome

The specified supervised -> actual-return calibration -> Adam-preserving TD
procedure ran to completion with the network and existing training settings
unchanged. It partially restores the original scenes, but does not restore all
three or retain all acquired training-scene successes. On the fresh paired test
set, success changes from 9/100 to 13/100, with 11 gained and 7 lost successes.
This is a small observed improvement in one run, not evidence of reliable
generalization.

The [protocol](../030_expanded_stages.md), implementation and portable source
snapshot were committed as `6ea5d8e` before execution. Verified training-stage
results were separately committed as `d1152b3` while holdout evaluation was still
running. Execution lasted from 2026-11-15 09:37:44 UTC to 13:29:58 UTC.
No training budgets, weights, scene sets or selected endpoint changed after
observing results.

## Training and original-scene recovery

The source is the degraded attempt-029 expanded endpoint, not newly initialized
weights. Reuse exactly 1440 complete demonstration transitions from seeds
10000--10002 and 290000--290029, with the same separate recovery label. No new
demonstration collection, online training interaction or action correction occurs.
The architecture remains 341-256-256-8 with ReLU; learning rate is 0.0001.

Run 20000 supervised updates, 20000 calibration updates and 10000 TD updates.
The two pre-TD stages each start with fresh Adam while preserving its settings.
Calibration fits fixed complete-episode discounted returns (gamma 0.99), with
no bootstrapping across episode boundaries. The existing focused supervision
and recovery term are present throughout. TD retains coefficient 0.5 and has
no MC term. Original rows remain 117/1440 of the full-data mean; no rebalancing.

| Endpoint | Original scenes | Added training scenes | All training scenes | Action agreement | Return MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attempt-029 source | 1/3 | 2/30 | 3/33 | 995/1440 | 4.2916 |
| Supervised 20000 | 0/3 | 4/30 | 4/33 | 1170/1440 | 6.2497 |
| Calibrated 20000 | 1/3 | 8/30 | 9/33 | 1185/1440 | 0.7745 |
| TD 10000 | 2/3 | 5/30 | 7/33 | 1227/1440 | 2.0482 |

The source succeeds on 10001, whereas the calibrated endpoint succeeds on 10000.
Final TD succeeds on 10000 in 38 steps (return 107.589934) and 10001 in 47 steps
(return 107.282922), both without blocked or stationary actions. Scene 10002
fails from damage at step 58, return -62.168230 and peak deformation 1.00049285.
Thus **the original three-scene capability is only partially restored**.

All eleven sampled TD checkpoints retain success on 10000. Scene 10001 succeeds
only at the sampled TD 8000 and 10000 endpoints; 10002 succeeds at none of them.
Unexamined intermediate policies are not covered by these checks.

From calibration to final TD, five training successes remain, four are lost
(290006, 290013, 290023, 290026), and two are gained (10001, 290017).
Total training success therefore falls from 9/33 to 7/33 even while original
scene success improves. Action agreement and calibrated value error do not
substitute for complete autonomous rollouts. Return MAE is relative to the
demonstrated returns, not to an independently known optimal Q function.
See [stage results](TRAINING_RESULTS.md) and
[paired training transitions](evidence/training_transition_comparison.json).

## Fresh paired generalization test

Both final inference files were hash-frozen before testing. Exactly 100 new seeds
490000--490099 were evaluated with each policy. All 100 scenes were valid in
both arms; initial geometry, encoded input and reset-certification records
match pairwise. Initial hashes are distinct from training and previously
observed test sets. No failures were replaced and no holdout trajectory entered
training. The model is the fixed TD-10000 endpoint, not a selected best checkpoint.

| Metric | Frozen attempt-029 expanded source | Staged TD endpoint |
| --- | ---: | ---: |
| Successes | 9/100 | 13/100 |
| Wilson 95% success interval | 4.81%--16.23% | 7.76%--20.98% |
| Damage failures | 47 | 50 |
| Budget failures with sustained blocking | 39 | 33 |
| Other budget failures | 5 | 4 |
| Mean episode return | -42.449148 | -34.774916 |
| Mean episode steps | 115.10 | 104.81 |
| Mean steps among successful episodes | 39.11 | 59.23 |

The paired outcomes are two successes with both policies, eleven with staged
training only, seven with the source only, and eighty with neither. The observed
success difference is +4 percentage points. A clearly marked **post-hoc**
two-sided exact test on the 18 discordant pairs gives p = 0.480682; this run
does not provide strong evidence for a reproducible generalization improvement.
See [post-hoc statistics and seed lists](evidence/posthoc_statistics.json).

Damage failures increase despite the small success-count increase. Mean episode
length includes early failures, and the successful subsets differ between
policies, so these length statistics do not establish greater efficiency.
The category `budget_while_moving` means the trailing-stall criterion was not
met; it does not imply productive motion or absence of blocking.

This is one initialization/optimization history and one same-generator holdout
set, not a multi-seed or out-of-distribution study. The baseline receives no
matched extra 50000-update budget, so the comparison concerns the whole added
staged procedure, not isolated causality of calibration or optimizer retention.
The simulator's existing route-certification rule remains unchanged; these are
policy results on certified feasible scenes, not an unbiased planner benchmark.

## State integrity and verification

Adam first moments, second moments, per-parameter step counters and all settings
are exactly identical immediately before and after calibration-to-TD transition.
The optimizer hash is unchanged:
`4dd8e318100f6b6841a68c4fd03db164937530ea422596214b702431b496d00d`.
Only the target-sync phase counter resets, with target weights synchronized to
online. The Adam clock advances from 20000 to 30000; TD performs 50 target syncs.

All 56 pre-execution tests passed. The independent update oracle reproduces the
first and last ten updates of each of the three phases, including complete
model, target, optimizer, counter and RNG states. All 344 main rollout traces
pass action-argmax, state-chain, reward, outcome and file-hash checks. Fourteen
normal-evaluator reload episodes reproduce metrics and certification records.
Bound source files and frozen inference policies remain unchanged.

| Resource | Count |
| --- | ---: |
| Main training updates | 50000 |
| Disposable independent verification updates | 60 |
| Main rollout episodes | 344 |
| Normal-evaluator reload episodes | 14 |
| Total environment resets | 358 |
| Executed policy actions, including reloads | 40332 |
| Separate reset-certification simulation actions | 14922 |
| Total simulated actions in this attempt | 55254 |
| New demonstration collection / online training actions | 0 / 0 |
| Forced evaluation actions | 0 |

Unit-test operations and the inherited model's earlier training/collection work
are excluded from these new-run counts. The 1440 reused demonstration rows and
their collection provenance remain in attempt 029.

## Reproducible artifacts

- [Portable source full snapshot](inputs/source_state.pt) reproduces the initial
  model, target, Adam and RNG state without ignored historical output folders.
- [Archive manifest](evidence/archive_manifest.json) indexes all raw phase and
  holdout traces, all 50000 update-log rows, actual return targets, checkpoint
  hashes, provenance, failures, accounting and checks.
- [Full analysis](evidence/analysis.json), [per-scene metrics](evidence/per_scene.csv)
  and [accounting](evidence/accounting.json) contain the machine-readable results.
- Three phase inference endpoints are tracked under `models/`:
  `stages_030_supervised.pt`, `stages_030_calibration.pt` and `stages_030_td.pt`.
  Their adjacent JSON files record hashes and scope. The pre-TD endpoints were
  not evaluated on the fresh holdout set. Intermediate resume snapshots remain
  in the local run directory.

To evaluate the exported final endpoint using the normal evaluator:

```bash
python scripts/evaluate_dqn.py --checkpoint models/stages_030_td.pt --episodes 3 --seed 10000
python scripts/evaluate_dqn.py --checkpoint models/stages_030_td.pt --episodes 100 --seed 490000
```

These commands replay already observed evaluation scenes; they do not create a
new independent test. The earlier policy, reports and submission archives were
not replaced. All new code and comments are in English.

## Implication for the next attempt

The larger dataset is still incompletely acquired: the best predeclared training
endpoint here succeeds on only 9/33 scenes. Before another long continuation,
diagnose the remaining 10002 failure and the four calibration-to-TD losses on
training scenes. A subsequent remedy should change one training factor at a
time and use a newly reserved holdout bank after its policy is fixed. Neither
stable three-scene recovery nor dependable broad-scene behavior has been
established by this attempt.
