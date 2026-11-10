# Attempt 008: lower learning rates during TD adaptation

## Protocol recorded before execution

Attempt 007 reproduced the learned supervised route, but TD+margin at learning
rate 0.0003 lost it after one update, recovered temporarily and then failed again.
The supervised continuation control remained successful. Test whether reducing
the TD-phase step size preserves behavior over the same adaptation horizon.

Use the **verified attempt-007 supervised checkpoint after 20000 updates**, and
its identical frozen 35-transition demonstration dataset. Check the reference
source, checkpoint, encoded data, parameter and reproduction hashes. This is an
explicit transfer of pretrained weights followed by a fresh optimizer, not a
resume of optimizer state. No additional demonstration collection or supervised
pretraining is performed; their original costs remain in attempt 007.

Two new branches use **TD+margin** (margin 0.8, weight 1), with phase learning
rates **0.00003** and **0.000003**. The already-recorded 007 TD+margin branch at
0.0003 is the historical baseline, not a newly executed branch. Its runner and
production source hashes must still match. Everything except phase learning
rate remains fixed: architecture, inputs, environment, reward, data, full-batch
updates, optimizer type, clipping and target synchronization every 200 updates.

Apply the new rate before constructing fresh Adam; record the resolved rate in
each branch's settings and saved model. Synchronize the target from pretrained
online weights and start the phase counter at zero. Keep the inherited 20000
supervised updates separate from new updates. Require phase-zero weights,
whole-dataset action metrics and the greedy route to exactly match the 007
supervised baseline before updating. Use the unchanged attempt-007 phase runner.

Each new branch receives exactly **2000 TD+margin updates**, evaluated at
**0, 1, 10, 50, 100, 200, 500, 1000 and 2000**. Retain all snapshots, including
temporary failures/recoveries. Success at every sampled checkpoint is the stated
retention criterion; it does not establish behavior at unobserved updates.
There is no adaptive tuning, early stopping or selection of a successful
intermediate checkpoint. Verify actual parameter changes and update counts so
retention is not mistaken for simply freezing the policy.

The new budget is **4000 optimizer updates**, zero new collection actions, zero
online exploration steps and zero held-out scenes. Separately count scheduled
neural evaluations and post-run reload verification, excluding reset feasibility
probes. Reusing the existing dataset does not make these policies independent
of privileged planner-generated training data.

## Reproduction

With the verified reference run available and a new output directory:

```powershell
python docs/experiments/008_td_learning_rate/run_learning_rate_retention.py --reference-run results/training/staged_retention_20261111 --output-dir results/training/td_learning_rate_20261111
```

The input manifest identifies the exact reference and its immutable hashes.
Both new rates, all failure cases and the previously recorded higher-rate
baseline will remain visible in the result comparison. This experiment tests
retention on one known scene over 2000 updates, not convergence, optimal learning
rate, improvement on new tasks, online learning or multi-scene generalization.

## Outcomes

Executed from commit `12ae932` on **11 November 2026, 02:31:27--02:33:33
Asia/Shanghai**. Evidence timestamps use UTC. Both lower-rate branches started
from the exact verified supervised weights, with the same 35 training states,
fresh Adam state, synchronized targets and identical phase-zero action metrics
and 35-step successful rollout.

| TD+margin learning rate | First sampled failed checkpoint | Final action agreement | Final rollout after 2000 updates |
| --- | --- | --- | --- |
| 0.0003, historical attempt 007 | 1 | 32/35 (91.4%) | Damage after 55 steps |
| 0.00003, new branch | 500 | 34/35 (97.1%) | Budget exhausted at 200 steps |
| 0.000003, new branch | 500 | 33/35 (94.3%) | Budget exhausted at 200 steps |

Both lower rates succeeded at checkpoints **0, 1, 10, 50, 100 and 200**. Both
failed at **500, 1000 and 2000**. The 0.00003 branch damaged a branch after 141
steps at checkpoint 500, then timed out at the later checkpoints. The 0.000003
branch timed out at all three later checkpoints. Final returns were -8.2960 and
-8.2179, respectively. Both final policies first deviated from the reference at
action 5, choosing north-small instead of east-small.

Reducing the rate prevented the immediate one-update collapse seen in 007, but
**neither rate preserved success throughout the 2000-update comparison**. The
first observed failure shifted to checkpoint 500, so the actual onset between
200 and 500 is not known. This experiment does not establish convergence,
stable retention beyond the observed horizon or an optimal rate. In particular,
97.1% demonstration-action agreement still accompanied a failed closed-loop
episode; agreement alone is not a sufficient model-selection criterion.

The updates were real: final parameter-vector L2 changes from the supervised
starting model were **1.780151** at 0.00003 and **0.369576** at 0.000003. Both
branches performed all 2000 updates and ten periodic target synchronizations.
The smaller-rate policies were not frozen or replaced with the pretrained model.

New accounting: **4000 TD+margin optimizer updates**, **1561 scheduled neural
evaluation actions**, and **400 additional actions** for independent final-model
reload verification. New demonstration collection, online exploration and
pretraining counts are all **zero**. The original 20000 pretraining updates and
35 collected demonstration actions are explicitly reused from attempt 007.
Counts exclude reset feasibility probes. No held-out scene was accessed.

## Interpretation and next question

Supervised acquisition followed by TD adaptation remains unstable on the known
route under all tested rates. Lowering the step size alone did not resolve this
over the matched adaptation budget. The continuing-supervision control in 007
still identifies TD introduction as the experimental difference associated with
the loss, but these comparisons do not identify its sole internal cause.

The next useful hypothesis is value calibration: action-margin pretraining
learns action preferences without forcing scores to equal discounted returns.
Test supervised return-target calibration before bootstrapping, while measuring
whether the existing action choices survive that calibration. Keep it a separate
controlled attempt; no such calibration has been implemented or shown to solve
the problem in the present records. Broader scene exposure and online replay
should be assessed separately from this fixed-data retention diagnostic.

## Verification and retained evidence

The shared learner and phase runner passed **235 tests** in attempt 007; they
were unchanged for 008. The new runner and analysis helpers passed Ruff.
Verification checked all 49 launch source hashes, 11 reference-input hashes,
18 saved checkpoints and their actual learning-rate configurations, all 4000
update records, target counts and exact phase-zero behavior. Both final policies
were independently reloaded and reproduced their full fitting metrics and
200-step rollout traces exactly.

The first reload-checker invocation stopped before inference because its field
lookup used an earlier draft protocol key (`phase_updates_per_arm`). The helper
was corrected to the recorded `phase_updates`/`pretraining_updates` schema and
verification passed. That failed analysis invocation performed **zero actions
and zero optimizer updates** and did not change any training artifact. Preserve
this note rather than presenting the first verification invocation as successful.

Evidence: [figure](008_td_learning_rate/evidence/learning_rate_retention.png),
[summary](008_td_learning_rate/evidence/summary.json),
[input manifest](008_td_learning_rate/evidence/input_manifest.json),
[historical baseline](008_td_learning_rate/evidence/historical_td_margin.json),
[reload verification](008_td_learning_rate/evidence/reload_checks.json), and
[archive manifest](008_td_learning_rate/evidence/archive_manifest.json).
All scheduled observations are retained; update samples include the first ten,
every 50th and final update, with hashes of the complete local logs. Model
weights, encoded data and full logs remain in the ignored local results tree.
