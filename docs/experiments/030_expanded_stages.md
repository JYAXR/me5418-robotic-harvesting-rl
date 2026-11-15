# Attempt 030: staged recovery with thirty-three training scenes

## Protocol declared before execution: 15 November 2026

Attempt 029's expanded final policy succeeds on only one of the original three
scenes and two of thirty added training scenes. This attempt tests whether
explicit supervised fitting and actual-return calibration before TD can restore
those abilities. It is a fixed-budget staged retraining experiment, not a
matched-budget comparison of individual components.

Start from the exact attempt-029 expanded final full snapshot (relative update
8000, absolute TD 18000), SHA-256
`8890fea9630badb438c0e8a380f64c9f1dddfa7d5911361b0bdeb2835503b559`.
Archive this input under `030_expanded_stages/inputs/source_state.pt` so a fresh
checkout can reproduce the attempt without untracked historical training files.
Retain its network weights, configuration and RNG. Do not initialize a new policy.

Reuse the exact 1440 physically executed demonstration transitions from attempt
029: original seeds 10000--10002 followed by 290000--290029. Preserve row order,
all original 117 transitions and the separate existing north_large recovery
observation from attempt 027. No new demonstrations, scene selection, resampling,
online exploration or action overrides. All training losses are full-data means.
Old rows constitute 117/1440 of the main sample set; do not rebalance them.

Network 341-256-256-8 (ReLU), encoder, environment and action space are unchanged.
Use learning rate 0.0001 throughout, gamma 0.99, clip norm 10, target interval 200,
supervision coefficient 1, margin 0.8, existing row-40 and row-55 weights 2, other
row weights 1, and separate recovery supervision coefficient 1/117.
Define M as the weighted demonstration margin mean plus this recovery term.

1. Supervised fitting: exactly 20000 updates minimizing M. Reset Adam at the
   beginning while preserving all optimizer settings and network weights.
2. Actual-return calibration: exactly 20000 updates minimizing
   M + mean Huber(Q(s,a_demo), G). G is the recorded discounted reward sum up to
   each demonstration's own true terminal endpoint, computed separately for
   every episode with gamma 0.99. No bootstrapping or return flow across resets.
   Reset Adam at this boundary, as in the earlier staged recipe; keep learned
   weights. The separate recovery observation has only an action label and
   receives no invented MC target.
3. TD: exactly 10000 updates with the unchanged attempt-029 objective
   0.5 * mean TD Huber + M. No MC term remains during TD.
   Preserve every Adam first/second moment and per-parameter clock from
   calibration. Synchronize the target to online and reset only the target-sync
   phase counter. Final Adam clock must be 30000. Sync target every 200 TD updates.

Compared with historical attempt 011, this uses the current learning rate,
focused supervision and recovery term, not the old uniform loss/0.0003 rate.
All three phase budgets are fixed in advance: no early stopping, budget extension,
best-checkpoint selection or response to holdout outcomes.

Evaluate original scenes at updates 0, 5000, 20000 in each pre-TD phase and at TD
0, 1, 10, 100, 500, 1000, 2000, 4000, 6000, 8000, 10000. At every phase endpoint
also evaluate the thirty added training scenes. Record action agreement on
original/new rows, MC value error, and selected/neighboring action margins.
Successful label fitting is not evidence of successful autonomous episodes.
Sampled checkpoint retention does not establish success at every update.

Freeze the predeclared final model before testing. Evaluate it and the unchanged
attempt-029 expanded policy on exactly the same fresh 100 seeds 490000--490099.
These are disjoint from training and previously observed test seeds
280000--280099 and 390000--390099. Record all requested seeds, including reset
errors, without replacement; compare initial geometry and encoded hashes.
Require held-out hashes to differ from training and all previous test hashes.
This paired comparison measures the total added staged-training procedure;
the source model receives no matched additional-update budget.
Do not reuse holdout trajectories as training or select a model from their scores.

Log every optimizer update, stage boundary, full-state checkpoint hash, autonomous
rollout and failure. Independently recompute return targets and replay the first
and last ten updates of each phase. Verify exact Adam preservation at the TD
boundary, unchanged source/configuration, and normal-evaluator reloads for the
three original scenes plus the first fresh-test seed of each observed category.
Count executed policy actions and reset-certification actions separately.
Historical demonstration-collection cost belongs to attempt 029; no new collection
occurs here. Unit-test optimizer operations are reported separately from the run.

## Commands

From the repository root in the configured environment:

```bash
python -m pytest tests/test_expanded_stages.py tests/test_training_coverage.py tests/test_optimizer_transition.py -q
python docs/experiments/030_expanded_stages/run_stages.py
python docs/experiments/030_expanded_stages/analyze_stages.py
```

The runner refuses to overwrite `results/training/expanded_stages_030_20261115`.
Full intermediate resume snapshots remain local; the source resume input, final
inference candidates, complete logs, traces and audits will be tracked in Git.
The previous policy, reports and submission archives are unchanged.
