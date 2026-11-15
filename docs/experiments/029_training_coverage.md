# Attempt 029: expand training coverage under a matched continuation budget

## Protocol declared before execution: 15 November 2026

Attempt 028 found only 1/100 unseen-scene successes for the repaired policy.
Continue the exact attempt-027 TD-10000 full snapshot in two paired arms.
Both retain online/target weights, Adam moments and internal clock, agent/Torch
RNG, architecture 341-256-256-8, learning rate 0.0001, gamma 0.99, TD weight 0.5,
margin 0.8, supervision weight 1, clipping 10, target interval 200, original
row-40/55 weights 2 and the same recovery observation/action/weight. No MC term,
new pretraining/calibration, scheduler, early stopping, action override or
validation-based model selection. Each arm performs exactly 8000 further updates,
absolute TD 10001--18000. The sole arm difference is training data coverage.

Control uses the unchanged 117 transitions from seeds 10000--10002. Expanded
training appends complete physically executed planner demonstrations from exactly
30 new seeds 290000--290029. Preserve every collection failure and do not substitute
seeds; abort training if any predeclared demonstration cannot be collected. Retain
all original transitions in order. No attempt-028 scenes (280000--280099) enter
training. The planner supplies privileged training data; record its rollout and
reset-certification work separately from policy evaluation.

Preserve the existing full-data mean rule for TD and original-row supervision.
For N total training transitions the objective is:

    0.5 * mean_TD_Huber(N) + sum(weighted_margin_terms(N)) / N
    + recovery_margin(north_large) / 117

Only the two existing selected rows have weight 2; all added rows have weight 1.
The separate recovery coefficient remains exactly 1/117. N grows with the data;
this is not a switch to mini-batching. Old demonstrations' fraction of the two
full-data means consequently decreases. Record N, per-scene row counts and that
fraction explicitly. Do not silently oversample old scenes or change coefficients
to rescue retention. The N=117 path must reproduce attempt-027 updates exactly.

Evaluate all three original scenes at relative updates 0, 1, 10, 100, 500, 1000,
2000, 4000, 6000 and 8000 in each arm; report every regression, not only the final
model. Track original/new label agreement and recovery choice at those boundaries.
Run all 30 added training scenes with the final expanded policy to distinguish
acquisition failure from holdout generalization failure.

Only after BOTH final snapshots are saved and hash-frozen, evaluate both on exactly
100 new test seeds 390000--390099 using unchanged environment settings and greedy
inference. Keep this set disjoint from all training seeds and attempt 028. Each
requested seed is retained regardless of result; reset errors are reported and
never replaced. Reuse attempt-028 full-trace instrumentation and failure categories.
Report paired success/failure outcomes, rates, descriptive Wilson intervals,
damage and budget exhaustion; do not infer improvement from unpaired 1/100 results
on attempt 028's different test set. No test outcome influences training or selection.

Use disposable exact control/independent-update gates before training. Independently
replay the first and last ten updates of each arm and match full saved state/logs.
Audit Adam clocks/target synchronization. Verify final inference weights against
full snapshots, then reload the final policies through the normal evaluator on
all three original scenes, the first five test seeds and the lowest seed of each
observed test outcome category (deduplicated). Count all new work. Archive all
outcomes and synchronize to Git even if coverage expansion fails. Preserve the
previous repaired model and reports; export both final candidates with their
measured limitations. This attempt tests fixed-data TD continuation, not a fresh
rerun of the full staged learning procedure. Results pending at protocol commit.
