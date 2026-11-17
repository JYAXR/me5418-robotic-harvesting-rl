# Attempt 008: at most two rounds of batch recovery aggregation

Branch: `experiment/your-method`. Parent: `ed2905e`.
This is a new bounded method, not another single-action or single-factor study.
Push the code and protocol before the first collection/training run. Publish
each completed round, including failures, before starting any second round.

## Fixed split and endpoint

- Training scenes: 10000, 10001, 10002 and 290000 through 290029 (33 total).
- Development validation: 590000 through 590029 (30 total), disjoint from all
  training scenes and the previously inspected 280000, 390000 and 490000 banks.
  Searches of prior experiment scripts and written protocols found no use of
  this validation bank. Its outcomes may be used to select between this bounded
  set of candidates; it is not a final unseen test bank.
- Final comparison with the teammate remains reserved for a separately agreed,
  untouched test bank. No new final test is run here.
- Start with `models/your_method_004_targeted.pt`, SHA256
  `ea42fd1a2d23317c9e15a960c58d90970a38edbcfb7724308a845ead92f6119f`.
  Its full source snapshot is attempt 004 `resume/targeted_state_14000.pt`, SHA256
  `ed32355abf23c463a523df6bf00d6a9708b727bcd15fc04f357eaadd6f68fd2c`.
- Network (341-256-256-8), encoder, scene generation, reward, dynamics and
  200-action task budget stay unchanged. No hand-selected action overrides are
  used during data-collection policy prefixes or autonomous evaluation.

## Recovery collection in each round

Freeze the incoming policy: the source in round 1, round-1 endpoint in round 2.
For each of the 33 scenes, execute a greedy prefix and query recovery before
actions 8, 24 and 40 when those nonterminal states are reached. Stop the prefix
after querying step 40; do not spend another full failure episode on collection.
Log any query skipped because the policy already terminated.

Each query clones the current simulated state, counters and RNG, then calls the
existing collision-free A* planner with current and resting branch shapes.
Execute that route in the cloned environment under the original remaining
budget and physical dynamics. Add all encoded state/action pairs from the route
only if its complete execution succeeds. Save unsuccessful routes and planner
failures too, with zero accepted labels. Failure to plan is not a proof of an
infeasible state. Check that the query never changes its parent simulation.

There are at most 99 recovery queries per round (198 across both), not one
manually chosen repair per failure. Later states beyond the scheduled prefixes
are not covered by this initial bounded method. Round 2 aggregates all accepted
round-1 and round-2 recovery samples. No validation observations/actions enter
training data; only the 33 training seeds are accepted by the collector.

## Supervised-only learning

Each round performs exactly 6000 Adam updates, maximum 12000 total, using
cross-entropy over the network's eight action outputs. There is no TD loss,
Bellman target, MC calibration or target-network update. The original TD clock
stays 14000; supervised updates are recorded separately. Outputs after this
stage serve as action scores and must not be described as calibrated Q values.

Learning rate is 0.0001, gradient clipping is 10, batch size is 256. Each batch
samples 128 rows from the original 1440 demonstrations plus the one legacy
verified recovery example, and 128 from accepted aggregated recovery examples,
with replacement. If recovery is empty, use 256 base rows and report that no
recovery data was obtained. All rows in each pool have equal sampling weight;
historical special weights on individual steps are not carried forward. Valid
alternative planner routes may produce different labels at similar states;
retain them and report data-fitting results rather than silently relabel them.

Reset Adam once at the start of round 1 because the objective has changed.
Preserve round-1 Adam moments, weights and sampling RNG into round 2. Seed the
sampling stream and Torch with 817008. Save each endpoint model and a resume
checkpoint containing optimizer and RNG state; verify inference reload weights.
Record every update loss and gradient norm, and data-fitting accuracy at fixed
updates 1, 1000, 3000 and 6000. No checkpoint selection within a round.

## Evaluation and stopping

Before training, evaluate the source on all 30 validation scenes. Reuse the
already audited source endpoint's 33-scene training evaluation (10/33, original
3/3) from attempt 004. Check collection reset states against those historical
scenes. After each round, evaluate all 33 training and 30 validation scenes
autonomously, recording success, damage, budget exhaustion, return, step count,
blocked/stationary steps and complete traces. Compare paired initial states.
Reset errors count as failures in the full denominator and are retained with
their error messages; their placeholder return is zero, reported with caveats.

Stop after round 1 only if all original three scenes succeed, at least 27/33
training scenes succeed, and validation achieves both at least 9/30 successes
and at least three more successes than the source. These are provisional
development targets, not proof of adequate final performance. Otherwise perform
the predeclared second round and stop regardless of its outcome. No third round,
learning-rate search, extra targeted correction or unplanned TD stage is allowed.

For a recommended retained candidate, require original 3/3 and training successes
at least the source's 10/33. Among eligible source/round endpoints, rank validation
successes first, then training successes, then lower total damage failures, then
the earlier endpoint. Retain all candidates and results, including ineligible
ones. If neither trained endpoint qualifies, retain the source and state the
method did not meet the retention condition. Do not present assisted recovery
or training-set accuracy as autonomous task success.

## Resources, reproducibility and verification

Use four separate worker processes, one Torch thread each. Each worker reuses a
process-local certified-scene cache; mutable episode states are copied, and
cache hits/certification work are recorded. Process assignment may change cache
timings, so this is not a controlled speedup experiment. Training uses one CPU
process with one Torch thread and a deterministic sample order independent of
worker completion order. Validation may run alongside collection, before fitting.

Record planning, physics, reset and task elapsed times; count collection-prefix,
accepted/rejected recovery, evaluation and certification actions separately.
Expected main evaluation count is 30 + 63 per completed round, maximum 156 fresh
episodes, plus the reused historical training baseline. Save compressed traces,
training data, all errors, model/data/code hashes and manifests. Keep model and
optimizer state frozen during collection/evaluation and prevent optimizer steps.

Tests cover recovery success filtering and parent-state isolation, budget limits,
planner failure, evaluator equivalence to the production evaluator, split
enforcement, supervised updates without TD/target changes, Adam resume and the
stopping rule. Preserve failed runs and partial outputs instead of overwriting
or silently restarting them. Distinguish preparation tests from actual trials.

Run each authorized round once from the repository root:
`python docs/experiments/your_method/008_batch_recovery/run_batch.py --round 1`
and, only if the stopping rule permits it, `--round 2`.
