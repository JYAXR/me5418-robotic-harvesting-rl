# Attempt 009: full-trajectory recovery, geometry, and protected behavior

This is one bounded, multi-component intervention motivated by attempt 008's
offline analysis. It is not a single-factor causal comparison. The baseline is
`models/your_method_004_targeted.pt`, SHA-256
`ea42fd1a2d23317c9e15a960c58d90970a38edbcfb7724308a845ead92f6119f`.
Its recorded performance is 10/33 training, 3/3 original, and 3/30 validation.
No manually prescribed scene/step correction is used.

## Splits and frozen collection

Training: 10000, 10001, 10002, 290000--290029 (33 scenes). Validation:
590000--590029 (30 scenes). This validation bank has already been inspected
during development; it is not an untouched final test bank. No validation state,
action, label, loss, or margin enters training or the retention constraints.
The source model executes one complete greedy episode per training seed.
Environment, rewards, dynamics, action set, seed, and 200-step task budget stay
unchanged. Paired initial states and source episode metrics must match history.

Select at most eight visited states per episode: one risk and one coverage
query in each band 1--25, 26--50, 51--100, 101--200. Risk prioritizes the length
of the preceding blocked run, then obstacle proximity. Coverage chooses the
remaining state nearest the band's midpoint. Deduplicate raw geometry rounded
to 0.001, excluding remaining budget. Do not collect planner suffixes as training
examples. A successful complete physics-verified planner continuation provides
exactly one action label at its queried root. A failed continuation is retained
with label -1 (unknown), never assigned a fabricated action.

Probe all eight actions once on independent exact state clones. Record damage,
motion, reward, termination, observations, and RNG state. Mark immediate damage
or no-motion without success as adverse. A verified successful root action
cannot be marked adverse solely for no-motion: relaxation can precede success.
An actual damage action is never exempted. A state with all actions marked adverse
does not contribute the adverse-action loss because no preferred action is known.
These labels are a soft training objective, never an inference action mask.

## Policy representation

Keep the original 341--256--256--8 network immutable. Convert its scores to an
initial policy prior `clamp(Q - max(Q), min=-4)`, preserving its greedy action.
Append 40 geometry features to the existing 341 encoded values. For each of
eight commanded moves: current and rest-shape signed clearance, workspace
clearance, target-distance progress, and nearest-branch log stiffness. Clearance
samples 13 points along a commanded move; it is not a prediction of deformable
physics. Features are inputs, not permission to forbid flexible contact.

A new 381--128--8 ReLU correction network adds logits to the fixed prior. Its
last layer is zero-initialized. Outputs are policy scores, not calibrated Q
values. The original DQN loader and models remain unchanged; a separate format-2
loader and `scripts/evaluate_recovery_policy.py` evaluate this candidate.

## Data, objective, and retained behavior

Retain the 1440 original planner demonstration rows. Unlike attempt 008, do not
append the legacy single correction or the 4103 planner-suffix rows. All new
recovery rows are actually visited source-policy states. Archive a separate
anchor bank containing every action on the source's ten successful training
trajectories; labels are the source's executed actions, not potentially
contradictory planner alternatives. Successful seed IDs are derived from the
archived source evaluation, not selected after this run.

At each attempted update sample with replacement, uniformly over available
scene IDs and then rows within the chosen scene:

- 64 verified recovery roots;
- 32 original demonstration rows;
- 64 successful trajectory anchors;
- 64 risk-labeled query states, if available.

Risk/coverage quotas apply during collection; training is scene-balanced, not
additionally category-balanced. Minimize recovery cross-entropy + 0.5 original
cross-entropy + anchor KL to the converted source distribution + anchor action
margin loss (margin 0.1) + 0.5 probability mass on adverse actions. Clip gradient
norm to 10. Train only the correction network, Adam learning rate 0.0001 with
fresh state, deterministic seed 817009. The historical Adam moments cannot be
transferred to newly introduced parameters and are not claimed to be retained.
The historical TD clock is 14000; this experiment performs zero TD updates.

After each 20-update block, check every retained action with batch-one inference.
Try parameter interpolation from the pre-block state to the proposed state at
scales 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64. Accept the largest scale retaining all
anchor argmax actions. Keep proposed Adam moments for an accepted projected
block. If none passes, restore both weights and Adam state. Sampling RNG still
advances. Record attempted updates separately from surviving Adam steps and
accepted blocks. This is a constrained heuristic, not standard unmodified Adam.

## Bounds and acceptance

Pilot: exactly 400 attempted updates. Continue to 2000 total only if at least
one block was accepted and accuracy on verified recovery roots improves by at
least 3 percentage points over initialization. Otherwise stop at 400. No extra
round, learning-rate adjustment, manual correction, or validation-driven
checkpoint search is allowed. Evaluate the single stopped/final candidate on
all 33 training and all 30 validation scenes, with greedy policy actions only.

Archive the candidate regardless of outcome. It may replace the retained
baseline only if all ten old successful training scenes still succeed, all
three original scenes succeed, and validation exceeds 3/30. Even an eligible
candidate is not claimed to be final-project-ready. Report gains, regressions,
damage, budget failures, returns, query coverage, failed queries, retention
scales, runtime, and simulation steps, including certification/probing cost.

Four worker processes, one Torch thread each, certified scene cache capacity
128 per process. Timing and trace hashing remain enabled. Pin code, models,
datasets, retention traces, protocol, and source evaluation metadata before
execution. Preserve errors and partial output; do not silently rerun trials.

## Reproduction

From the repository root in the configured environment:

```text
python -m pytest tests/test_recovery_policy.py tests/test_training.py -q
python docs/experiments/your_method/009_geometry_retention/preflight.py
python docs/experiments/your_method/009_geometry_retention/run_study.py
python scripts/evaluate_recovery_policy.py --checkpoint models/your_method_009_geometry.pt --seed 10000 --episodes 3
```

Preflight is offline on historical trajectories plus one synthetic two-step
process-pool smoke test; it performs no training or experimental-seed rollout.
The scripts refuse to overwrite existing preparation/evidence directories.
Use a clean output location in an isolated checkout to reproduce a recorded run.
