# Attempt 019: reduce the TD objective coefficient while retaining Adam

## Protocol recorded before execution

Attempt 018 diagnosed opposed TD and margin gradients at 25 of 28 selected
historical states and showed that retained Adam moments can change the direction
of a fragile action-gap update. Removing TD in one-step counterfactuals recovered
some local rankings but was not a universal remedy. Those retrospective findings
motivate a sustained single-factor comparison; they do not establish that a
smaller TD coefficient will preserve complete episodes.

The requested treatment changes **only the TD loss coefficient from 1 to 0.5**.
Use the verified attempt-016 baseline as the historical control, without the
MC constraint tested in attempt 017 or the local weighting tested in attempt 016.

For the same ordered batch of 117 demonstration transitions:

- Historical control: `mean(TD Huber) + mean(demonstration margin)`.
- New treatment: `0.5 * mean(TD Huber) + mean(demonstration margin)`.

All supervised sample coefficients remain 1, with margin 0.8. Each term retains
its mean over 117 transitions. Do not divide the combined objective by 1.5,
duplicate data, blend targets, add an actual-return loss, or change the common
learning rate. Actual discounted demonstration returns are reused only for
evaluation and calibration-error reporting. They are not part of the optimized
objective. Fix the TD coefficient at 0.5 before execution; do not retune it in
response to failures or intermediate successes.

## Exact continuation and unchanged conditions

Restore the original successful **TD-2000 full snapshot**, including online and
target networks, Adam first and second moments and parameter step counters, both
RNG states, and absolute TD clock. The inherited branch contains 20000 supervised
updates, 20000 actual-return calibration updates and 2000 TD updates. Adam begins
at step 22000. Do not rerun pretraining, clear moments, reset counters or perform
an extra target synchronization at the boundary.

Keep learning rate 0.0001, gamma 0.99, gradient clipping at 10, target interval
200, the 341/256/256/8 network, environment settings, scene seeds, action and
reward definitions, encoder, and full-batch transition order unchanged. The TD
coefficient changes the combined gradient's direction and magnitude; this is
not a claim that the resulting parameter displacement stays fixed.

Use the same helper/source chain and verified historical control as attempt 017:
attempt-016 baseline at `e51effb2f5e0e05fb70486088d38ab235a395106`, previously
verified to reproduce attempt 014 exactly. Audit its committed archive, original
demonstrations, complete model hashes, raw logs, full starting state and previous
reproduction gate before new training. Reuse its 8000 updates, 87 episodes and
3711 policy actions as **historical evidence**, not newly executed work.

Train exactly 8000 new updates, **TD 2001 through 10000**. The branch ends at
50000 cumulative updates and Adam step 30000. Ordinary target synchronizations
occur at 2200, 2400, ..., 10000. Do not stop early, roll back, freeze the network,
force actions or replace the prescribed final model with a selected checkpoint.
Keep production code and earlier experiment evidence immutable. All new code,
comments and runtime messages are English.

## Evaluation and measurements

Use the original 29 physical checkpoints:

`2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
2200, 2201, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500,
7000, 7500, 8000, 8500, 9000, 9500, 10000`.

At each checkpoint, evaluate scenes 10000, 10001 and 10002 with autonomous greedy
network actions, giving 87 new main episodes. Before the first update, require
the exact original successful behavior and full-batch Q values. Record success,
damage, budget exhaustion, return, actions, blocked/stationary behavior, and
first reference-action divergence. Preserve all failures and observed recoveries.
A first divergence is not automatically the cause of a failed continuation.

After every update and at the restored boundary, measure all eight Q values on
scene 10001's fixed demonstrated states at steps **6, 18, 20, 21, 22, 23 and 24**,
using batch-one inference and the original scalar subtraction procedure.
Record `Q(demo) - max(Q(other))`, greedy actions, negative intervals, exact ties,
minima and final gaps. Compare all 8001 records per arm, including the four
neighboring decisions at steps 20--23 and the earlier step 6. These 56007 new
probe forwards execute no environment actions. A reference-state reversal is
not a measured failed episode; later demonstrated states may be unvisited after
the actual trajectory changes.

Keep the original 67 model files: 29 inference checkpoints, 29 full snapshots,
eight otherwise missing target-source snapshots and the final full-state alias.
Log all 8000 updates, including the unweighted TD loss, unweighted supervised
loss, weighted combined loss, coefficient, pre-clipping gradient norm, parameter
change and target synchronization. Before/after TD diagnostics use the same
frozen pre-update Bellman targets. Compare shared unweighted components and
physical outcomes; combined objective values differ by design. Evaluate final
demonstration agreement after updating rather than treating pre-update logged
accuracy as final accuracy. Calibration RMSE is a diagnostic, not an MC term.

## Reset certification and work accounting

Keep `verify_reference_route=True` as in the historical environment. Each reset
may check one or more candidate scenes using a planner and a copied simulation.
Observe the existing reset return information without rerunning checks or
changing observations, RNG state or acceptance behavior. Record every completed
reference-route check and its executed steps, including rejected candidates.

Keep policy episode/action counts comparable to previous attempts. Separately
report reset calls, certification checks and copied-simulation steps, then their
combined executed-action total for the new treatment. Historical policy counts
remain explicitly reused; do not present an unlogged historical certification
total as newly measured. Mark accounting incomplete if a reset fails before
returning its diagnostic information.

## Verification and execution

Meaningful focused tests must verify original-updater equivalence at TD weight 1,
the independently computed 0.5 objective and gradients, unchanged Adam continuity,
correct target timing, coefficient validation before mutation, and reset instrumentation
that preserves returned values and restores the original method.

After the main run, independently implement the weighted objective and replay
updates 2001--2201 for both control and treatment. Compare all update logs, dense
probes and available full states, including the target synchronization at 2200:
**402 additional verification updates**. Audit all later hashes, counters,
diagnostics and target lineage without claiming numerical replay of every main
update.

Reload the new treatment at TD 2000, 2009, 2200, 2201 and 10000 on all three
scenes: **15 additional verification episodes**. Compare exact historical
treatment outcomes and separately count verification policy actions, resets,
certification checks and internal copied-simulation steps. Do not rerun the
historical control's episodes. Analysis and plotting add no optimizer updates
or physical actions.

Run from the repository root:

```powershell
.venv/Scripts/python.exe -u docs/experiments/019_td_weight/run_td_weight.py --reference-run results/training/local_supervision_20261112 --archive-dir docs/experiments/016_local_supervision/evidence --output-dir results/training/td_weight_20261114
.venv/Scripts/python.exe -u docs/experiments/019_td_weight/verify_td_weight.py --run-dir results/training/td_weight_20261114 --output results/training/td_weight_20261114/reload_checks.json
```

Retain the fixed protocol, source and script hashes, all outcomes, dense traces,
training diagnostics, verification results, accounting, and any failed attempts.
Archive compact evidence and English experiment notes in Git, with full binary
snapshots retained locally. Three known scenes and one inherited initialization
do not establish generalization. Sampled physical success does not prove success
between checkpoints. This experiment does not automatically change production
defaults.

Results are pending at the pre-execution protocol commit.

Before execution, 53 focused runner/analysis tests passed in 5.68 seconds, and
Ruff passed for the frozen runner, analyzer, plotter, archive tool and their
completed tests. A read-only source preflight verified 117 transitions, TD 2000,
Adam step 22000, the historical 3711 policy actions, 52 production/configuration
hashes and 14 helper hashes. It performed no optimizer updates or environment
actions. The independent verifier and its tests are developed separately and
must pass before accepting the completed experiment.
