# Attempt 010: preserve or reset Adam at the calibrated TD boundary

## Protocol recorded before execution

Attempt 009 calibrated demonstrated-action values to complete observed discounted
returns. Its calibrated policy still failed after TD updates 1 and 10, then
succeeded at every sampled checkpoint from 50 to 2000. The first TD update
increased the fixed-target combined loss from 0.08268 to 1.53547. This experiment
tests whether retaining the calibration optimizer state reduces that early
deterioration. It does not assume that Adam reset is the only cause.

Use the same single known multi-route scene (seed 10000), frozen 35-transition
demonstration, 341/256/256/8 MLP, gamma 0.99, full batch, learning rate 0.0003,
gradient clipping 10, margin 0.8 and demonstration weight 1. Do not change the
reward, observation, network, data, objective, exploration or target schedule.

The inference checkpoints from attempt 009 do not contain Adam moments. Rebuild
the missing state by replaying its **20000 calibration updates once** from the
verified attempt-007 supervised checkpoint. Verify all update records, scheduled
observations and final weights against attempt 009 before making a comparison.
Save a local full training snapshot including online weights, target weights,
Adam state, optimizer options and the update counter. Do not silently substitute
a newly initialized optimizer for a resumed optimizer. If exact reproduction
fails, retain the failed run and stop before the experimental TD branches.

Fork two independent copies of this same calibrated snapshot:

| Branch | Adam state at the TD boundary | New TD updates |
| --- | --- | --- |
| `reset` | Empty moments and a new per-parameter step count | 2000 |
| `preserve` | Keep calibration moments and per-parameter step count 20000 | 2000 |

Both branches copy their identical online weights to their target network, freeze
the target, clear stale gradients and start the **TD phase counter at zero**.
Target synchronization occurs every 200 TD updates in both branches. The Adam
step counter is separate: preserving it also preserves its bias correction.
All optimizer parameter-group settings are identical; only the accumulated
optimizer state differs. Forked tensors must not alias each other or the saved
snapshot. Train every network parameter in both branches.

Use identical one-step Huber TD plus action-margin updates on the frozen batch.
Evaluate at **every update from 0 through 10**, then at **20, 50, 100, 200, 500,
1000 and 2000** (18 observations per branch). Save actual greedy trajectories,
success/damage/budget outcomes, action agreement, demonstrated-return errors,
fixed-target losses before/after updates, gradient norms and parameter changes.
Record the optimizer state at the boundary, after the first update and at the
end. Check that the reset control reproduces attempt 009 at every shared
checkpoint and in all 2000 update records.

The primary comparison is early retention across updates 1--10. Report the first
sampled failure, every early outcome and later recovery or deterioration. These
are checkpoints along one training run, not independent test episodes. No early
stopping, checkpoint selection, policy rollback, planner fallback or tuning
between arms is permitted. A retention claim requires successful behavior before
the first TD update. Success only at sampled updates does not establish success
between observations.

Actual new work is **24000 optimizer updates**: one shared 20000-update
calibration replay and 4000 TD updates. The original 20000 supervised updates
are inherited and not rerun. Each branch has a 42000-update history, but the
shared work is counted once. Record evaluation and independent reload actions
separately, excluding reset-time feasibility probes. No demonstration collection,
online exploration or held-out access occurs.

After the run, independently replay the first ten TD updates for each treatment
from the shared snapshot and check update metrics, optimizer states and saved
weights. Count these **20 verification updates** separately from the experiment.
Reload the six policies at updates 1, 10 and 2000 for both treatments and verify
their metrics and complete actual trajectories, counting those actions separately.

## Reproduction

Run in the project environment with the verified local reference runs available:

```powershell
python docs/experiments/010_adam_state/run_adam_state.py --reference-run results/training/return_calibration_20261111 --output-dir results/training/adam_state_20261111
```

The reusable boundary helper is in
`src/me5418_harvesting/agents/optimizer_transition.py`. Previous experiment scripts
and production training defaults remain unchanged. All new code, comments and
runtime messages are in English. Source hashes, input hashes, resolved settings,
full local training snapshots and failed observations are retained.

## Interpretation limits

This comparison isolates accumulated Adam state as a whole, including first and
second moments and the bias-correction counter. It cannot identify the separate
contribution of each component. A benefit here would apply to this one fixed
training scene and checkpoint; generalization, online learning and final project
readiness remain untested. Fixed-route memorization through the remaining-budget
feature remains possible.

## Outcomes

Pending execution. Do not interpret the predeclared hypothesis as a result.
