# Attempt 009: calibrate demonstrated returns before TD adaptation

## Protocol recorded before execution

Attempts 007 and 008 showed that TD adaptation can destroy a successfully learned
route, even with action supervision and smaller learning rates. This experiment
tests a specific hypothesis: calibrating demonstrated-action Q scores to actual
discounted trajectory returns before bootstrapping may reduce that deterioration.
The comparison includes an equal-budget control so additional training alone is
not mistaken for a calibration benefit.

Use the verified attempt-007 supervised checkpoint after **20000 updates** and
the same frozen **35-transition** complete demonstration at training seed 10000.
Preserve the multi-route environment, 341-input/256/256/8-output MLP, encoder,
reward, gamma 0.99, full batch, Adam learning rate 0.0003, gradient clipping 10,
margin 0.8 and target synchronization every 200 TD updates. No new demonstration,
scene, exploratory action or held-out example is introduced. The shared warmup
is reused from its checkpoint, with exact data and parameter hashes verified.

Compute fixed Monte Carlo targets from the recorded environment rewards:

```text
G[T-1] = reward[T-1]
G[t] = reward[t] + gamma * G[t+1]
```

The trajectory must be ordered and continuous, contain no external truncation,
and end with its only task termination. The terminal reward is included exactly
once. No target-network value or reset observation enters this calculation.
Independently check the recurrence in float64 against the training targets and
save the rewards and targets. These are returns of this demonstrated continuation,
not independently known optimal Q values. The recorded float32 rewards give
targets of approximately **77.4551 to 100.4807**; their rounding accounts for the
small difference from sums of the original simulator's Python float rewards.

Fork two branches from the identical successful supervised policy:

| Branch | Additional pre-TD phase | Budget |
| --- | --- | --- |
| `continued_margin` | Continue action-margin supervision alone | 20000 updates |
| `calibrated` | Huber loss between chosen-action Q and fixed MC return, plus action-margin loss with weight 1 | 20000 updates |

Both branches begin with fresh Adam and the same weights. The target network is
copied for a consistent boundary but is not read during either pre-TD objective.
Evaluate this phase at **0, 1, 10, 100, 1000, 5000, 10000 and 20000**. Record
per-state return errors, RMSE, MAE, maximum absolute error and bias, as well as
whole-dataset action agreement and actual greedy episode outcomes. An average
score close to the average return does not establish per-state calibration.

Then apply the **same 2000 TD+margin updates** to both branches. Copy each branch's
current online weights to its target, reset Adam equally and start a fresh phase
clock. The rate remains 0.0003 and the margin weight remains 1. Evaluate at
**0, 1, 10, 50, 100, 200, 500, 1000 and 2000**. Verify that the boundary itself
does not change the preceding phase's fit or rollout. All network parameters
remain trainable; no policy rollback or planner fallback is permitted.

Run both predeclared TD phases even if a pre-TD branch fails. If a branch does
not succeed at TD step zero, explicitly mark it **ineligible for a retention
claim**; its later outcomes describe downstream behavior or recovery instead.
Never select only a transient successful checkpoint, stop early, or tune settings
between arms. Preserve every sampled failure and recovery. Success at sampled
checkpoints does not establish success at unobserved intermediate updates.

New cost is **44000 optimizer updates**: 40000 across the two additional pre-TD
phases and 4000 TD updates. Each branch has a 42000-update history including the
shared inherited warmup, but the inherited 20000 updates are not executed or
charged twice. Count scheduled evaluation and post-run verification actions
separately; reset feasibility probes are excluded. New demonstration collection,
online exploration and held-out access remain zero.

## Reproduction

In the project environment, with the verified reference run available:

```powershell
python docs/experiments/009_return_calibration/run_return_calibration.py --reference-run results/training/staged_retention_20261111 --output-dir results/training/return_calibration_20261111
```

The new return calculation and calibration update are reusable functions in
`src/me5418_harvesting/agents/return_calibration.py`. Existing learner behavior
and earlier experiment scripts are unchanged. Source, input and model hashes,
resolved settings, fixed targets and complete local update logs are recorded.
All code, comments and runtime messages remain in English.

## Interpretation limits

This is one known training scene and a fixed-data TD comparison. Any improvement
would support this calibration procedure under these settings, not prove that
scale mismatch was the sole cause of prior failure. The procedure changes the
optimization objective and gradients, not just a displayed score scale. Fixed
route learning may still exploit the remaining-budget feature. Generalization,
online replay, exploration and final project readiness are outside this test.

## Outcomes

Pending the predeclared run. Keep the equal-budget control and all failures.
