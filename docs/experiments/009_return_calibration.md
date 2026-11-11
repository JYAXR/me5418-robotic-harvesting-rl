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

Executed from commit `42396a0` on **11 November 2026, 22:39:53--22:46:14
Asia/Shanghai**. Evidence timestamps use UTC. Both branches started from exactly
the same successful supervised weights and data. The independently checked fixed
returns spanned 77.4551--100.4807; float32 recurrence differed from the float64
reference by at most 0.0000214, within the predeclared 0.0001 tolerance.

| Measurement | Equal-budget margin-only continuation | Return calibration + margin |
| --- | --- | --- |
| Initial RMSE against demonstrated returns | 83.1175 | 83.1175 |
| RMSE after the additional 20000 updates | 80.8141 | 0.2747 |
| Episode at the end of that phase | Success, 35 steps | Success, 35 steps |
| First sampled failure after starting TD | Update 1 | Update 1 |
| Successful positive-update TD checkpoints | 100, 200 | 50, 100, 200, 500, 1000, 2000 |
| Final TD action agreement | 33/35 (94.3%) | 35/35 (100%) |
| Final TD episode | Timeout, 200 steps; return -17.1309 | Success, 35 steps; return 107.6108 |
| Final RMSE after 2000 TD updates | 79.3237 | 0.4311 |

Both branches were eligible for a retention comparison because both entered TD
with a successful policy. **Neither retained success at every sampled TD
checkpoint**. Calibrating returns reduced persistent later deterioration in this
test, but did not prevent the initial loss of behavior at the phase switch.

The calibration process itself was not uniformly successful: its checkpoints
1, 10, 100 and 1000 failed, including branch damage at update 10. By update 5000,
RMSE had fallen to 0.5218 and the policy completed the route in 35 steps. At
20000, all 35 demonstrated actions matched; MAE was 0.2277 and maximum absolute
return error was 0.6161. The continuation control succeeded at all pre-TD
checkpoints, including a 36-step alternative route at update 10000.

After TD began, the calibrated policy first dropped to 24/35 action matches and
timed out at update 1. At update 10 it matched 31/35 and damaged a branch after
61 steps. It then completed the reference route in 35 steps with 35/35 agreement
at **every sampled checkpoint from 50 through 2000**. In comparison, the control
recovered only at the sampled 100- and 200-update checkpoints and ended in a
timeout. These are repeated checkpoints of one deterministic training scene,
not independent test episodes or a generalization success-rate estimate.

The first calibrated TD update increased the same-batch combined loss from
0.08268 to 1.53547 against the same frozen targets. Its parameter-vector change
was 0.05727, compared with 0.05860 in the control's first update. This observation
motivates a separate test of the phase transition and optimizer reset; it does
not establish that resetting Adam is the sole cause. Each branch deliberately
used the fresh-Adam boundary stated in the protocol.

## Conclusion and next controlled question

Fixed-return calibration substantially reduced numerical value error and improved
the later TD checkpoints over the matched additional-supervision control. It is
therefore a useful component to investigate further under this single-scene
setup. It is not yet a complete solution to capability retention: both strict
all-checkpoint retention flags remain false, and no new scene was tested.

A next isolated comparison could keep versus reset the calibration optimizer's
state when TD begins, holding the calibrated weights and all other settings
fixed. This has **not** been implemented or tested here. Existing inference
checkpoints do not contain Adam moments; a valid comparison must retain or
reproduce the calibration optimizer state rather than pretend that an inference
checkpoint resumes it. Expansion to multiple scenes remains a separate task.

## Verification and retained evidence

**269 tests passed**, including 34 new unit/integration cases; Ruff passed.
Verification checked 50 launch source hashes, all input/helper hashes, all 44000
update records and 34 checkpoint hashes/configurations. The four phase-end
models were independently reloaded and exactly reproduced action metrics,
per-state MC errors and complete rollout traces. Historical target-network
metrics are not claimed to survive inference-only loading.

New accounting: **44000 optimizer updates**, including **4000 TD updates**;
**2724 scheduled neural evaluation actions**; and **305 additional actions** for
independent reload verification. No demonstration was recollected and no online
exploration, original warmup rerun or held-out access occurred. The shared
20000-update warmup and 35 demonstration transitions were reused from attempt
007. Interaction counts exclude reset feasibility probes.

Evidence: [figure](009_return_calibration/evidence/return_calibration.png),
[summary](009_return_calibration/evidence/summary.json),
[fixed targets and numerical check](009_return_calibration/evidence/mc_targets.json),
[input manifest](009_return_calibration/evidence/input_manifest.json),
[reload verification](009_return_calibration/evidence/reload_checks.json), and
[archive manifest](009_return_calibration/evidence/archive_manifest.json).
All scheduled observations and first-ten/every-50th/last update samples are
archived, with complete-log hashes. Full logs, encoded data and model weights
remain in the ignored local run directory. Existing production training defaults
and reports were not modified.
