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

Executed from commit `879ea1b` on **11 November 2026, 23:39:21--23:45:27
Asia/Shanghai**. Evidence timestamps use UTC. Calibration reproduced all 20000
update records, all eight scheduled observations and the final policy exactly.
The reconstructed optimizer had six initialized parameter tensors, each with
Adam step 20000. Both treatments started from identical calibrated weights and
return RMSE 0.274704, with a successful 35-step greedy episode.

| Measurement | Reset Adam | Preserve Adam |
| --- | --- | --- |
| Successful checks at TD updates 1--10 | 2/10 (updates 7 and 9) | 10/10 |
| First sampled failure | Update 1 | None observed |
| Success at all 18 scheduled checks, including update 0 | No (9/18) | Yes (18/18) |
| First-update action agreement | 24/35 | 35/35 |
| First-update episode | Budget exhausted at 200 steps | Success in 35 steps |
| First-update parameter-change norm | 0.0572692 | 0.0040680 |
| First-update loss before | 0.0826786 | 0.0826786 |
| First-update loss after, against the same frozen targets | 1.5354657 | 0.0734115 |
| Final demonstrated-return RMSE after 2000 TD updates | 0.4310585 | 0.5367391 |
| Final action agreement | 35/35 | 35/35 |
| Final episode | Success in 35 steps; return 107.6108 | Success in 35 steps; return 107.6108 |
| Final internal Adam step | 2000 | 22000 |
| Target-network synchronizations during TD | 10 | 10 |

The reset control exactly reproduced all 2000 TD update records and all nine
shared observations from attempt 009. The denser schedule additionally revealed
timeouts at updates 2--6, a successful 35-step episode at update 7, damage after
141 steps at update 8, a successful 38-step episode at update 9, and damage after
102 steps at update 20. Its complete failed checkpoint list is
**1, 2, 3, 4, 5, 6, 8, 10 and 20**. It succeeded at every sampled checkpoint from
50 to 2000, as in the preceding experiment.

The preserved-state policy matched all 35 demonstrated actions and completed the
same route in 35 steps at **every scheduled checkpoint**, including each of the
first ten updates. Its strict sampled-retention flag is true; the reset control's
is false. These fractions describe scheduled checkpoints of one deterministic
training trajectory, not independent episode success rates or held-out accuracy.

Both first updates had exactly the same initial TD loss, margin loss and
pre-clipping gradient norm (27.0190849). Retaining Adam reduced the parameter
change norm by about a factor of 14.1 and reduced the fixed-target loss, whereas
the reset update increased it. This supports keeping the accumulated optimizer
state for this specific calibrated transition. It does not isolate which Adam
moment or bias-correction component matters most. The later return RMSE was
slightly higher in the preserved branch, so the observed benefit is early
behavioral retention rather than improvement in every metric.

## Conclusion and next controlled question

Keeping Adam state eliminated the observed early TD failures in this controlled
comparison and retained success at all sampled later checkpoints through update
2000. The previously observed immediate failure is therefore avoidable under
these fixed-scene settings without lowering the learning rate or changing the
network, reward, demonstration or target synchronization schedule.

The next question is whether the same supervised calibration followed by TD with
retained Adam works across multiple training scenes and eventually unseen layouts.
That experiment has **not** been run here. This result does not establish broad
policy robustness, online learning performance or final project readiness.

## Verification and retained evidence

The full suite passed **296 tests**, including 20 new optimizer-transition tests
and seven new snapshot/experiment integration tests. Ruff passed. The new
implementation and the predeclared protocol were committed and pushed before
execution. Historical input logs and key records also matched attempt 009's
committed archive hashes before launch.

Actual experiment work was **24000 new optimizer updates** (20000 calibration
replay plus 4000 TD), with **3254 scheduled neural evaluation actions**: 802
during calibration reproduction, 1822 for reset and 630 for preserve. The original
20000 supervised updates were inherited. No demonstration was recollected, and
online exploration and held-out access remained zero. Reset feasibility probes
are excluded from action counts.

Independent verification passed: 51 launch source hashes, 16 attempt-009 input
hashes, nine inherited input hashes, all 24000 update records, 44 inference
checkpoints and three full training snapshots were checked. Replaying the first
ten TD updates for each treatment independently reproduced the exact update
metrics, online weights and Adam state. All six reloaded models reproduced
their action metrics, demonstrated-return errors and complete actual trajectories.
This added **20 verification optimizer updates and 401 evaluation actions**,
recorded separately in `reload_checks.json`. Historical target losses were
verified during the independent early replay; inference-only loading does not
restore those historical target networks.

All scheduled observations and first-ten/every-50th/last update samples are
retained, including failures and exact-reproduction checks. Complete local logs,
encoded data, all inference checkpoints and three full training snapshots remain
in the ignored run directory. Production training defaults and reports were not
changed.

Evidence: [figure](010_adam_state/evidence/adam_state.png),
[summary](010_adam_state/evidence/summary.json),
[calibration reproduction](010_adam_state/evidence/calibration_reproduction.json),
[reset-control reproduction](010_adam_state/evidence/reset_reproduction.json),
[optimizer snapshot record](010_adam_state/evidence/training_snapshot.json),
[independent verification](010_adam_state/evidence/reload_checks.json), and
[archive manifest](010_adam_state/evidence/archive_manifest.json).
