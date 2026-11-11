# Attempt 011: staged learning and Adam-state retention across three training scenes

## Protocol recorded before execution

Attempt 010 preserved a known single-scene policy at every sampled TD checkpoint
when calibration Adam state was retained. This experiment extends the complete
pipeline to **three predeclared training seeds: 10000, 10001 and 10002**. It asks
whether one network can acquire these routes, survive return calibration and
retain its acquired behavior during TD. No held-out scene is accessed.

Start a fresh network with initialization seed 0. Do not initialize from the
single-scene model. Keep `environment_multiroute.yaml` and
`dqn_multiroute_pilot.yaml` unchanged: 14 branches, a 200-step task budget,
341/256/256/8 MLP, Adam learning rate 0.0003, gamma 0.99, gradient clipping 10,
action margin 0.8, demonstration weight 1 and a 200-update TD target period.
There is no seed identifier added to the network input.

Collect one reference-planned trajectory per seed and execute it in the actual
environment. Retain all collection attempts and count executed actions separately
from planning/reset feasibility probes. Each accepted trajectory must succeed,
have exactly one final task termination, have no external truncation and be
continuous within that episode. If any predeclared seed fails collection, save
the failure and stop; do not replace it with an easier seed. Check that the three
initial encoded observations differ. Record trajectory lengths, scene offsets,
initial-state hashes, label counts and any exactly duplicated states with
conflicting demonstrated labels. Do not remove conflicts or select trajectories
based on subsequent neural performance.

Concatenate all transitions in seed order for every optimizer update. The loss
is a **per-transition full-batch mean**; longer trajectories therefore have more
weight. This is not an equal-per-scene objective. Retain explicit episode offsets
and compute actual discounted returns separately within each trajectory:

```text
G[episode_end - 1] = reward[episode_end - 1]
G[t] = reward[t] + gamma * G[t + 1]
```

Never propagate returns across an episode boundary or bootstrap these calibration
targets from a network. The targets describe demonstrated continuations, not a
known optimal value function. Check each episode's recurrence independently.

| Stage | Objective and boundary | Updates |
| --- | --- | --- |
| Supervised pretraining | Margin-only action supervision; fresh seed-0 network and Adam | 20000 |
| Return calibration | Fixed-return Huber loss plus action margin; keep weights, reset Adam as in 009 | 20000 |
| Reset TD control | One-step Huber TD plus margin; reset Adam from the shared calibrated snapshot | 2000 |
| Preserved-state TD | Identical TD objective; retain calibration Adam moments and internal steps | 2000 |

Save complete fixed-data training snapshots, including online weights, target
weights, optimizer state and phase counters. Fork both TD arms independently
from the exact same calibrated state. Both copy online weights to frozen target
networks, clear gradients and restart the TD phase clock at zero. The preserved
Adam counter remains 20000; the reset optimizer has empty state. All optimizer
settings and target synchronization schedules remain equal. Keep every network
parameter trainable and prevent tensor aliasing between treatments.

Evaluate **all three scenes** at every checkpoint:

- Supervision: 0, 1000, 5000, 10000 and 20000 updates.
- Calibration: 0, 1, 10, 100, 1000, 5000, 10000 and 20000 updates.
- Each TD arm: every update from 0 through 10, then 20, 50, 100, 200, 500, 1000 and 2000.

Record pooled and per-scene action agreement, return error, actual greedy
success/damage/budget outcomes and complete trajectories. Log fixed-target losses
before/after each TD update, parameter changes and optimizer state at each
scheduled checkpoint. Planning is used only for demonstration collection and
environment feasibility checks; neural rollouts do not use planner actions.

Do not require success before proceeding to the next scheduled stage. Instead,
make **retention eligibility explicit per scene**. Compare calibration against
the successful supervision-end subset, then compare TD against the successful
calibration-end subset. A scene already failing at a boundary cannot demonstrate
forgetting of an acquired successful policy; later success there is recovery or
new acquisition. If the eligible subset is empty, retention is not applicable,
not vacuously successful. All-three-scene retention requires that all three
scenes succeed at the preceding boundary. Preserve failures even when a pooled
score improves. Do not tune, stop early, select a transient checkpoint, change
seeds, roll back weights or use a planner fallback during this run.

The experiment performs **44000 new optimizer updates**: 20000 supervision,
20000 shared calibration and 2000 TD updates per arm. There are no inherited
optimizer updates. With all stages completed, the schedule contains 147 neural
evaluation episodes (49 checkpoints times three scenes). Count their actual
environment actions, collection actions and independent verification actions
separately. Online exploration and held-out access remain zero.

## Reproduction

```powershell
python docs/experiments/011_multiscene_retention/run_multiscene_retention.py --output-dir results/training/multiscene_retention_20261112
```

The reusable trajectory assembly and per-episode return logic are in
`src/me5418_harvesting/agents/multiscene_demonstrations.py`. Earlier code and
experimental records remain unchanged. New code, comments and runtime messages
are in English. Record the launch commit, source/helper hashes, resolved settings,
dataset hashes, stage snapshots and all observed outcomes.

Independently replay the first ten TD updates per arm after completion, accounting
for **20 verification updates** separately. Reload the supervision-end and
calibration-end models plus updates 1 and 2000 from both TD arms, and execute each
on all three scenes (18 verification episodes). Check exact metrics and full
trajectories, including failed rollouts.

## Interpretation limits

These are three known training scenes from one network initialization, not an
unseen-layout evaluation. Three trajectories are still a small data set, and
remaining-budget inputs may support route memorization. Equal update counts do
not mean equal computation or batch size compared with the single-scene test.
This run tests the existing staged pipeline under a modest increase in training
diversity; it does not establish online RL performance or final project readiness.

## Outcomes

Executed from commit `b22f0cb` on **12 November 2026, 00:02:59--00:24:23
Asia/Shanghai** (11 November 16:02:59--16:24:23 UTC). The implementation and
predeclared protocol were committed in `f686a19`; the following commit refined
only a numerical cross-check in the independent verifier before execution.

All three predeclared scenes yielded successful physically executed reference
routes, with **35, 41 and 41 transitions**, respectively. The combined batch
contained **117 transitions**, three distinct initial observations, no duplicate
observations and no duplicate-state action-label conflicts. No seed or trajectory
was replaced after collection or training.

### Acquisition and calibration

At 5000 supervised updates, seeds 10001 and 10002 succeeded while 10000 timed out.
All three first succeeded together at the sampled 10000-update checkpoint. At
20000, the shared policy matched **117/117 demonstrated actions** and completed
the three routes in **35/41/41 steps**. These results establish acquisition on
the three training scenes, not on new layouts.

Return calibration reduced pooled demonstrated-return RMSE from **81.8364 to
0.3931**. Its final model again matched 117/117 actions and succeeded on all three
scenes, so every scene was eligible for the subsequent TD retention comparison.
However, calibration itself did **not** preserve behavior throughout: all three
scenes failed at updates 1, 10, 100, 1000 and 5000; only seed 10000 succeeded at
10000. The other two recovered by the final checkpoint. A well-calibrated final
model must not be confused with uninterrupted preservation during calibration.

### Matched TD comparison

| Measurement | Reset Adam | Preserve Adam |
| --- | --- | --- |
| Scene successes after the first TD update | 0/3 | 2/3 |
| Demonstrated actions matched after the first update | 46/117 | 114/117 |
| Checks with all three scenes successful during updates 1--10 | 0/10 | 6/10 |
| Individual scene successes across those first ten checks | 3/30 | 26/30 |
| Checks with all scenes successful among all 18 scheduled checks, including update 0 | 5/18 | 13/18 |
| Individual scene successes across all scheduled TD checks | 23/54 | 49/54 |
| Scenes retained at every scheduled TD checkpoint | None | 10000 and 10002 |
| First-update parameter-change norm | 0.0638853 | 0.0045233 |
| First-update combined loss before | 0.1767203 | 0.1767203 |
| First-update combined loss after, against the same frozen targets | 4.0181131 | 0.1796351 |
| Final return RMSE after 2000 TD updates | 0.5836207 | 0.5630840 |
| Final demonstrated action agreement | 116/117 | 117/117 |
| Final scene successes and route lengths | 3/3; 35/41/41 steps | 3/3; 35/41/41 steps |
| Strict all-scene sampled retention | False | False |

These fractions count repeated checkpoints of the same three training scenes.
They are not estimates of independent test success rates or unseen-layout
generalization. Both arms entered TD with all scenes successful, so differences
during TD are valid retention comparisons rather than failure to acquire a route
before the phase began.

The preserved-state policy failed only on seed **10001**, at updates
**1, 6, 9, 10 and 200**, always by exhausting the 200-step budget. Seeds 10000 and
10002 succeeded at all 18 scheduled TD checkpoints. All three scenes succeeded
together at updates 2, 3, 4, 5, 7, 8, 20, 50, 100, 500, 1000 and 2000, in
addition to the common starting checkpoint.

The reset control's failures were more widespread:

- Seed 10000 failed at each update from 1 through 10.
- Seed 10001 failed at 1, 2, 3, 5, 6, 7, 8, 9, 10, 20, 50 and 1000.
- Seed 10002 failed at 1, 2, 3, 4, 7, 8, 9, 10 and 20.

The reset control recovered all three scenes at updates 100, 200 and 500, lost
10001 again at 1000, and recovered it by 2000. Its final successful model therefore
does not erase the intervening failures. At preserved update 200, seed 10001
matched 40/41 demonstration actions but still timed out, illustrating why action
agreement alone is insufficient.

Both first TD updates used the same starting losses, target values and
pre-clipping gradient norm (28.1947441). Retaining Adam reduced the parameter
change by about **14.1 times**, but its total loss still increased slightly,
from 0.17672 to 0.17964. Its TD component decreased from 0.11410 to 0.09278 while
its margin component increased from 0.06262 to 0.08685. Unlike attempt 010, this
multi-scene first update did not reduce the combined loss or preserve every
scene. The comparison isolates accumulated Adam state as a whole, not any one
moment or bias-correction component.

## Conclusion and next controlled question

The full pipeline now runs on multiple training scenes. One network acquired all
three routes, and both final TD models remained functional on those same scenes.
Preserving calibration Adam state substantially reduced the number and spread of
observed TD failures compared with the matched reset control. The single-scene
result did **not** extend to perfect multi-scene retention: both strict all-scene
flags remain false, and calibration temporarily lost the acquired routes too.

The next diagnostic should examine the first divergent action and competing Q
values on seed 10001 at the preserved-state failure checkpoints. That can inform
a separate, predeclared adjustment of the TD update size or supervision strength,
changing one factor at a time. This analysis and any further intervention have
**not** been performed here. Broader training diversity and unseen-layout tests
remain separate work; these three fixed routes do not establish broad
generalization or final project readiness.

## Verification and retained evidence

**333 tests passed**, including 27 new trajectory/return tests and ten new
experiment integration tests. Ruff passed. The code, protocol and verification
logic were committed and pushed before the main run. All new code, comments and
runtime messages remain in English.

Actual main-run work: **44000 optimizer updates**, **117 executed demonstration
actions**, and **147 scheduled neural episodes / 13959 evaluation actions**.
Evaluation actions were 1744 during supervision, 3320 during calibration, 5991
for reset TD and 2904 for preserved TD. There were no inherited optimizer
updates, online exploration steps or held-out accesses. Planner search and reset
feasibility probes are excluded from these action counts.

Independent verification passed: 52 launch source hashes, two configuration
inputs, all recorded helper hashes, all 44000 update records, 49 inference
checkpoints and five full training snapshots were checked. Independent return
recurrences respected all episode boundaries; the maximum float32 versus
float64 difference was 0.0000254, below the 0.0001 tolerance. The first ten TD
updates per arm independently reproduced the exact metrics, online tensors and
Adam states. Six reloaded policies reproduced exact metrics and trajectories
across all three scenes, including their failed episodes. This added **20
verification optimizer updates, 18 episodes and 1344 actions**, counted separately
in `reload_checks.json`. Historical target losses are verified by the early
update replay; inference-only loading does not restore historical target networks.

All observations and first-ten/every-50th/last update samples are retained,
including failures.
Full local update logs, encoded data, inference checkpoints and five complete
fixed-data training snapshots remain under the ignored run directory. Existing
production training defaults, earlier experiments and reports were not modified.

Evidence: [all per-scene outcomes](011_multiscene_retention/evidence/multiscene_retention.png),
[summary](011_multiscene_retention/evidence/summary.json),
[collection](011_multiscene_retention/evidence/demonstrations.json),
[dataset and episode offsets](011_multiscene_retention/evidence/dataset.json),
[per-episode return checks](011_multiscene_retention/evidence/mc_targets.json),
[independent verification](011_multiscene_retention/evidence/reload_checks.json), and
[archive manifest](011_multiscene_retention/evidence/archive_manifest.json).
