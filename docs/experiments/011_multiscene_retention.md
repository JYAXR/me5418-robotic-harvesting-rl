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

Pending execution. Acquisition, calibration and TD retention will be reported
separately after the predeclared run, including negative outcomes.
