# Attempt 005: fixed-scene fitting diagnostic

## Question and protocol recorded before execution

Attempt 004's frozen pretrained model agreed with only 125 of its 421 demonstrated
actions. This diagnostic asks whether the existing encoder and MLP can memorize
one successful route before trying to improve random-scene generalization.

Keep the complete multi-route environment, its 14 branch slots, 341 observation
features, eight actions, original physics, reward and 200-action budget. Use only
training scene seed **10000**, the first previously used demonstration seed.
Execute the A* route through the actual environment and require a successful,
continuous trajectory. No validation or reserved final-test scenes are used.

Compare three objectives, each starting from exactly the same seed-0 weights:

| Arm | Objective | Purpose |
| --- | --- | --- |
| `td_margin` | Production one-step Huber TD loss + action-margin loss, weight 1 and margin 0.8 | Test the existing combined objective on one route. |
| `margin_only` | The same action-margin loss alone | Isolate the addition of TD in the preceding arm. |
| `cross_entropy` | Cross-entropy over the eight action scores | Check whether a different supervised objective can fit the labels. This is not RL training. |

All arms use the existing 256/256 ReLU MLP, observation encoder version 1,
Adam learning rate 0.0003, gradient clipping at 10, and exactly **2000 updates**.
Each update uses the entire single trajectory in the same order. Its actual
length is recorded as the effective batch size; the pilot configuration's batch
size 64 is unused here. The TD arm synchronizes its target every 200 updates.
The supervised controls do not use a target network, rewards or next observations.
There is no online exploration, early stopping, tuning between arms or checkpoint
selection by success. This protocol differs from attempt 004's ten-scene sampled
minibatches, so it is not a direct performance comparison against attempt 004.

At updates **0, 500, 1000 and 2000**, measure agreement on the full fixed training
set and perform a greedy neural rollout from the same scene reset. Record losses,
confusion matrices, first reference-action divergence, success, damage, task
budget exhaustion and movement statistics. Save every scheduled checkpoint, all
update records, exact input settings, source/dataset/model hashes and run status.
Keep demo collection actions and evaluation actions separate from optimizer
updates. Counts exclude planner search and reset-time feasibility probes.

## Reproduction

From the repository root in the project environment, with a new output directory:

```powershell
python docs/experiments/005_fixed_scene_fit/run_fixed_scene_fit.py --output-dir results/training/fixed_scene_fit_20261110
```

## Interpretation rules

- Full action agreement plus a successful neural rollout demonstrates fitting of
  this one deterministic trajectory. It does not demonstrate a generally capable
  RL agent: the remaining-budget feature can encode route order, and the scene
  contains no variation during fitting.
- Better margin-only fitting than TD+margin would implicate the addition of TD
  under these settings and budget. It would not establish a universal DQN flaw.
- CE differs in loss form and scale as well as removing TD; it cannot isolate TD
  by itself. Its outputs are classification scores, not return estimates.
- Partial action agreement is insufficient to infer rollout success. An early
  wrong action can take the policy outside the demonstrated states, and expert
  labels need not be the only successful action choices.
- Failure of all three arms would still leave optimization budget, feature
  conditioning and objective choices unresolved; it would not prove the scene
  impossible. The physically executed reference route tests feasibility.

## Outcomes

Executed from commit `c90b0a6` on 10 November 2026, 09:38:02--09:39:19 UTC.
The single reference route succeeded in **35 steps**, return **107.6108**, without
deformation or damage. All arms used those same 35 transitions and identical
initial parameter hashes. There were no contradictory duplicate input labels.

| Arm | Final full-data agreement | Neural rollout | First wrong reference action |
| --- | --- | --- | --- |
| TD + margin | 19/35 (54.3%) | Budget exhausted at 200 steps | Step 6: east-small instead of north-small |
| Margin only | 25/35 (71.4%) | Budget exhausted at 200 steps | Step 6: east-small instead of north-small |
| Cross-entropy | 25/35 (71.4%) | Budget exhausted at 200 steps | Step 6: east-small instead of north-small |

All twelve scheduled rollouts failed without damage. At the final checkpoint the
policies stalled near the first gate, with 192--193 stationary actions out of 200.
Final returns were approximately 1.5953 and the target remained about 0.679 away.
The three arms made **6000 optimizer updates** in total; collection used **35
actions** and scheduled neural evaluation used **2400 actions**. An additional
**600 actions** independently verified the three reloaded final checkpoints;
their metrics, full trajectories and saved hashes matched exactly.

This is underfitting even on a known scene, not merely a failure on new layouts.
Adding TD reduced agreement relative to the identical margin-only control in
this particular run, but neither supervised control solved the route either.
All curves were still improving at the tested checkpoints. Only five input
dimensions varied along the collision-free demonstration (robot position, target
offset and remaining budget); this observation does not prove an encoding bug.
The next recorded diagnostic will change only the update budget, to test whether
the present fitting failure persists with more optimization.

Verification: **218 tests passed**, Ruff passed, all 48 launch source hashes and
the runner hash matched, all 12 scheduled checkpoint hashes were checked, and
the final policies were independently reloaded. No production learning defaults,
reward rules, environment generation or submitted reports were changed.

Evidence: [figure](005_fixed_scene_fit/evidence/fitting_diagnostic.png),
[summary](005_fixed_scene_fit/evidence/summary.json),
[reload verification](005_fixed_scene_fit/evidence/reload_checks.json), and
[archive manifest](005_fixed_scene_fit/evidence/archive_manifest.json).
The archive includes every checkpoint observation and first/every-100th/last
update samples; full update logs, encoded data and weights remain in the local
ignored run directory. The manifest records full-log hashes and sample coverage.
