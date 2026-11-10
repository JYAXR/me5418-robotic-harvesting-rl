# Attempt 006: fixed-scene optimization budget

## Protocol recorded before execution

Attempt 005 could not fit all 35 actions even on its single known multi-route
scene. Its three objectives were still improving after 2000 updates, and none
completed a neural rollout. Before changing the encoder or the environment,
repeat the same comparison with **20000 updates per arm**.

All training inputs, architecture, encoder, seed-0 initialization, optimizer,
learning rate, clipping, objective definitions, TD target synchronization,
training scene seed 10000 and full-batch protocol remain as in
[attempt 005](005_fixed_scene_fit.md). All three arms restart from scratch;
there is no incomplete optimizer-state resume. There is no early stopping,
adaptive tuning or access to validation/reserved final-test scenes.

Evaluate at updates **0, 500, 1000, 2000, 5000, 10000 and 20000**. The first four
snapshots and first 2000 update records should exactly reproduce attempt 005;
verify and report this rather than assuming it. Keep all later checkpoints even
if performance deteriorates. The total optimization budget is **60000 updates**;
collection and evaluation actions are counted separately. No online RL is run.

```powershell
python docs/experiments/006_fixed_scene_budget/run_extended_fit.py --output-dir results/training/fixed_scene_budget_20261110
```

The extended driver delegates to the unchanged attempt-005 runner. Both script
hashes, launch source hashes, settings, data and checkpoint hashes are retained.
Verify final saved checkpoints with the attempt-005 reload helper; their outputs
must reproduce both whole-dataset fitting metrics and greedy episode traces.

## Interpretation

A successful fixed-scene fit would establish that the current network and
encoder can represent this one route under the tested optimizer. It does not
show that the policy understands geometry: budget input and fixed scene features
can support memorization. Supervised-only success is not a trained RL result.
If only supervised controls succeed, investigate the combined TD objective
before investing in broader random-scene training. If TD+margin also succeeds,
the earlier failure was not a hard inability to fit this route, and training
duration, coverage and generalization remain separate questions. Do not infer a
generalization improvement or claim that more updates always solve training.

## Outcomes

Executed from commit `49da583` on 10 November 2026, 09:44:39--09:49:03 UTC.
The dataset, initial parameters, first 2000 update records in every arm, first
four observation/rollout records and all corresponding checkpoint tensors
**exactly reproduced attempt 005**. The comparison was captured while the longer
run was still running; its status field truthfully records that time. Final run
completion and the later checkpoints were verified separately.

| Objective | Agreement after 2000 updates | Agreement after 20000 updates | Final greedy rollout |
| --- | --- | --- | --- |
| TD + margin | 19/35 (54.3%) | 33/35 (94.3%) | Damage after 49 steps; return -59.1592 |
| Margin only | 25/35 (71.4%) | 35/35 (100%) | Success in 35 steps; return 107.6108 |
| Cross-entropy | 25/35 (71.4%) | 35/35 (100%) | Success in 35 steps; return 107.6108 |

The margin-only arm first succeeded at the **10000-update sampled checkpoint**
(33/35 action matches, 39 rollout steps). CE first succeeded at the **5000-update
sampled checkpoint** (34/35 action matches, 35 rollout steps), reaching perfect
demonstration agreement at 10000. These are the earliest observed successful
checkpoints, not precise learning-onset times. Their partial-agreement successes
also confirm that reference labels are not the only possible successful choices.

At 20000 updates both supervised controls exactly reproduced the reference
action sequence and remained damage-free. TD+margin first deviated at action 27:
it chose east-large where the reference chose south-large, then damaged a branch.
Its 10000-update checkpoint also failed by damage. Retain both failures; improved
training-set accuracy alone did not ensure safe closed-loop execution.

Accounting: **60000 optimizer updates**, **35 reference-collection actions** and
**3076 scheduled neural evaluation actions**. Independently reloading the three
final checkpoints added **119 verification actions** and no optimizer updates.
No validation or held-out test scene was accessed. Search/reset feasibility
probes remain excluded from these interaction counts.

## Conclusions and next development step

The current encoder and MLP can fit and execute this fixed route. The failure of
the supervised controls at 2000 updates was therefore not a hard representational
barrier: increasing only their optimization budget resolved it. The single
trajectory can still be memorized using the budget input; this does not establish
generalization, task understanding or completion of the final RL requirement.

Under a matched 20000-update budget, adding TD to the same margin objective still
gave worse fitting and execution. This implicates that combined training setup
for this scene and initialization, but does not identify a universal TD error or
explain every earlier multi-scene failure. Gradient scale, moving bootstrap
targets and finite optimization may each matter; none was separately tested.

The next justified test is staged learning: fit demonstrations with a supervised
objective first, verify execution, then introduce TD updates while checking
whether the acquired behavior is retained. Expand to a small fixed set of
training scenes before broader random-scene training, and keep independent
validation scenes separate. This is a proposed next experiment, not an already
implemented production change or a claim that supervised policies replace RL.

## Verification and retained evidence

The unchanged runner had passed **218 tests** in attempt 005. The extension and
analysis helpers passed Ruff; the extension delegates to that same runner.
All launch source and driver hashes were checked. All 21 checkpoint hashes and
all three 20000-row update sequences were verified, including 100 TD target
synchronizations and zero in the supervised controls. Final reloaded checkpoints
exactly reproduced their saved metrics and full rollout traces.

Evidence: [figure](006_fixed_scene_budget/evidence/fitting_diagnostic.png),
[summary](006_fixed_scene_budget/evidence/summary.json),
[exact common-prefix comparison](006_fixed_scene_budget/evidence/prefix_comparison.json),
[reload verification](006_fixed_scene_budget/evidence/reload_checks.json), and
[archive manifest](006_fixed_scene_budget/evidence/archive_manifest.json).
As in attempt 005, the archive retains all checkpoint observations, first/every
100th/last update samples and full-log hashes. Models, encoded data and complete
update logs stay in the ignored local run directory. Production training defaults,
environment behavior and reports remain unchanged.
