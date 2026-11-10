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

Pending the predeclared run. Preserve all three arms and all scheduled outcomes.
