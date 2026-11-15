# Verified inference policy

`step6_repair_027.pt` is the final TD-10000 inference checkpoint from
[attempt 027](../docs/experiments/027_step6_supervision/RESULTS.md). It includes
network weights, encoder version and the environment configuration. It can be
loaded without the local historical experiment outputs.

From the repository root in the configured project environment:

```bash
python scripts/evaluate_dqn.py --checkpoint models/step6_repair_027.pt --episodes 3 --seed 10000
```

This evaluates known training scenes 10000, 10001 and 10002. The verified
results are success in 35, 41 and 41 steps, respectively. No inference-time
correction or scripted prefix is used by this command. A subsequent frozen
evaluation on 100 unseen scenes (seeds 280000--280099)
succeeded in only 1/100 cases, with 50 damage and 49 budget failures. See
[attempt 028](../docs/experiments/028_frozen_generalization/RESULTS.md). This
checkpoint demonstrates known-scene retention, not reliable generalization; its
weights are unchanged by that evaluation.

`step6_repair_027.json` records the SHA-256, source update, independent normal
evaluation and scope. The checkpoint is for inference, not resuming Adam state.
Full training-state snapshots remain in the local experiment output. Running
the generic trainer uses its existing defaults; it does not automatically
reproduce the staged calibration and per-observation supervision of this repair.

SHA-256: `06733fccb5422d10ad7246e5368bbd5ed9cc3367f0b413847544ec26a96b0c88`.

## Attempt 029 experimental candidates

`coverage_029_control.pt` and `coverage_029_expanded.pt` are paired final
TD-18000 inference candidates from [attempt 029](../docs/experiments/029_training_coverage/RESULTS.md).
Both achieve 4/100 on test seeds 390000--390099. The control retains 3/3 original
scenes; the expanded candidate retains 1/3 and is not a recommended replacement.
Adjacent JSON files record hashes and outcomes. The original attempt-027 file
is unchanged. These are inference files, not Adam-resume snapshots.

```bash
python scripts/evaluate_dqn.py --checkpoint models/coverage_029_expanded.pt --episodes 100 --seed 390000
```

## Attempt 030 staged-training endpoints

`stages_030_supervised.pt`, `stages_030_calibration.pt` and `stages_030_td.pt`
are the fixed phase endpoints from [attempt 030](../docs/experiments/030_expanded_stages/RESULTS.md).
The network is unchanged. Training across 33 scenes uses 20000 supervised,
20000 actual-return calibration and 10000 TD updates; calibration Adam state
is retained exactly into TD.

Final TD restores two of three original scenes (10000 and 10001); 10002 fails
from damage. It succeeds on 5/30 added training scenes and 13/100 fresh test
scenes, versus 9/100 for the frozen attempt-029 source on the same test set.
Eleven test successes are gained and seven are lost; reliable generalization
has not been established. The pre-TD endpoints have no fresh holdout score.
Adjacent JSON files contain hashes and measured scope. These are inference
files; the portable full starting state is in the attempt's `inputs/` folder,
and intermediate/final resume states remain in its local run directory.

```bash
python scripts/evaluate_dqn.py --checkpoint models/stages_030_td.pt --episodes 3 --seed 10000
```

[Attempt 031](../docs/experiments/031_scene10002_rescue/RESULTS.md) freezes this
same final TD policy and diagnoses scene 10002. Replacing only its step-23
`north_small` with `north_large` does not restore success: the original 58-step
damage failure becomes a 200-step blocked budget failure, with a later reference
disagreement at step 27. No training or policy replacement occurred. Its complete
final Adam-resume snapshot is now tracked at
`docs/experiments/031_scene10002_rescue/inputs/final_td_state.pt`.

[Attempt 032](../docs/experiments/032_scene10002_step27/RESULTS.md) holds that
step-23 correction fixed and additionally replaces step 27 with `north_small`.
This also fails: the 200-step blocked control becomes a 58-step damage failure,
with a further reference disagreement at step 28. Independent reloads reproduce
both trajectories; model parameters, Adam state and all training settings remain
unchanged. These are assisted diagnostic interventions, not a repaired policy.

[Attempt 033](../docs/experiments/033_scene10002_step28/RESULTS.md) additionally
corrects step 28 to north_large, keeping the step-23/27 corrections fixed. This
restores assisted success in 42 steps, verified by an independent reload. All
other actions are autonomous, including a harmless reference disagreement at
step 33. The model itself remains unchanged; autonomous repair is not established.

## Attempt 034 autonomous supervision candidates

`supervision_034_targeted.pt` and `supervision_034_control.pt` are matched fixed
2000-update continuations of the final attempt-030 policy. Only three existing
scene-10002 supervision weights differ (2 versus 1). Targeted final success is
3/3 on original scenes, in 40/47/43 steps; control achieves 2/3. Both arms have
intermediate 10002 failures, and targeted 10000 needs two extra final steps.
This is known-scene recovery, not continuous retention or verified generalization.

See [attempt 034](../docs/experiments/034_scene10002_supervision/RESULTS.md).
Adjacent JSON files contain hashes and outcomes. Full final Adam-resume states
are tracked in the experiment's `checkpoints/` directory. Prior models are unchanged.

```bash
python scripts/evaluate_dqn.py --checkpoint models/supervision_034_targeted.pt --episodes 3 --seed 10000
```
