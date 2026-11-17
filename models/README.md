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

## Own-route attempt 004: experimental five-label continuations

`your_method_004_control.pt` and `your_method_004_targeted.pt` are fixed endpoints
after 2000 additional updates from the attempt-034 targeted full snapshot. Only
five existing verified-action supervision weights differ (1 versus 2). Control
achieves 12/33 training successes and 2/3 targeted-scene successes; treatment
achieves 10/33 and 1/3. Treatment retains the original three scenes but loses
source successes on 290017/290022. These are experimental candidates, not a
validated replacement or a generalization result.

See [own-route attempt 004](../docs/experiments/your_method/004_verified_supervision/RESULTS.md)
for all failures, update history and checks. Full Adam-resume states are retained
in that attempt's `resume/` directory, with final TD/Adam clocks 14000/34000.
Model hashes are recorded in its `evidence/model_exports.json`. Prior models
remain available.

```bash
python scripts/evaluate_dqn.py --checkpoint models/your_method_004_targeted.pt --episodes 3 --seed 10000
```

## Own-route attempt 008: bounded batch-recovery supervision

`your_method_008_round1.pt` and `your_method_008_round2.pt` are experimental
supervised-only endpoints after 6000 and 12000 cumulative updates. They use the
same network/encoder/environment as `your_method_004_targeted.pt`, with 4103
successful recovery-route labels collected in two rounds. No new TD updates
were performed; their outputs are action scores, not calibrated Q estimates.

| Endpoint | Original scenes | Training scenes | Development validation |
| --- | --- | --- | --- |
| Source attempt-004 targeted | 3/3 | 10/33 | 3/30 |
| Attempt-008 round 1 | 3/3 | 9/33 | 5/30 |
| Attempt-008 round 2 | 2/3 | 11/33 | 5/30 |

Neither new endpoint satisfies the predeclared retention rule. Round 1 loses
training coverage; round 2 times out on 10000. Retain the source as the current
baseline; both experimental checkpoints remain available for reproducibility.
The 30 validation scenes were excluded from training but used for development
selection. No final untouched test set was used.

See [attempt 008](../docs/experiments/your_method/008_batch_recovery/RESULTS.md)
and its [model hashes and selection record](../docs/experiments/your_method/008_batch_recovery/comparison.json).
Each round directory contains an optimizer/RNG resume checkpoint, full training
data and evaluation traces. The run is stopped at the two-round limit.

```bash
python scripts/evaluate_dqn.py --checkpoint models/your_method_008_round2.pt --episodes 3 --seed 10000
```

## Own-route attempt 009: geometry correction with retained behavior

`your_method_009_geometry.pt` is a format-2 experimental policy combining the
immutable attempt-004 source network with a geometry-aware residual policy.
Use the separate loader below; its outputs are policy scores, not calibrated
Q values. No new TD updates were performed. The source file remains intact.

It retains all ten source training successes (10/33 overall, original 3/3) and
improves development validation from 3/30 to 4/30, meeting the declared candidate
eligibility rule. Training damage nevertheless increases from 11 to 12 scenes.
The pilot stopped at 400 attempted updates: one 20-update block was accepted at
scale 0.125 and 19 blocks were rolled back. The surviving Adam clock is 20.
This is a provisional own-route candidate, not evidence of reliable generalization.

See [attempt 009](../docs/experiments/your_method/009_geometry_retention/RESULTS.md)
for the data, objective conflicts, complete evaluation, and model hashes.
Optimizer/RNG state is in that attempt's `evidence/resume.pt`. Its 30 development
validation scenes remain separate from training but are not an untouched final test.

```bash
python scripts/evaluate_recovery_policy.py --checkpoint models/your_method_009_geometry.pt --episodes 3 --seed 10000
```
