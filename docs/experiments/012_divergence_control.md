# Attempt 012: first-divergence diagnosis and one-factor TD control

## Diagnostic protocol recorded before physical replay

Attempt 011 acquired three known training routes and calibrated their demonstrated
discounted returns. Keeping Adam state reduced TD failures, but scene 10001 still
exhausted its task budget at updates **1, 6, 9, 10 and 200**. This attempt first
examines those failures, then chooses one scalar intervention from that evidence.
It does not retrain supervision or calibration, select a new scene, or modify the
environment, network, reward or demonstration data.

Use the committed attempt 011 evidence and retained raw artifacts as immutable
inputs. Verify hashes before analysis. Examine all 18 preserved-state checkpoints,
including successful checkpoints as controls. At the first action disagreement,
the preceding action prefix and encoded state must match the demonstration.
Report all eight Q scores, selected and demonstrated actions, the signed gap
`Q(demonstrated) - max(Q(other))`, the top-two gap and the shortfall from the 0.8
action margin. Evaluate one observation at a time, exactly as greedy action
selection does. Compare against the same state's calibrated initial scores.
After a policy leaves the demonstrated route, the demonstration's action at the
same time index is not automatically a correct label for its new state.

Q-only extraction from existing checkpoints found the following before any new
physical replay or training:

| Failed TD update | First divergent step | Policy action | Demonstrated action | Signed Q gap | Calibrated gap |
| --- | --- | --- | --- | --- | --- |
| 1 | 6 | east_small | south_small | -0.0169678 | +0.2810135 |
| 6 | 6 | east_small | south_small | -0.1030655 | +0.2810135 |
| 9 | 18 | east_large | north_small | -0.3432541 | +0.8334656 |
| 10 | 18 | east_large | north_small | -0.1132813 | +0.8334656 |
| 200 | 6 | east_small | south_small | -0.0366898 | +0.2810135 |

These are action-label transitions: the preceding demonstrated action is
east_small before step 6 and east_large before step 18. Successful checkpoints
also contain disagreements, including a different eastward step size at step 24.
Therefore disagreement alone is not evidence of a causally harmful action.

For each of the five failed models, replay the unmodified greedy policy and
require exact agreement with the archived trajectory. Then reset the same scene,
force **only the demonstrated action at its first divergence**, and resume the
same greedy neural policy immediately. Record every action, position, Q vector,
blocked/stationary behavior and terminal outcome. This is a diagnostic
counterfactual, not a planner fallback in policy evaluation. It can show whether
correcting one choice is sufficient on a particular fixed trajectory; failure to
rescue can reveal a later independent problem. Preserve both outcomes.

No optimizer updates occur during diagnosis. Count original replay and
counterfactual environment actions separately. Planner/reset feasibility probes
are excluded from these action counts, consistently with earlier attempts.

## Controlled comparison structure

The treatment scalar and value will be selected and committed **after** the
physical diagnostic, before executing either training arm. The candidate runner
requires an explicit factor/value and has no treatment default.

Both arms restore the identical attempt 011 calibration-end full training
snapshot, including online weights and Adam moments/internal steps. Both copy
online weights to frozen targets, clear gradients and reset the TD phase clock
to zero. The starting Adam internal step is 20000. Reuse the same ordered 117
transitions and independently calculated return targets, with 35/41/41 samples
from training seeds 10000/10001/10002. The full-batch objective is weighted equally
per transition. No held-out scenes or online exploration are introduced.

The baseline repeats preserved-state TD from attempt 011 without changes:
learning rate 0.0003, demonstration weight 1, margin 0.8, target interval 200 and
gradient clipping 10. Require exact reproduction of all 2000 update records and
all 18 observations, including optimizer summaries, before beginning treatment.
The treatment changes exactly one declared scalar. It receives the same 2000
updates and evaluations at **0 through 10, then 20, 50, 100, 200, 500, 1000 and
2000**, across all three training scenes. No early stopping, rollback or model
selection based on evaluation success is allowed.

Primary comparisons are all-scene retention at every sampled checkpoint and the
number of first-ten checkpoints where all three scenes succeed. Also retain each
scene's failures, action agreement, demonstrated-return error, TD and margin
losses, gradient norms, parameter changes and complete rollouts. Final success
alone cannot erase earlier failures. Checkpoint fractions describe repeated
checks of three training scenes, not independent generalization success rates.

The comparison uses **4000 new TD updates**, reusing a shared historical 40000
supervision/calibration updates. Count the 108 scheduled evaluation episodes and
their actual actions separately from diagnosis. Independently replay the first
ten updates per arm (20 verification updates) and reload initial/update-1/final
models from both arms on all three scenes (18 verification episodes). Retain all
failed outcomes and source/input hashes. Full model/data artifacts stay in the
ignored results directory; reviewable evidence and this record are committed.

## Interpretation limits

One initialization and three known routes cannot establish generalization or
online RL performance. The remaining-budget input may support memorization.
Changing a loss weight can also change the total gradient and clipping behavior;
changing a learning rate changes update magnitude. A successful control can
support the proposed intervention in this setting without proving a unique
underlying cause. Calibration's temporary failures in attempt 011 remain part of
the history and are outside this TD-only comparison.

## Completed diagnosis and predeclared treatment

The physical diagnostic ran from commit `c1eec38` on **12 November 2026,
00:40:42--00:42:23 Asia/Shanghai**. All five unmodified policies exactly
reproduced their recorded 200-step failures. Correcting only the first divergent
choice restored success in **all five** cases, after which every action again
came from the unchanged neural policy:

| TD update | Forced action and step | Original blocked / stationary steps | Corrected rollout steps | Corrected peak deformation |
| --- | --- | --- | --- | --- |
| 1 | south_small at 6 | 171 / 170 | 41 | 0.000000 |
| 6 | south_small at 6 | 171 / 170 | 41 | 0.000000 |
| 9 | north_small at 18 | 165 / 164 | 41 | 0.075780 |
| 10 | north_small at 18 | 165 / 164 | 42 | 0.685353 |
| 200 | south_small at 6 | 171 / 170 | 41 | 0.000000 |

At step 6, continuing east instead of turning south starts an off-route contact
sequence; the robot eventually keeps pushing against a branch. At step 18,
taking another large eastward step moves the robot too far right before its
northward turn, leading to a later prolonged blocked push. The failures are not
two-action oscillations. Exact common-state checks and the single-action rescue
support these choices as causally consequential on these fixed rollouts.

Four corrected rollouts subsequently differ from the demonstration at step 24
and still succeed; update 200 follows the demonstration after the intervention.
Update 10's corrected route has appreciable deformation and
a lower return than the other corrected routes. The intervention therefore
restores task completion, not necessarily identical route quality. The
calibrated policy's step-6 gap was already only 0.2810, below the desired 0.8
margin; TD reverses that fragile ordering at the failed checkpoints.

The diagnostic used **zero optimizer updates**, **10 episodes and 1206 actions**:
1000 original-greedy and 206 counterfactual actions. All 18 Q analyses and every
physical step are retained under `results/training/divergence_diagnostic_20261112`.
The completed summary SHA-256 is
`6b3d0e20f835efdbbb113db2cd98542f545e17d05a2dc148a5555494a919111f`;
the canonical Q-analysis SHA-256 is
`f899a4a24c552e890f5fff8b295e4231a42e15cdb42514da3d7b65b93760d98b`.

Based on this diagnosis, the **only treatment change is demonstration weight
1.0 to 2.0 during TD**. Both arms retain learning rate 0.0003, margin 0.8 and
the same calibrated Adam moments and internal steps. Doubling is a predeclared
moderate strengthening of the existing action-ranking objective, not a value
selected by trying several weights. It tests whether more supervision reduces
the observed turn-action ordering regressions. No counterfactual actions are
used in either arm's evaluation, and no diagnostic trajectories enter training.

The total objective is `TD_Huber + weight * demonstration_margin_loss`.
Changing the weight may change both gradient direction and clipping scale, so
we also examine component losses, gradient norms and parameter changes. This
control does not isolate a single gradient component or guarantee retention.
If it fails, retain that result rather than adding another intervention here.

### Commands

```powershell
python docs/experiments/012_divergence_control/diagnose_divergence.py --reference-run results/training/multiscene_retention_20261112 --output-dir results/training/divergence_diagnostic_20261112
python docs/experiments/012_divergence_control/run_control.py --reference-run results/training/multiscene_retention_20261112 --archive-dir docs/experiments/011_multiscene_retention/evidence --output-dir results/training/divergence_control_20261112 --factor demonstration_weight --value 2
python docs/experiments/012_divergence_control/verify_control.py --run-dir results/training/divergence_control_20261112 --output results/training/divergence_control_20261112/reload_checks.json
```
