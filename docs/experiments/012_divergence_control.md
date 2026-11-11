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
