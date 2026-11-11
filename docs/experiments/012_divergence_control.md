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
python docs/experiments/012_divergence_control/analyze_control_q.py --run-dir results/training/divergence_control_20261112 --output results/training/divergence_control_20261112/q_analysis.json
```

## Controlled outcomes

The paired run executed from commit `4d4943a` on **12 November 2026,
00:44:35--00:54:37 Asia/Shanghai** (11 November 16:44:35--16:54:37 UTC).
Read-only Q/component analysis and figure labeling were committed in `84078e1`
before their execution; the running training implementation remained unchanged.

The baseline exactly reproduced **all 2000 update records, all 18 observations
and the complete final training snapshot** from attempt 011. This includes
weights, target weights, Adam state, counters and RNG state. The gate passed
before the treatment began. Both arms then finished their fixed budgets without
early stopping or checkpoint selection.

| Measurement | Weight 1 baseline | Weight 2 treatment |
| --- | --- | --- |
| All-three-scene successes during updates 1--10 | 6/10 checks | 8/10 checks |
| Individual scene successes during updates 1--10 | 26/30 | 28/30 |
| All-three-scene successes across all scheduled checks, including update 0 | 13/18 | 14/18 |
| Individual scene successes across all scheduled checks | 49/54 | 48/54 |
| Seed 10000 failed updates | None | 50 |
| Seed 10001 failed updates | 1, 6, 9, 10, 200 | 5, 10, 50, 200 |
| Seed 10002 failed updates | None | 50 |
| Scenes successful at every sampled checkpoint | 10000 and 10002 | None |
| Final demonstrated action agreement | 117/117 | 117/117 |
| Final actual route lengths | 35/41/41 | 35/41/41 |
| Final demonstrated-return RMSE | 0.563084 | 0.704667 |
| Strict all-scene sampled retention | False | False |

Every failure was a 200-step budget exhaustion. The treatment reduced early
failures, but introduced a simultaneous failure of all three scenes at update
50. It recovered all three at update 100, lost seed 10001 at 200, and recovered
all three at 500, 1000 and 2000. Both arms ended successfully. The increase in
all-scene checkpoint count and decrease in individual scene success count are
compatible: the treatment concentrated three failures into one checkpoint.
Neither count is an independent test success rate, and the evidence does not
support an unqualified overall improvement.

### Action ordering after the intervention

All Q measurements use batch-of-one inference on the same recorded reference
states. A positive signed gap means the demonstrated action ranks first. The
analysis explicitly labels reference states that lie beyond the actual common
trajectory prefix; those scores are not presented as states the agent visited.

| TD update / reference step in seed 10001 | Baseline signed gap | Treatment signed gap | Observed difference |
| --- | --- | --- | --- |
| 1 / 6 | -0.016968 | +0.015060 | The treatment retains the turn and succeeds, but the gap remains very small. |
| 5 / 6 | +0.289116 | -0.180321 | The treatment newly misses the turn and times out. |
| 6 / 6 | -0.103065 | +0.123497 | The baseline fails; the treatment succeeds. |
| 9 / 18 | -0.343254 | +0.385765 | The baseline fails; the treatment succeeds. |
| 10 / 18 | -0.113281 | +0.737534 | The treatment preserves this turn but first deviates later, at step 22. |
| 50 / 6 | +0.106476 | -0.354515 | Seed 10001 newly fails in the treatment's all-scene collapse. |
| 200 / 6 | -0.036690 | -0.024712 | Both arms fail despite the smaller treatment reversal. |

At treatment update 10, the first actual disagreement moves to step **22**:
`north_small` outranks the demonstrated `north_large` by **0.051712**. Improving
the earlier turn's ranking therefore does not guarantee the remaining route.
At treatment update 50, seed 10000 first chooses `east_small` instead of
`north_small` at step **32** (signed gap **-0.239342**), while seed 10002 chooses
`east_small` instead of `south_small` at step **6** (gap **-0.092140**).
These are verified first disagreements and failed rollouts; new one-action
counterfactuals were not performed for the treatment failures.

By update 2000, treatment gaps at seed 10001's reference steps 6 and 18 are
**0.888466 and 1.109108**, compared with **0.209938 and 0.115250** in the baseline.
This stronger final separation does not erase the treatment's intervening
failures. Successful step-24 disagreements are also retained as controls.

### Loss and update behavior

Doubling the loss weight did not simply enlarge every parameter step. The first
pre-clipping gradient norm rose from **28.1947 to 40.2989**, while the actual
parameter-change norm fell from **0.00452334 to 0.00399864** under the retained
Adam state and fixed clipping threshold. Both first combined losses rose
slightly; they have different weights and must not be ranked as the same
objective. Their initially identical unweighted TD loss fell from 0.114103 to
0.092785 in the baseline and to 0.096026 in treatment. Their initially identical
unweighted margin loss rose from 0.062617 to 0.086850 and 0.073886, respectively.

Across 2000 updates, the mean unweighted margin loss before updating fell from
**0.086709 to 0.068774**, but the mean pre-clipping gradient norm rose from
**37.3744 to 57.0620**. Norms exceeded the clipping threshold of 10 on
**1996/2000** baseline updates and **2000/2000** treatment updates. Mean actual
parameter-change norms were 0.00562691 and 0.00581951. Later TD-loss averages use
different evolving target networks and are descriptive, not an identical-target
comparison. The final MC-return RMSE comparison does use the same fixed targets.

The measured effect is thus a change in the TD/supervision tradeoff and update
dynamics, not proof that gradient clipping alone caused the failures. Lower
average action-margin loss is insufficient to guarantee correct decisions at
every important state.

## Conclusion

The diagnosis identifies a concrete failure chain on the original five failed
policies: a turn-action Q ranking reverses, the robot leaves the known route,
and prolonged blocked pushing exhausts the budget. Correcting that first action
alone rescues all five. However, **globally doubling TD supervision weight is
not a stable retention fix**. It helps some early decisions, introduces other
failures and leaves both strict retention flags false. The existing production
defaults are unchanged; this attempt does not promote weight 2 as a validated
replacement or claim final project readiness.

All 117 training samples remain on demonstration routes, so this experiment
does not train recovery behavior after an off-route mistake. A later experiment
can separately examine TD step size under the calibrated, Adam-preserved setup,
or recovery data, but should change only one of those factors at a time. Neither
additional intervention was tested in this attempt.

## Verification and evidence

The full suite passed **374 tests**; Ruff passed. New experiment code, comments
and runtime messages are in English. The read-only Q analysis checked all 74
saved model files (36 inference checkpoints and 38 full training snapshots),
verified identical initial Q vectors on all 117 states, and reproduced stored
actions along every common demonstration prefix without resetting or stepping
an environment. The outcome figure was visually inspected.

Main-run work was **4000 new TD updates and 108 evaluation episodes / 5971
actions**. The shared **40000 historical supervision/calibration updates** were
reused, not rerun or counted twice. Diagnosis added the separately recorded
10 episodes / 1206 actions and zero updates. No new demonstrations, online
exploration or held-out scenes were used. Independent verification work is
reported separately below.

Independent verification passed: 52 unchanged production source hashes,
committed reference evidence, every stored model, all 4000 update records and
all per-scene observations were checked. The first ten updates per arm were
independently recomputed, reproducing exact metrics, online/target tensors and
Adam state. Initial/update-1/final policies from both arms reproduced exact
metrics and trajectories across all three scenes, including the baseline's
update-1 failure. This added **20 verification updates, 18 episodes and 861
actions**, separately recorded in `reload_checks.json`. Verification does not
numerically replay all 2000 updates: at checkpoint 500, the stored target is
integrity-checked but its unsaved update-400 origin is not independently rerun.

The original diagnostic was also independently checked against its raw step
logs: all ten trace files agree, each counterfactual forces exactly one action,
and every subsequent action equals the unchanged policy's argmax. The diagnostic
and control keep separate accounting. Reset feasibility and planner probes are
excluded consistently from executed policy-action totals.

All scheduled outcomes, all first-divergence diagnostics, complete diagnostic
traces and first-ten/every-50th/last update samples are archived, including
unsuccessful trials. Full model checkpoints and raw update logs remain locally
under ignored `results/training/` directories. Earlier evidence and the user's
existing report files were not modified.

Evidence: [all scheduled outcomes](012_divergence_control/evidence/control/control_outcomes.png),
[original failure Q analysis](012_divergence_control/evidence/diagnosis/q_analysis.json),
[one-action diagnostic outcomes](012_divergence_control/evidence/diagnosis/summary.json),
[paired control summary](012_divergence_control/evidence/control/summary.json),
[post-control Q and component analysis](012_divergence_control/evidence/control/q_analysis.json),
[independent verification](012_divergence_control/evidence/control/reload_checks.json),
and [archive manifest](012_divergence_control/evidence/archive_manifest.json).
