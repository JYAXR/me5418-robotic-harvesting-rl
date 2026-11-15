# Attempt 027: repair the early turn while preserving learned recovery

Completed on 15 November 2026. The [protocol](../027_step6_supervision.md),
update implementation, runner and tests were committed before execution in
`a546fa43fa642c6756ac6c8f76791b93585b391c`. Doubling only the existing step-six
supervision weight removes both residual attempt-025 failures. All 105 declared
three-scene checks pass, including 69 matched historical checks and 36 additional
early checks. The final policy is exported for normal inference. This establishes
a repair on the measured known scenes, not generalization to unseen layouts.

## Isolated training change

Attempt 026 established that one south_small correction at scene 10001 step 6
rescues the frozen TD-2004 and TD-2009 failures. This attempt changes the learned
policy instead: original demonstration row 40 receives weight 2 rather than 1.
It resumes the same complete TD-2000 state as attempt 025 and performs 8000
updates through absolute TD 10000. The objective is:

```text
0.5 * mean_TD_Huber(original 117 transitions)
+ weighted_original_hinge_sum / 117
+ recovery_state_hinge(north_large, margin=0.8) / 117
```

Original row 40 (step 6, south_small) and row 55 (step 21, north_small) have
weight 2; all other original rows have weight 1. Only row 40 changes relative
to the control. The one off-reference recovery observation and its weight 1
are reused byte-for-byte. No new transition is added to TD. The denominator
117, margin 0.8, learning rate 0.0001, gamma 0.99, target interval 200, gradient
clip 10, inherited Adam moments/clocks and RNG remain unchanged. There is no
MC constraint, additional pretraining, early stopping or inference action override.
The environment and network architecture are unchanged. Evaluation is checked
to preserve the complete training state.

## Autonomous episode results

The immutable attempt-025 results supply the 69 matched control checkpoints.
They are reused rather than rerun or counted as new simulation. The treatment
also checks every update from TD 2000 through 2050, adding 36 checkpoints to
that schedule. Every action in these full episodes is selected by the network.

| Matched measurement | Attempt 025 | Attempt 027 |
| --- | ---: | ---: |
| All three scenes successful | 67/69 | 69/69 |
| Successful individual episodes | 205/207 | 207/207 |
| All-scene success at TD 2004 and 2009 | 0/2 | 2/2 |
| All-scene success at TD 2001--2010 | 8/10 | 10/10 |
| All-scene success at TD 9946--9966 | 21/21 | 21/21 |
| Final scene successes | 3/3 | 3/3 |

The additional 36 checkpoints yield 108/108 successful episodes. Across all
105 treatment checkpoints, all 315 episodes succeed. The complete early window
TD 2000--2050 passes 51/51 three-scene checks (153/153 episodes). This window
includes the unchanged initial snapshot; TD 2001--2050 are the 50 post-update checks.

At both previously failing models, scene 10001 now finishes autonomously in
41 steps with return 107.585050, zero blocked/stationary steps and zero peak
deformation. The controls exhausted 200 steps, with returns -104.715459 and
-104.696677, 171 blocked steps and 170 stationary steps. Both repaired models
are independently reloaded and reproduce the successful trajectories.

| Final policy, TD 10000 | Steps | Return | Damage | Blocked steps |
| --- | ---: | ---: | --- | ---: |
| Scene 10000 | 35 | 107.610812 | No | 0 |
| Scene 10001 | 41 | 107.585050 | No | 0 |
| Scene 10002 | 41 | 107.643380 | No | 0 |

## Recovery and neighboring decisions

Each checkpoint has a separate conditional probe: replay the same recorded
first 24 actions, then let the policy act freely from step 25. These probes
are not included in autonomous episode success rates. The initial TD-2000
model fails this probe, as it did before the experiment. All 104 post-update
conditional probes succeed and choose north_large at the recovery state.
On the matched schedule, both control and treatment pass 68/68 post-start probes.

After every one of the 8000 training updates, the policy selects south_small
at the original step-six input and north_large at the recovery input. Their
minimum gaps against every competing action are +0.128814697 and +0.183555603;
final gaps are +0.621139526 and +1.303817749, respectively. Thus the early-turn
repair does not erase the previously learned recovery choice.

Eight fully autonomous scene-10001 episodes also reach the exact recorded
recovery input after the same 24-action prefix: TD 2027, 2032, 2047, 2048, 2224,
2225, 2228 and 2229. They independently select north_large and succeed in 42
steps. Their prefix actions and paths exactly match the conditional probe;
no prefix is forced in these autonomous episodes.

| Reference step | Negative Q-gap updates, control | Negative Q-gap updates, treatment |
| --- | ---: | ---: |
| 6 | 10 | 0 |
| 18 | 0 | 0 |
| 20 | 0 | 0 |
| 21 | 0 | 0 |
| 22 | 663 | 636 |
| 23 | 0 | 0 |
| 24 | 8 | 6 |

Neighboring ranking changes remain, even though the sampled episodes all
succeed. Original-label agreement is perfect at 7055/8000 post-update models
and ends at 117/117, but its minimum is 114/117 (control minimum 115/117).
This adverse minimum is retained rather than treating a few monitored actions
as a guarantee about the whole policy. Final actual-return RMSE is 0.715207,
versus 0.717245 in the control; no return constraint was added during TD.

## Verification and work accounting

Four disposable gate updates check that weight 1 exactly reproduces the
historical update and complete state, and that a separately implemented
weight-2 update matches the treatment. No gate state enters training.
Independent computation then reproduces updates 2001--2201 and 9801--10000
(401 total), matching update metrics and applicable full snapshots. Target
lineage, 40 target synchronizations and inherited Adam clocks are verified.
All 8000 main updates have nonzero parameter changes.

Seven saved models (TD 2000, 2004, 2009, 2031, 2345, 9956 and 10000) are
reloaded for three full episodes and one conditional probe each: 28 matching
replays. No autonomous failures require additional post-hoc replays. The final
inference checkpoint is loaded separately through DQNAgent.load and the normal
training.evaluate function; its three episode metrics exactly match the main
run, without weight or RNG changes. It is then copied byte-identically into
models/step6_repair_027.pt.

| New simulator work | Resets | Executed actions | Copied certification actions |
| --- | ---: | ---: | ---: |
| Treatment autonomous episodes | 315 | 12295 | 12285 |
| Treatment conditional probes | 105 | 4568 | 4305 |
| Predeclared autonomous reloads | 21 | 819 | 819 |
| Predeclared conditional reloads | 7 | 452 | 287 |
| Final inference delivery check | 3 | 117 | 117 |
| Total | 451 | 18251 | 17813 |

There are 36064 total simulator actions, including copied scene-certification
work. Executed actions include 2688 scripted prefix actions in conditional
probes, with zero forced recovery actions. No new recovery collection occurs.
The 207 historical full episodes and 69 historical conditional probes are
reused evidence, not new work. Optimizer work is 8000 main + 4 gate + 401
independent = 8405 updates. Unit tests are outside these formal counts.

The focused validation command passes 41 tests; Ruff passes for the experiment
scripts and new tests. Source hashes are checked at run completion. The evidence
archive retains complete episode traces, conditional decisions, reset accounting,
model hashes, and a compact record of all 8001 boundary/post-update probes.
Selected update logs are accompanied by full raw-log SHA-256 hashes. Full
resume snapshots and full raw logs remain local; the final inference model is
included separately in Git. See evidence/archive_manifest.json and evidence/validation.json.

## Run the exported policy

From the repository root in the configured environment:

```bash
python scripts/evaluate_dqn.py --checkpoint models/step6_repair_027.pt --episodes 3 --seed 10000
```

This command evaluates the three known training scenes, not held-out data.
The model embeds its environment configuration and requires no local historical
training outputs for inference. It is not an Adam-resume checkpoint. Generic
training defaults are unchanged; this repaired policy uses the documented
staged training and selected supervision weights.

The experiment was executed with:

```bash
python docs/experiments/027_step6_supervision/run_step6_supervision.py
python docs/experiments/027_step6_supervision/verify_failure_reloads.py
python docs/experiments/027_step6_supervision/export_verified_policy.py
python docs/experiments/027_step6_supervision/analyze_step6_supervision.py
```

These archival experiment commands require the hash-bound historical artifacts
and refuse to overwrite their existing run/export/archive. They are not a
fresh-clone training shortcut. Test command:

```bash
python -m pytest -q tests/test_step6_supervision.py tests/test_recovery_supervision.py tests/test_step6_rescue.py tests/test_reference_reset.py
ruff check docs/experiments/027_step6_supervision tests/test_step6_supervision.py
```

## Interpretation and next boundary

The localized supervision change repairs both known early failures while
retaining autonomous recovery and all declared three-scene checks through
TD 10000. This supports exporting the final policy as a known-scene repair.
It does not establish success for every untested intermediate model: complete
episodes were run at 105 checkpoints, not after all 8000 updates. It also does
not establish unseen-scene performance, multiple-initialization robustness or
learning from scratch. The schedule deliberately includes previously problematic
models and is not an unbiased deployment-success estimate. The next useful
step is to freeze this candidate and evaluate a disjoint scene set before any
further tuning, keeping that evaluation separate from these training diagnostics.
