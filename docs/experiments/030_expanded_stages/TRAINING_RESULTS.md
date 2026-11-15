# Attempt 030: completed training-stage results

These are the predeclared training-scene checks, recorded after all 50000 main
updates and before completion of the fresh holdout comparison. They do not
establish generalization. See [protocol](../030_expanded_stages.md) and
[raw stage summary](stage_summary.json). The final holdout and artifact audit
will be reported separately in RESULTS.md.

## Stage endpoints

| Endpoint | Original scenes | Added training scenes | All 33 scenes | Demonstrated actions matched | Actual-return MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attempt-029 expanded source | 1/3 | 2/30 | 3/33 | 995/1440 | 4.2916 |
| Supervised, 20000 updates | 0/3 | 4/30 | 4/33 | 1170/1440 | 6.2497 |
| Calibration, 20000 updates | 1/3 | 8/30 | 9/33 | 1185/1440 | 0.7745 |
| TD, 10000 updates | 2/3 | 5/30 | 7/33 | 1227/1440 | 2.0482 |

The source succeeds on original scene 10001; the calibrated model succeeds on
10000. The final TD model succeeds on 10000 in 38 steps and 10001 in 47 steps,
without action overrides or blocked/stationary steps. Scene 10002 fails from
damage at step 58, with peak deformation 1.00049285 and one blocked action.
All three original scenes have therefore **not** been restored.

At TD checkpoints 0, 1, 10, 100, 500, 1000, 2000, 4000 and 6000, only 10000
succeeds. Both 10000 and 10001 succeed at TD 8000 and 10000. This checks 11
models, not every intermediate policy. It supports retention of 10000 at the
sampled checkpoints, without establishing uninterrupted retention.

Improved demonstration agreement does not imply successful closed-loop behavior.
Among all 33 training scenes, the supervised endpoint has 12 damage failures
and 17 budget failures; calibration has 10 damage and 14 budget failures; final
TD has 17 damage and 9 budget failures. Return MAE measures deviation from the
fixed demonstrated returns, not error against the unknown optimal Q function.

## What TD retained and lost

From the calibrated endpoint to final TD, five successful scenes remain
successful: 10000, 290003, 290008, 290011 and 290022. Four successes are lost:
290006, 290013, 290023 and 290026. Two previously failing scenes succeed:
10001 and 290017. Thus the total changes from 9/33 to 7/33 even while original
scene success improves from 1/3 to 2/3.

These comparisons describe successive endpoints of the specified procedure.
They are not matched-compute component ablations and do not identify which
gradient or individual parameter causes a failure.

## Implementation and state verification

Architecture, encoder, environment, 1440 demonstration rows, learning rate
0.0001 and the existing focused/recovery supervision are unchanged. Complete
episode returns are independently checked against float64 discounted sums;
the maximum float32 discrepancy is 0.0000609532. No newly collected or online
training transitions are used.

The independently written update oracle reproduces the first and last ten
updates of each phase exactly, including full model/target/Adam/RNG snapshots:
60 disposable verification updates in addition to 50000 main updates.
Supervised and calibration phases have zero target synchronizations; TD has 50.
All Adam tensors and settings are identical across calibration-to-TD transition:

`4dd8e318100f6b6841a68c4fd03db164937530ea422596214b702431b496d00d`.

The Adam clock is 20000 at the transition and 30000 at final TD. The target-sync
phase clock alone resets at the boundary. Both inference models are frozen
before any fresh holdout episode, with the final TD model SHA-256:

`e236374791ce5b7bb59c00d6e99c3ab557821fc06ccddbcbed6686b638b67868`.

The 56 pre-execution tests passed; the initial independent-oracle construction
order issue and its exact-equality fix are preserved in
[preexecution_validation.json](preexecution_validation.json).
