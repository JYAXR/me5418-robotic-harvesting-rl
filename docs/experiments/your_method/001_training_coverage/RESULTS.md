# Own-route attempt 001: broad training competence remains limited

## Outcome

The frozen targeted model succeeds on 10/33 training scenes (30.3%), compared
with 9/33 (27.3%) for the matched attempt-034 control. Both succeed on only 7/30
added training scenes. Original-scene recovery does not establish broad training
competence, and this audit does not measure unseen-scene generalization.

| Metric | Control | Targeted |
| --- | ---: | ---: |
| Original training successes | 2/3 | 3/3 |
| Added training successes | 7/30 | 7/30 |
| All training successes | 9/33 | 10/33 |
| Damage failures | 17 | 16 |
| Budget failures | 7 | 7 |
| Mean return across all 33 scenes | -5.070996 | -6.328920 |
| Mean steps among successful episodes | 47 | 48 |

Successful-episode step means use different subsets and are not a matched
efficiency comparison. Targeted has one more success but lower mean return;
the evidence does not justify describing it as uniformly better.

Targeted gains successes on 10002, 290011, 290017 and 290026, while losing
successes on 290007, 290012 and 290021. On the added scenes, three gains and
three losses cancel. Six of targeted's seven budget failures end in blocked
stationary stalls; the remaining timeout is classified as occurring while
moving. These are descriptive signatures, not identified causal mechanisms.

## Execution and verification

The [protocol](PROTOCOL.md) and [runner](run_audit.py) were committed and pushed
as `4e2dcef` before execution on `experiment/your-method`. The single evaluation
run completed from 2026-11-16 07:52:49.935 to 07:59:57.687 UTC, approximately
7 minutes 8 seconds. No simulation retry, training update, action override,
parameter change or fresh test-scene evaluation occurred.

All 33 paired reset records match, including geometry and RNG state. All six
original-scene metrics exactly reproduce attempt 034. All 66 episodes preserve
learner/optimizer/RNG state and choose greedy Q argmax actions. Source and model
hashes remain unchanged. Complete raw-observation/Q/action/reward/info traces
are retained with lossless compression and hashes.

| Work | Count |
| --- | ---: |
| Autonomous evaluation episodes | 66 |
| Executed episode actions | 5438 |
| Actual reset-certification actions | 1440 |
| Total simulator actions | 6878 |
| Actual certification calls / cache hits | 33 / 33 |
| Optimizer updates / forced actions | 0 / 0 |

The [offline review](review_evidence.py) verifies 71 manifest files and all 66
decompressed trajectories, independently checking rewards, action selection,
blocking/stationary counts and terminal outcomes. It uses no simulator actions.
Ruff and runner import/path/seed checks pass. Core training code is unchanged.

The first offline-review invocation failed its return equality assertion because
Python's built-in floating-point sum differed from the evaluator's sequential
accumulation. An audit of all 66 traces found 22 such differences, with maximum
absolute difference 8.53e-14. Matching the evaluator's accumulation order restores
exact equality for every episode; the second review invocation passes. This was
a verifier issue, not a rerun or alteration of experimental evidence.

## Post-hoc reference diagnostics

Every episode, including successful ones, eventually departs from the recorded
demonstration at an identical encoded input. Therefore reference disagreement
alone must not be labeled a policy error or a failure cause.

For the three successes lost relative to control, targeted's first reference
disagreements are:

| Scene | Step | Policy action | Reference action | Reference Q minus best competing Q |
| --- | ---: | --- | --- | ---: |
| 290007 | 8 | north_small | east_large | -0.089867 |
| 290012 | 19 | east_small | east_large | -0.593163 |
| 290021 | 12 | north_small | east_large | -0.095543 |

These positions are candidates for diagnostic interventions, not validated
recovery labels. All reference comparisons and Q values are in [review.json](review.json).

## Next experiment recommendation, not yet executed

1. On the three lost-success scenes, use matched frozen rollouts to test whether
   correcting the recorded first disagreement restores success. Retain both
   successful and unsuccessful interventions. If it does not, examine later
   visited states rather than assuming the first disagreement caused failure.
2. Build verified recovery supervision across multiple failing training scenes,
   including off-demonstration states. Check proposed expert recovery paths in
   copied physical states before accepting labels, and record planning/simulation
   cost and rejected recovery attempts.
3. Compare unchanged continuation against added recovery supervision from the
   same complete targeted snapshot. Keep architecture, TD coefficient, learning
   rate, Adam state and update budget fixed; declare data weighting and evaluation
   checkpoints before execution. Evaluate all 33 training scenes, including
   retained successes, at fixed checkpoints rather than selecting the best one.
4. Once training competence improves, evaluate on separate development-validation
   scenes. Keep the eventual common final test with the teammate untouched.

This recommendation addresses cross-scene recovery coverage. Its benefit remains
an experimental hypothesis; no new trained model is produced by this audit.

## Evidence

- [Summary](evidence/summary.json), [all episodes](evidence/episodes.json),
  [paired outcomes](evidence/pairs.json), [status](evidence/status.json).
- [Provenance](evidence/provenance.json), [trace manifest](evidence/manifest.json),
  [independent review](review.json).
- Models remain `models/supervision_034_targeted.pt` and
  `models/supervision_034_control.pt`; their exact hashes are in the protocol.
