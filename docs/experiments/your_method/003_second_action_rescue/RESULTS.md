# Own-route attempt 003: both second corrections restore conditional success

## Main result

With the first correction retained in both arms, adding the specified second
correction restores success on both frozen scenes. The one-correction controls
exactly reproduce the attempt-002 timeouts. No model weights, optimizer states,
training settings or environment conditions were changed.

| Scene | Fixed correction | Additional treatment correction | Control outcome | Treatment outcome |
| --- | --- | --- | --- | --- |
| 290012 | Step 19: east_large | Step 26: north_large | 200-step budget failure | 62-step success |
| 290021 | Step 12: east_large | Step 17: east_small | 200-step budget failure | 46-step success |

| Metric | 290012 control | 290012 treatment | 290021 control | 290021 treatment |
| --- | ---: | ---: | ---: | ---: |
| Return | -24.294699 | 107.909262 | -48.759516 | 93.784023 |
| Blocked steps | 163 | 0 | 172 | 0 |
| Stationary steps | 162 | 0 | 170 | 0 |
| Peak normalized deformation | 0.445839 | 0 | 0.360530 | 0.832282 |
| Travel distance | 1.248496 | 2.214998 | 1.370792 | 1.916249 |
| Final target distance | 0.340763 | 0.005872 | 0.208419 | 0.048068 |

Neither treatment incurs damage. Scene 290021 succeeds with nonzero deformation,
which increases relative to its stalled control. The intervention therefore does
not improve every individual physical metric, despite restoring task success.

These paired results establish sufficiency of the additional action **conditional
on the fixed first correction**, for the two specified seeds and frozen model.
They do not show that either second action alone is sufficient, that these are
the only successful action sequences, or that the policy has learned recovery.
There is no new autonomous-success or generalization result in this attempt.

## Exact comparison and verification

The [protocol](PROTOCOL.md), implementation and tests were committed and pushed
as `cd5f9be` before execution. One physical run completed from 2026-11-16
09:58:17.423 to 09:59:26.722 UTC, approximately 69.3 seconds. There were no
simulation retries, extra corrections, parameter searches or training updates.

- For both controls, every historical transition, metric, initial geometry,
  reset record and certification record matches attempt 002 exactly.
- Paired trajectories match for the first 25 steps of 290012 and 16 steps of
  290021, including the common first correction. At the new intervention input,
  all Q values and greedy decisions match exactly between arms.
- At those inputs, the unchanged model favors north_small over north_large by
  0.193405 on 290012, and south_large over east_small by 1.168007 on 290021.
- Each control has exactly one forced action and each treatment exactly two.
  All other actions remain greedy choices at the resulting visited states.
- Four independently reloaded normal-evaluator episodes reproduce complete
  transitions, certification and endpoints exactly.
- Full online/target weights, Adam moments, RNG state and configuration remain
  unchanged. TD clock is 12000 and Adam clock is 32000. Gradients are disabled,
  optimizer steps are blocked, and all pinned source/input hashes are unchanged.

| Work | Count |
| --- | ---: |
| Primary / independent verification episodes | 4 / 4 |
| Primary / verification forced actions | 6 / 6 |
| Executed episode actions, including verification | 1016 |
| Reset-certification simulation actions | 408 |
| Total simulator actions | 1424 |
| Certification calls | 8 |
| Optimizer updates / new training samples | 0 / 0 |

All eight audit tests passed on their first invocation; Ruff and full-state
preflight checks passed. See [preparation record](PREPARATION.md). The separate
[offline reviewer](review_evidence.py) verifies archived hashes, intervention
locations, greedy actions, sequential reward sums and terminal metrics without
additional simulation.

## Implication for the next learning experiment

Together with attempt 002, the diagnostics identify a sufficient intervention
sequence on each of the three regressed scenes: one correction on 290007 and
two on each of 290012/290021. A subsequent learning experiment could reinforce
these verified state/action labels while preserving the full training snapshot
and checking all 33 training scenes for new regressions. It must compare against
matched unchanged continuation and evaluate without forced actions. No such
training or additional action intervention was performed here.

## Evidence and reproduction

- [Primary results](evidence/episodes.json), [summary](evidence/summary.json),
  [pair checks](evidence/pair_checks.json), [independent replays](evidence/replays.json).
- [Provenance](evidence/provenance.json), [run status](evidence/status.json),
  [hash manifest](evidence/manifest.json), [offline verification](review.json).
- Model and snapshot hashes are fixed in the protocol. The same targeted model
  remains the current candidate; this attempt produces no new trained model.

The runner refuses to overwrite evidence. For a fresh reproduction, use the
pre-execution commit `cd5f9be` in a separate checkout and run the protocol command.
