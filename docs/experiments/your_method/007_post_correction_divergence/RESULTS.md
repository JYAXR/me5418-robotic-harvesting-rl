# Attempt 007: the next divergence is immediately at step 6

With seed 290017 step 5 fixed to south_large, the frozen targeted model next
diverges from **both successful comparison policies at step 6**. It chooses
south_small; both successful policies choose south_large. The complete corrected
episode still fails at the 200-step budget and exactly reproduces attempt 006.
No additional correction or training was performed.

## Same-state decision comparison

Steps are one-based. The first five executed actions and resulting transitions
match each successful policy exactly, including the externally fixed step 5.
At step 6, raw observation, encoded observation, position and pre-action RNG
match. Comparisons stop at this fork; later unequal states are not aligned.

| Frozen policy | Step-6 choice | Q(south_small) | Q(south_large) | Q(south_small) - Q(south_large) |
| --- | --- | ---: | ---: | ---: |
| Current targeted endpoint, step 5 corrected | south_small | 71.156639 | 71.102066 | +0.054573 |
| Pre-training source (attempt-034 targeted) | south_large | 72.059288 | 72.144928 | -0.085640 |
| Matched continuation control (attempt-004 control) | south_large | 71.151375 | 71.237206 | -0.085831 |

The current model's south_small preference is also its top-two action gap:
south_large is the runner-up. These are within-model Q differences, not
probabilities or measured return differences. Absolute Q levels across models
are not directly used to explain success. All eight action values for each
comparison are retained in [diagnostics](evidence/diagnostics.json).

The shared step-6 encoded input SHA256 is
`720120f526104a9dbe6b4be51f59330061eaec8f587355a13867f48811d5c51e`.
There is no earlier executed-action divergence after imposing the step-5 action.
The earlier unmodified policy disagreement at step 5 has not been learned away;
it is still corrected externally in this experiment.

## Full episode context

| Policy/trajectory | Outcome | Steps | Return | Blocked steps | Stationary steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Current endpoint with step 5 corrected | Budget failure | 200 | -6.644137 | 174 | 172 |
| Successful pre-training source, archived | Success | 35 | 103.703192 | 0 | 0 |
| Successful matched control, archived | Success | 41 | 107.817614 | 0 | 0 |

The current run finishes without damage, at target distance 0.335922 and peak
normalized deformation 0.104952. Budget exhaustion uses the existing intrinsic
termination rule (`terminated=True`, `truncated=False`). The comparison policies'
complete outcomes are historical evidence; their simulations were not rerun.

## Interpretation and next diagnostic

Correcting step 5 brings the trajectory onto the successful policies' common
prefix, but the current model immediately favors the smaller southward movement
again. This identifies a second adjacent action-ranking difference. It does not
prove that step 6 is the sole remaining cause or that correcting it is sufficient.

The next controlled intervention would keep the model frozen, retain step 5
south_large, and compare that one-correction control against a treatment that
also forces step 6 to south_large. Every later action should remain autonomous.
That intervention has **not** been run here. Training parameters and supervised
samples remain unchanged, and no generalization claim is made.

## Execution and verification

The protocol and runner were pushed as `7de33b9` before execution. Thirteen
synthetic checks passed and Ruff passed; see [preparation](PREPARATION.md).
There was one fresh, uncached episode with no restart or execution failure.

Execution ran from 2026-11-16 18:14:19 to 18:15:11 UTC
(2026-11-17 02:14:19 to 02:15:11 Asia/Shanghai). Total elapsed time was 52.535
seconds; the evaluator episode took 45.472 seconds. Costs were 200 episode
actions plus 42 scene-certification actions in one certification call. There
was exactly one override (step 5), zero additional overrides, zero optimizer
updates and zero new training samples.

The full attempt-004 targeted snapshot is unchanged: TD clock 14000, Adam clock
34000, online/target weights, optimizer moments, configuration and agent RNG.
All pinned input hashes are unchanged. Fresh raw observations, Q vectors,
actions, rewards, transitions, RNG, reset state and final metrics exactly match
the archived attempt-006 corrected episode. The original baseline-to-corrected
pair audit confirms the sole override is step 5. Comparison traces are checked
against their compressed and raw hashes before use.

Evidence: [protocol and endpoint hashes](PROTOCOL.md),
[provenance](evidence/provenance.json), [episode](evidence/episode.json),
[fixed-intervention audit](evidence/intervention_check.json),
[diagnostics](evidence/diagnostics.json), [summary](evidence/summary.json),
[manifest](evidence/manifest.json), [artifact verification](artifact_verification.json).
