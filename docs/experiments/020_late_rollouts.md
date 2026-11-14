# Attempt 020: complete episodes around the late Q-ranking reversal

## Protocol declared before execution

Evaluate every attempt-019 treatment model from TD 9946 through 9966 inclusive
on known scenes 10000, 10001 and 10002 (63 autonomous complete episodes).
The existing dense log identifies negative step-21 gaps at TD 9955 and 9956;
9957 is positive. This is a targeted retrospective diagnostic, not an unbiased
estimate of the frequency of failure or a held-out generalization test.

Restore the unchanged full TD-9500 snapshot and replay 500 updates through
TD 10000 with TD coefficient 0.5, supervised coefficient 1, learning rate
0.0001, retained Adam, original targets/counters/data and no MC constraint.
Compare every replayed loss record and all seven reference Q vectors with
attempt 019, and require full endpoint equality before physical evaluation.
Retain all 21 reconstructed full snapshots locally. A separate implementation
replays the same 500 updates and compares all 21 snapshots and the endpoint.
These 1000 replay updates reconstruct existing models; they are not a new
training remedy or additional updates on the original branch.

Use the original greedy rollout helper and unchanged environment settings,
including reset certification and the 200-step budget. Observe all actual
policy decisions, full encoded-state matches, and Q values without replacing
any action. Record the first action divergence and whether the actual step-21
state equals the demonstrated state. Preserve all complete trajectories,
returns, failures, damage, blocking and recovery outcomes. Separately count
policy and reset-certification actions. No online learning or action rescue.

Independently reload TD 9955, 9956 and 9957 on all three scenes (nine additional
complete episodes), compare exact trajectories and certification checks, and
verify evaluation does not change weights, targets, Adam, counters or RNG.
A failure after a matching-prefix divergence establishes a behavioral failure
at that model; it does not isolate the action's causal effect without an
intervention. Do not claim success at untested updates.

Record source hashes, local model hashes, full observations, verification and
work accounting. Keep binary checkpoints and raw logs local; commit compact
JSON evidence and this English record to Git. Existing code, results, reports
and production defaults remain unchanged. Results pending at protocol commit.

## Completed results: 14 November 2026

Both reconstructions passed exactly: each replayed 500 historical updates,
matched every update log and seven-state Q probe, and reached the original
full TD-10000 endpoint. The independently implemented update also matched all
21 reconstructed full intermediate states. No hyperparameter or policy action
was changed. The main execution and analysis each completed once without error.

The 21 newly evaluated models produced **61/63 successful episodes** and
**19/21 checkpoints with all three scenes successful**. Scenes 10000 and 10002
succeeded at every checkpoint. Scene 10001 failed only at TD 9955 and 9956;
it succeeded throughout 9946--9954 and 9957--9966.

| TD update | Step-21 Q(north_small) - Q(best alternative) | Actual action | Scene 10001 result | Steps | Return |
| --- | ---: | --- | --- | ---: | ---: |
| 9954 | 0.12645721 | north_small | Success | 41 | 107.585050 |
| 9955 | -0.02781677 | north_large | Budget exhausted | 200 | 2.479857 |
| 9956 | -0.01574707 | north_large | Budget exhausted | 200 | 2.479857 |
| 9957 | 0.13756561 | north_small | Success | 41 | 107.585050 |
| 9958 | 0.39015961 | north_small | Success | 41 | 107.585050 |

At both failed models, the first 20 actions matched the demonstration. The
actual encoded observation at step 21 was exactly the reference input, and its
entire eight-action Q vector exactly matched the original dense log. This
rules out the interpretation that the negative gap occurred only on an
unvisited reference state. The first action divergence was `north_large`
(action 4) instead of `north_small` (action 0). Both continuations recorded
178 blocked steps, 177 stationary steps, no damage, and budget exhaustion at
200 actions, with return 2.479856778681279. The recovery model TD 9957 completed
the original 41-action route with no blocked or stationary steps and return
107.58505000025033.

Thus the late ranking reversal corresponds to a real, reproducible behavioral
failure. This experiment identifies the common-prefix decision where the
policy departs and then fails. It does not isolate that action as the sole
cause: all subsequent actions remain model-controlled, and no forced-action
rescue was performed. A separate intervention at step 21 would be needed for
that narrower causal claim. Other unsampled negative-gap updates have not been
converted into episode-level failures by assumption.

This qualifies attempt 019's result: the coefficient reduction improves its
original sampled retention count, but the apparent late success at sparse
checkpoints hides at least these two failures. The new window was selected
because a known Q reversal occurred there, so its 61/63 count is not an unbiased
success-rate estimate and should not be pooled with the earlier schedule as a
new benchmark. No unseen layouts, new initialization, or longer training were
tested.

## Verification and work accounting

Nine uninstrumented reload episodes at TD 9955, 9956 and 9957 reproduced the
complete trajectories and reset certification exactly. The observer therefore
did not change outcomes at either failed model or the adjacent recovery model.
Every evaluation preserved the entire training snapshot, including online and
target weights, Adam, counters and RNG. All 21 local model hashes and each
recorded actual action were audited. All 93 existing coefficient/replay/analysis
tests passed in 9.61 seconds; Ruff passed for the two new scripts.

| Work | Main evaluation | Independent reload | Total |
| --- | ---: | ---: | ---: |
| Full episodes / resets | 63 | 9 | 72 |
| Autonomous policy actions | 2775 | 669 | 3444 |
| Copied-simulator reset certification actions | 2457 | 351 | 2808 |
| All simulator actions | 5232 | 1020 | 6252 |

The two reconstruction passes used 1000 optimizer updates in total, solely to
recover and verify already-trained states. There were zero new remedy-training
updates, zero online learning steps and zero forced actions. Certification
remained enabled and all reset records were preserved. Unit tests are separate
from these formal work counters.

## Reproduction and evidence

Run from the repository root with the original local attempt-019 artifacts:

```powershell
.venv/Scripts/python.exe -u docs/experiments/020_late_rollouts/run_late_rollouts.py
.venv/Scripts/python.exe docs/experiments/020_late_rollouts/analyze_late_rollouts.py
```

The tools refuse to overwrite the existing run or archive. All 21 full model
snapshots remain in `results/training/late_rollouts_20261114/models`. The
[archive manifest](020_late_rollouts/evidence/archive_manifest.json) records
exact hashes and bytes for 11 JSON evidence files. These include all complete
trajectories and actual-decision Q vectors, source provenance, model hashes,
reset checks, [analysis](020_late_rollouts/evidence/analysis.json),
[verification](020_late_rollouts/evidence/verification.json), test validation
and work accounting. Model binaries and console logs remain local.
