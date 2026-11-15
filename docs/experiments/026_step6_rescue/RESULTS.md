# Attempt 026: one step-6 correction rescues both early failures

Completed on 15 November 2026. The [protocol](../026_step6_rescue.md), runner
and tests were committed before execution in `f687ba9`. Both frozen
attempt-025 failures recover after replacing only scene 10001's one-based
step 6 with `south_small` (action 1). No training update is performed.

## Paired outcomes

| Frozen model | Unassisted outcome | Step-6 correction | Unassisted return | Corrected return |
| --- | --- | --- | ---: | ---: |
| TD 2004 | Budget exhausted, 200 steps | Success, 41 steps | -104.715458756 | 107.585050000 |
| TD 2009 | Budget exhausted, 200 steps | Success, 41 steps | -104.696677315 | 107.585050000 |

Both unassisted controls reproduce their archived outcomes, action sequences
and complete paths exactly. They also match the independently reloaded failure
records from attempt 025. Within each pair, the first five complete transitions
and step-6 input, Q values and proposed action are identical. The step-6 input
is the visited reference state, and its Q values exactly match the historical
dense reference probe.

| Step-6 quantity | TD 2004 | TD 2009 |
| --- | ---: | ---: |
| Q(south_small) | 75.370048523 | 75.346298218 |
| Q(east_small) | 75.392829895 | 75.353431702 |
| Q(south_small) - Q(east_small) | -0.022781372 | -0.007133484 |
| Policy action | east_small (2) | east_small (2) |
| Single corrected action | south_small (1) | south_small (1) |

After the correction, every later action is selected by the unchanged neural
policy. The complete corrected action sequence matches the reference trajectory
for both models, with no further action divergence. No correction is made at
step 21, 22, 25, or any other step. In particular, the off-reference step-25
recovery intervention from attempt 024 is not used here.

| Episode metric, identical for both models | Unassisted | Corrected |
| --- | ---: | ---: |
| Steps | 200 | 41 |
| Collision-blocked steps | 171 | 0 |
| Stationary steps | 170 | 0 |
| Peak normalized deformation | 0.565228505 | 0 |
| Damage termination | No | No |
| Success | No | Yes |

## What this establishes

A single externally imposed `south_small` at step 6 is sufficient to rescue
these two frozen failures. On the corrected trajectory, both policies can make
all remaining decisions successfully. This supports targeting the early
step-6 action ranking when investigating the residual instability of attempt
025; fixing the later recovery state did not eliminate this upstream error.

The intervention does not improve the learned models. Their original greedy
policies still fail, and attempt 025's unassisted 67/69 all-scene checkpoint
result remains unchanged. This is not proof that step 6 is the unique possible
intervention, an explanation of the training-gradient cause, a demonstration
of uninterrupted stability, or a held-out generalization result. No new
training remedy or evaluation of the other two scenes is performed here.

## Verification and actual work

Each arm restores the same complete frozen snapshot and resets seed 10001.
The environment, reward, 200-step budget, encoder, online/target parameters,
Adam state, counters and saved agent/Torch RNG state remain unchanged. Full
saved training states are compared before and after every episode. All source
models, input evidence and bound dependencies pass their recorded SHA-256
checks before and after evaluation.

Four independent reload episodes use the original episode helper and a
separately written policy adapter. Every decision observation, Q vector,
proposed/executed action and full recorded transition matches the main loop,
including reward, motion, blocking, stationary and terminal flags. Full paths,
episode metrics and reset-certification records also match. The step observer
is installed after reset so copied-environment certification is unaffected.
A final audit confirms exactly one override in each corrected episode and
no override in either control.

| Formal work | Main pairs | Independent reloads | Total |
| --- | ---: | ---: | ---: |
| Episodes / resets | 4 | 4 | 8 |
| Executed policy actions, including overrides | 482 | 482 | 964 |
| Copied-simulator certification actions | 164 | 164 | 328 |
| Total simulator actions | 646 | 646 | 1292 |
| Forced actions, included above | 2 | 2 | 4 |
| Optimizer updates | 0 | 0 | 0 |

All 482 main transitions are reproduced by the independent loop. The formal
run completed once without failure, from 06:39:54.200212 to 06:43:00.976317 UTC.
Forty-nine tests passed in 4.84 seconds and Ruff passed. Before the formal run,
two synthetic observer tests initially failed because the fixture resolved
`env.step` before calling `agent.act` within one expression. The fixture was
corrected to use the production helper's separate action and step statements;
the complete suite then passed. This preflight correction and the initial
47-pass/2-failure result are retained in validation evidence. Unit-test work
is excluded from the formal simulator counts.

## Reproduction and evidence

From the repository root, with verified attempt-025 artifacts available:

```powershell
.venv/Scripts/python.exe -u docs/experiments/026_step6_rescue/run_step6_rescue.py
```

The runner refuses an existing output or archive. Local raw output is under
`results/training/step6_rescue_20261115`. It reuses full snapshots
`results/training/recovery_supervision_20261115/models/training_state_2004.pt`
and `training_state_2009.pt`; their bytes are not modified or duplicated into
new trained models.

The [archive manifest](evidence/archive_manifest.json) binds 12 JSON evidence
files (670,778 bytes), including the [paired outcomes](evidence/pairs.json),
complete main traces, [independent transition verification](evidence/verification.json),
[summary and work counts](evidence/summary.json), [post-run audit](evidence/analysis.json),
[validation](evidence/validation.json), reset certification and source provenance.
The post-run audit and final validation metadata were added after successful
execution; all other runner outputs remain byte-for-byte unchanged. The
pre-execution protocol and code remain bound to their original hashes.
All code and comments are English. No report or production default changed.
