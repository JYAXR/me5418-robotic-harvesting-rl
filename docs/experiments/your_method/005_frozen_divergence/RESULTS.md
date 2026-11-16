# Attempt 005: frozen first-divergence diagnosis

The four failures reproduce exactly. The targeted endpoint from attempt 004
times out at 200 actions on every requested scene. No learning, action override,
parameter change or new model export was performed. These are development
diagnostics, not an independent test of generalization.

## Endpoint and execution

- Protocol and implementation were committed and pushed as `2293534` before execution.
- Frozen inference model: `models/your_method_004_targeted.pt`, SHA256
  `ea42fd1a2d23317c9e15a960c58d90970a38edbcfb7724308a845ead92f6119f`.
- Full resume snapshot: attempt 004 `resume/targeted_state_14000.pt`, SHA256
  `ed32355abf23c463a523df6bf00d6a9708b727bcd15fc04f357eaadd6f68fd2c`.
- TD clock 14000 and Adam clock 34000 remained unchanged, including optimizer
  moments, target network and RNG state. CPU inference used one Torch thread.
- One run, four complete uncached episodes; no failed physical trial or restart.
  Execution ran from 2026-11-16 17:46:20 to 17:49:29 UTC
  (2026-11-17 01:46:20 to 01:49:29 Asia/Shanghai).
- Total elapsed time was 188.891 seconds: 800 episode actions plus 194 scene
  certification actions in four certification calls; zero optimizer updates.
  Historical source and control traces were reused without extra simulations.

## First difference from the demonstration

Steps are one-based. Every preceding action and encoded input matches the
stored demonstration. The gap is **Q(current choice) - Q(comparison action)**
under the current frozen model, not a probability or a measured return difference.

| Scene | Step | Current choice | Demonstration | Q(current) | Q(demonstration) | Gap |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 290012 | 19 | east_small | east_large | 67.640778 | 67.460747 | +0.180031 |
| 290021 | 17 | south_large | east_small | 83.945023 | 83.014275 | +0.930748 |
| 290017 | 5 | south_small | east_small | 69.674011 | 69.396996 | +0.277016 |
| 290022 | 14 | south_small | east_small | 75.350334 | 75.336014 | +0.014320 |

The remaining targeted failures first leave the reference at supervised rows
653 and 1057. Their preferred-action rankings have not flipped. This establishes
an unresolved reference mismatch, but does not establish that either isolated
choice causes the eventual failure.

## First difference from the pre-training source

The source is the attempt-034 targeted endpoint from which attempt 004 started.
Only common trajectory prefixes are compared: raw observation, encoded input,
position and pre-action RNG are identical at the reported fork. The last column
is the source model's preference for its own choice over the current model's
choice, evaluated at that same input.

| Scene | Step | Current choice | Source choice | Current gap | Source opposite gap | Source outcome |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 290012 | 22 | east_large | north_small | +0.065605 | +0.170105 | Timeout |
| 290021 | 12 | east_large | north_small | +0.361023 | +0.095543 | Damage |
| 290017 | 5 | south_small | south_large | +0.079620 | +0.024742 | Success |
| 290022 | 9 | north_large | north_small | +0.015999 | +0.004944 | Success |

For the two regressions, the earliest change from a successful source is a
small-versus-large movement decision. On 290017, the successful source already
differs from the demonstration at step 5 (`south_large` versus `east_small`).
On 290022, the current model first differs from the successful source at step 9,
while its first demonstration difference is only at step 14. Demonstration
agreement alone would therefore miss the earliest policy change on 290022.

## First difference from the matched continuation control

This control has the same additional training duration as the targeted endpoint,
but did not double the five selected supervision weights. The gap convention is
the same as above; the last gap is the control model's opposite preference.

| Scene | Step | Current choice | Control choice | Current gap | Control opposite gap | Control outcome |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 290012 | 54 | south_small | east_large | +0.112732 | +0.048531 | Timeout |
| 290021 | 27 | north_large | north_small | +0.065002 | +0.001762 | Success |
| 290017 | 5 | south_small | south_large | +0.079620 | +0.054779 | Success |
| 290022 | 9 | north_large | north_small | +0.015999 | +0.067436 | Success |

On 290021, the successful control shares the first 26 actions with the failing
targeted policy, including its step-17 deviation from the demonstration. Their
first decision difference is step 27. This makes step 27 a useful candidate for
a future frozen intervention, rather than treating step 17 as a proven cause.
On 290012, source, control and current policies all fail; the first difference
from either baseline is not evidence of a successful alternative.

## Complete episode outcomes

| Scene | Current return | Current outcome | Blocked steps | Stationary steps | Source steps/outcome | Control steps/outcome |
| --- | ---: | --- | ---: | ---: | --- | --- |
| 290012 | -24.194670 | 200-step timeout | 163 | 162 | 200 / timeout | 200 / timeout |
| 290021 | -50.751326 | 200-step timeout | 172 | 170 | 46 / damage | 39 / success |
| 290017 | -6.640137 | 200-step timeout | 174 | 172 | 35 / success | 41 / success |
| 290022 | -8.511034 | 200-step timeout | 181 | 179 | 63 / success | 58 / success |

All four current rollouts spend many steps blocked or stationary and exhaust
the budget without damage. Here, timeout means intrinsic task-budget termination
(`terminated=True`, `truncated=False`), consistent with the existing finite-horizon
environment. These trajectory outcomes do not prove which earlier
decision caused the blocking. In particular, the successful corrective replays
in attempts 002/003 used the older attempt-034 model; they cannot establish
rescue of this newer endpoint.

## Verification and next diagnostic candidates

Each new trajectory exactly matches its attempt-004 historical trace, including
all observations, actions, Q vectors, rewards, transition outcomes and RNG states.
All recorded metrics and seeded initial states also match. Full learner snapshots
and pinned input hashes are unchanged before/after execution. Five diagnostic
tests passed; the preparation record documents the one corrected style warning.
The post-run artifact audit verifies nine manifest files, all pinned inputs, four
trace returns and 800 greedy actions. Its initial read-only assertion incorrectly
expected truncation on budget exhaustion; after checking the environment's
finite-horizon termination rule, the assertion was corrected and the audit passed.
No simulator rerun or evidence change was needed. Details are in
[artifact verification](artifact_verification.json).

No intervention was performed here. A subsequent single-action study could test
290017 step 5 `south_large`, 290022 step 9 `north_small`, and 290021 step 27
`north_small` against frozen, unmodified controls. For 290012, step 19
`east_large` remains the earliest reference-based candidate, but its efficacy
must be checked again on this endpoint. Any later corrections would require
their own conditional comparison.

Evidence: [protocol](PROTOCOL.md), [preparation](PREPARATION.md),
[diagnostics with all eight Q values](evidence/diagnostics.json),
[episodes and timing](evidence/episodes.json), [provenance](evidence/provenance.json),
[summary](evidence/summary.json), [hash manifest](evidence/manifest.json).
