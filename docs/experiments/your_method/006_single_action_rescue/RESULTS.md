# Attempt 006: three of four single-action interventions restore success

Exactly one action replacement restores success on 290012, 290021 and 290022.
The prescribed correction on 290017 is insufficient: it still exhausts the
200-action budget. All four unmodified controls reproduce the prior failures
exactly. No model, training parameter or optimizer state was changed.

## Complete paired outcomes

Steps are one-based. Every corrected run begins from reset, with no inherited
corrections. All actions except the indicated replacement remain autonomous.

| Scene | Single replacement | Baseline outcome | Corrected outcome | Baseline return | Corrected return |
| --- | --- | --- | --- | ---: | ---: |
| 290012 | Step 19: east_large | 200-step budget failure | **Success, 59 steps** | -24.194670 | 90.980978 |
| 290021 | Step 27: north_small | 200-step budget failure | **Success, 39 steps** | -50.751326 | 100.372868 |
| 290017 | Step 5: south_large | 200-step budget failure | **Budget failure, 200 steps** | -6.640137 | -6.644137 |
| 290022 | Step 9: north_small | 200-step budget failure | **Success, 63 steps** | -8.511034 | 106.676118 |

No episode ends in damage. Budget exhaustion is intrinsic finite-horizon
termination (`terminated=True`, `truncated=False`), not external truncation.

| Scene | Corrected blocked steps | Corrected stationary steps | Final target distance | Peak normalized deformation |
| --- | ---: | ---: | ---: | ---: |
| 290012 | 0 | 0 | 0.027814 | 0.632018 |
| 290021 | 0 | 0 | 0.005021 | 0.517735 |
| 290017 | 174 | 172 | 0.335922 | 0.104952 |
| 290022 | 1 | 0 | 0.048499 | 0.101358 |

## Decision values at the intervention

The baseline and corrected run have identical observations, position, RNG,
Q values and policy decision immediately before replacement. The gap is
Q(original policy action) minus Q(replacement) under the same frozen model.

| Scene | Original policy action | Replacement | Q(original) | Q(replacement) | Gap |
| --- | --- | --- | ---: | ---: | ---: |
| 290012 | east_small | east_large | 67.640778 | 67.460747 | +0.180031 |
| 290021 | north_large | north_small | 89.665619 | 89.600616 | +0.065002 |
| 290017 | south_small | south_large | 69.674011 | 69.594391 | +0.079620 |
| 290022 | north_large | north_small | 72.452255 | 72.436256 | +0.015999 |

## Interpretation

- **290012:** The step-19 replacement now suffices on the attempt-004 endpoint.
  This differs from the older attempt-034 model tested in attempt 002, which
  needed a further correction in attempt 003. No step-26 action is forced here.
- **290021:** Replacing only step 27 suffices. The first 26 actions, including
  the step-17 deviation from the demonstration, remain unchanged. The earliest
  demonstration difference was therefore not an unavoidable failure on this
  trajectory: the later intervention restores success without revising it.
- **290017:** Replacing the earliest difference from the successful source/control
  does not restore success. Both current baseline and corrected run finish at
  the same target distance, with 174 blocked and 172 stationary steps. The
  corrected travel distance is slightly larger (0.861791 versus 0.841791).
  The first differing action alone does not explain the failure sufficiently
  to repair it. Later decisions have not been diagnosed in this attempt.
- **290022:** The step-9 north_small replacement suffices, despite the model's
  original north_large preference being only 0.015999 higher in Q value.

The successful interventions establish sufficiency for these frozen seeded
episodes. They do not show that the selected actions are uniquely optimal,
that other corrections cannot work, or that the model has learned the repairs.
The 3/4 assisted success fraction is a result on four selected development
failures, not a held-out success rate or a new overall 33-scene score.

## Execution, integrity and costs

Protocol and runner were pushed as `6491304` before execution. There was one
physical run with eight complete episodes, no restart and no failed execution.
The unsuccessful 290017 treatment is retained as a measured experimental result.
Seven synthetic tests and Ruff passed before execution; see
[preparation](PREPARATION.md).

Execution ran from 2026-11-16 18:02:22 to 18:07:23 UTC
(2026-11-17 02:02:22 to 02:07:23 Asia/Shanghai), taking 300.354 seconds overall.
Costs were 1161 episode actions and 388 scene-certification actions across eight
certification calls: 1549 simulator actions in total. There were exactly four
action replacements, zero optimizer updates and zero new training samples.
No cache or repeated treatment episodes were used.

Every episode restored the same full attempt-004 targeted snapshot. The model
and snapshot hashes are pinned in the [protocol](PROTOCOL.md) and
[provenance](evidence/provenance.json). TD clock 14000, Adam clock 34000, optimizer
moments, online/target weights and agent RNG remain unchanged on close.
All pinned input hashes are unchanged at run completion. Each fresh baseline
exactly matches its attempt-005 historical trace and metrics; every pair has
identical seeded reset records and pre-intervention trajectories. Per-action
audits verify zero baseline replacements and exactly one treatment replacement.

## Next step

A separate frozen diagnosis can hold 290017 step 5 at south_large and locate
its next disagreement from a successful policy on the common state prefix,
before proposing another conditional intervention. The three successful
state/action corrections are candidates for a future supervised experiment,
which would need fresh autonomous rollouts and all-33-scene regression checks.
Neither additional intervention nor training was performed here.

Evidence: [paired checks and full metrics](evidence/pair_checks.json),
[episode records and timing](evidence/episodes.json), [summary](evidence/summary.json),
[manifest](evidence/manifest.json), [artifact verification](artifact_verification.json).
