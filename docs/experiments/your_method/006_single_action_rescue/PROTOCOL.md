# Attempt 006: four frozen single-action interventions

Branch: `experiment/your-method`. Parent result: attempt 005, `0ad8396`.
Predeclare and push this protocol before any physical rollout.

## Question

Can exactly one prescribed action replacement restore success on each failed
development scene? No training, parameter tuning or additional correction is
allowed during this attempt, regardless of the outcome.

| Scene | One-based step | Replacement | Action ID |
| --- | ---: | --- | ---: |
| 290012 | 19 | east_large | 6 |
| 290021 | 27 | north_small | 0 |
| 290017 | 5 | south_large | 5 |
| 290022 | 9 | north_small | 0 |

Each corrected episode starts from reset. There are no inherited corrections
from earlier attempts. In particular, 290021 step 17 remains autonomous.

## Frozen conditions

Use `models/your_method_004_targeted.pt` (SHA256
`ea42fd1a2d23317c9e15a960c58d90970a38edbcfb7724308a845ead92f6119f`)
and the matching full snapshot
`docs/experiments/your_method/004_verified_supervision/resume/targeted_state_14000.pt`
(SHA256 `ed32355abf23c463a523df6bf00d6a9708b727bcd15fc04f357eaadd6f68fd2c`).
Restore online/target weights, Adam moments, clocks, configuration and agent RNG
for every episode. TD clock stays 14000; Adam clock stays 34000. Disable gradients
and reject optimizer steps. Use CPU, one Torch thread, zero exploration, the
existing environment and seeded reset, and no scene cache. The task budget
remains 200; intrinsic budget exhaustion sets terminated, not truncated.

## Execution and checks

Run eight complete episodes in the table's order: an autonomous baseline followed
by a corrected episode for each seed. Use the normal evaluator and instrument
its action/transition calls. Apart from the one declared replacement, all actions
must equal the frozen network's greedy choice. Reset certification is unchanged
and not counted as a policy intervention.

- Require every baseline transition, raw observation, Q vector, RNG state and
  metric to match the archived attempt-005 baseline exactly.
- Require identical seeded reset records and full trajectory prefixes for each
  pair; at the intervention, require identical raw/encoded input, position, RNG,
  Q values and unmodified policy decision.
- Audit every action and require exactly zero baseline overrides and one corrected
  override at the declared step. The replacement must actually change the action.
- Check the full model/Adam/clock/RNG snapshot on episode close and all pinned
  source/model/evidence hashes at run completion. Save complete traces, metrics,
  failure categories, costs, timings, provenance and SHA256 manifest.
- Preserve partial evidence and record any failure; never silently restart an
  experiment. Unit checks use synthetic traces and do not run physical episodes.

Success means the environment reports success during the complete corrected
episode. No additional adjustment is made if a scene remains unsuccessful.
A rescue is evidence that this intervention suffices for this frozen endpoint
and seed; it is not autonomous recovery, a training gain or a generalization claim.
Independent repeated treatment episodes are not scheduled: fresh paired controls,
exact historical baseline replay and per-action audits provide the planned checks
without doubling the simulation workload again.

## Reproduction

From the repository root, run the project interpreter with
`docs/experiments/your_method/006_single_action_rescue/run_rescue.py`.
The evidence directory must not already exist. To repeat later, use a separate
checkout/output location and retain the original evidence.
