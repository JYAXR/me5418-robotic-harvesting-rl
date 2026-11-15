# Attempt 031 results: one correction does not rescue scene 10002

## Outcome

Replacing only step 23 with `north_large` does **not** restore success for the
frozen final attempt-030 TD policy. The intervention changes a 58-step damage
failure into a 200-step blocked budget failure. A second action disagreement
appears at step 27 on the corrected trajectory. No further correction or
training update was performed.

| Episode | Success | End condition | Steps | Return | Peak normalized deformation | Blocked steps | Stationary steps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen autonomous baseline | No | Damage | 58 | -62.168230 | 1.000493 | 1 | 0 |
| Only step 23 replaced | No | Budget exhausted | 200 | -41.612027 | 0.295240 | 174 | 171 |
| Complete scripted reference | Yes | Success | 41 | 107.643380 | 0.000000 | 0 | 0 |

The scripted row checks feasibility and the stored demonstration, not autonomous
policy competence. The baseline exactly reproduces all historical attempt-030
transitions and metrics. Both diagnostic episodes were then independently
reloaded through the normal evaluator and matched at every step.

## First disagreement and its counterfactual

The original greedy and successful reference trajectories have identical encoded
states and actions for steps 1--22. At the step-23 input, the end effector is at
(0.542228520, 0.422055900). The model ranks `north_small` first and `north_large`
second. There is no tie or change in the encoder.

| Visited state | Greedy action | Greedy Q | Reference action | Reference Q | Greedy minus reference |
| --- | --- | ---: | --- | ---: | ---: |
| Baseline step 23 | north_small (0) | 80.4008636475 | north_large (4) | 80.3592376709 | +0.0416259766 |
| Corrected trajectory step 27 | east_large (6) | 87.2107772827 | north_small (0) | 87.1775360107 | +0.0332412720 |

The reference-minus-best-competitor margins are therefore -0.0416259766 and
-0.0332412720. Both comparisons use single-observation forward passes at visited
states. Full eight-action Q vectors and input hashes are in the raw evidence.

At step 23, the correction moves north by 0.06 instead of 0.025. The fork input,
network values and all preceding transitions remain exactly identical between
arms. Steps 24--26 are autonomous and agree with the reference, taking the
corrected trajectory to the reference step-27 input at
(0.662228405, 0.542055786). At this point the policy chooses `east_large` instead
of `north_small`.

The baseline first contacts an obstacle and is blocked at step 25, later ending
in damage. The corrected trajectory first contacts an obstacle and is blocked
at step 27. It reaches (0.719014883, 0.542055786) and remains stationary for all
171 steps from step 30 through 200. Its final target distance is 0.212678328.
No action at step 27 or later was externally replaced.

This establishes that the first disagreement alone is insufficient to explain
or repair the complete failure. Correcting it changes the trajectory and avoids
damage in this run, but another fragile reference-state decision is encountered.
The later Q reversal is an observed candidate for follow-up; this experiment
does not establish that correcting it would be sufficient for success, nor does
it identify which training loss caused either ranking.

## Controls and verification

- The [predeclared protocol](../031_scene10002_rescue.md), code and full snapshot
  were committed and pushed as `44e667b` before new simulator rollouts.
- All 58 baseline transitions reproduce the archived input/successor hashes,
  positions, actions, eight Q values, rewards, end flags and environment info.
- The corrected arm differs externally only at step 23. Its prefix through
  step 22 and its step-23 input/Q vector match the baseline exactly.
- The positive control executes all 41 reference transitions; encoded current
  and next states, stored float32 rewards and termination flags match the
  demonstration arrays. The current float64 accumulated return is reported in
  the table; the archived training rewards are float32.
- Two fresh full-snapshot reloads use `training.evaluate`, with the same one
  override in the corrected arm. Every transition, final metric and reset
  certification matches the primary manual loop.
- All five episodes preserve the complete online/target weights, Adam moments,
  per-parameter Adam clock 30000, TD clock 10000, configuration, agent RNG and
  Torch RNG exactly. Source and inference file hashes are unchanged.
- The production simulator, encoder and agent source hashes match attempt 030.
  Networks have gradients disabled; optimizer stepping raises an exception.
  Architecture, learning rate and all historical loss settings are unchanged.

Model SHA-256:
`e236374791ce5b7bb59c00d6e99c3ab557821fc06ccddbcbed6686b638b67868`.
Portable full-snapshot SHA-256:
`1a63fba8f5c12b8cf13129b6dc290264c709d639e9a2b435d502d87d31ae7394`.
The snapshot is retained in `inputs/final_td_state.pt`; no new policy was trained
or exported.

## Execution and development record

The single simulator run completed from 2026-11-15 18:30:42.742 UTC to
18:33:49.980 UTC (2026-11-16 02:30--02:33 Asia/Shanghai), approximately 187 seconds.
There were no interrupted or restarted simulator runs and no extra action search.

| Work | Episodes | Executed actions | Reset-certification actions | Forced corrections |
| --- | ---: | ---: | ---: | ---: |
| Primary baseline and corrected pair | 2 | 258 | 82 | 1 |
| Scripted reference validation | 1 | 41 | 41 | 0 |
| Independent pair verification | 2 | 258 | 82 | 1 |
| Total | 5 | 557 | 205 | 2 |

There were 762 total simulator actions including certification, zero optimizer
updates and zero new training samples. The 41 scripted reference actions and
205 certification actions are explicitly separate from policy-controlled work.
Historical data-collection and training costs belong to earlier attempts.

The first unit-test run passed 13 tests and failed one exact optimizer-summary
norm comparison: the test used the machine's default Torch thread count, while
the source run and diagnostic runner use one thread. The test was corrected to
use the same reduction order; exact tensor/summary equality was retained rather
than relaxed. The next run passed 14 tests. Two additional intervention-audit
cases were then added; the final version passed all 16 tests. These tests did not
run simulator episodes or perform optimizer updates. Ruff passed on both new
Python files. All new code, comments and experiment records are in English.

## Evidence and limits

- [First fork and all Q values](evidence/first_fork.json)
- [Paired intervention audit and later disagreement](evidence/pair_audit.json)
- [Summary, metrics and resource counts](evidence/summary.json)
- [Baseline trace](evidence/baseline.jsonl), [corrected trace](evidence/corrected.jsonl),
  [scripted reference trace](evidence/reference.jsonl)
- [Independent baseline verification](evidence/verification_baseline.json),
  [independent corrected verification](evidence/verification_corrected.json)
- [Source and runtime provenance](evidence/provenance.json),
  [raw archive manifest](evidence/archive_manifest.json)
- [Archive and Git-index verification](artifact_verification.json): all 13 raw
  evidence files (396282 bytes) match the run byte-for-byte; 129 source/input
  hashes remain unchanged, and 16 raw artifacts match their staged Git blobs.

This is a deterministic diagnostic of one already-observed training scene.
The two corrected executions are a reproducibility check, not independent
evidence of a population success rate. No other original scene or held-out scene
was reevaluated, and no claim of learned repair or improved generalization is made.
A possible next isolated test would hold the step-23 correction fixed and change
only step 27 to `north_small`; that second intervention has not been performed.
