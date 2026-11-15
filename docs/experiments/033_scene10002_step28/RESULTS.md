# Attempt 033: the additional step-28 correction restores assisted success

## Result

With steps 23 and 27 fixed to north_large and north_small respectively, replacing
only step 28 with north_large restores scene-10002 success for the frozen final
attempt-030 policy. Every action after step 28 is autonomous. The control exactly
reproduces attempt 032's damage failure; independent reloads reproduce both arms.

| Arm | Fixed overrides | Success | End condition | Steps | Return | Peak normalized deformation | Blocked / stationary steps |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Control | 23: north_large; 27: north_small | No | Damage | 58 | -62.610496 | 1.000319 | 0 / 0 |
| Treatment | Same, plus 28: north_large | Yes | Target reached | 42 | 107.957262 | 0.000000 | 0 / 0 |

The treatment ends 0.015022681 from the target, with travel distance 1.644998513.
It receives exactly three overrides, compared with two in the control. Only the
third override differs between arms. The weights have not learned these choices.

## Action-level evidence

All transitions through step 27, including the two common corrections, are
identical between arms. The step-28 input, position and eight Q values also match.
At this state the policy prefers east_large over the demonstrated north_large:

| State | Greedy action | Greedy Q | Reference action | Reference Q | Greedy minus reference |
| --- | --- | ---: | --- | ---: | ---: |
| Common step 28 | east_large (6) | 87.3921737671 | north_large (4) | 86.9300079346 | +0.4621658325 |
| Treatment step 33 | south_small (1) | 90.5045013428 | east_small (2) | 90.4862289429 | +0.0182723999 |

The step-28 correction returns the trajectory to the demonstrated route. Actions
29--32 are autonomous and agree with the reference. At step 33 the model chooses
south_small rather than east_small at an identical reference input, but the
resulting autonomous trajectory still succeeds at step 42. No correction is made
at step 33 or later. This observed disagreement does not prevent success; action
agreement alone is therefore not an adequate substitute for complete episodes.

The control's first contact occurs at step 28, followed by repeated east_large
actions and damage at step 58, as recorded in attempt 032. The added correction
removes that failure in this particular conditional comparison. It does not
establish which training loss produced the Q ranking.

## Verification and immutable model

The [protocol](../033_scene10002_step28.md) and code were committed and pushed as
`b72c9ae` before simulator execution. All 22 new tests passed on their first run,
and Ruff passed on both Python files. Tests exercise both fixed interventions,
the additional override, common-prefix integrity, greedy actions, source hashes
and the optimizer guard, without simulator episodes or optimizer updates.

The control reproduces the attempt-032 trace byte-for-byte (SHA-256
`a31adeba588d0f08bd7559ca0c4789ff1dccfcc067e32d06098b2b03e8647c22`).
Both arms are independently reloaded from the complete snapshot and replayed
through training.evaluate. Every transition, action, Q vector, encoded
input/successor hash, position, reward, end flag and info dictionary matches the
primary loop. Final metrics, initial geometry and reset certification also match.

All four episodes preserve online/target parameters, Adam moments and clock
30000, TD clock 10000, configuration, agent RNG and Torch RNG exactly. Gradients
remain disabled and optimizer stepping raises an error. The encoder, network,
environment, learning rate and historical loss settings are unchanged.

- Inference model SHA-256:
  `e236374791ce5b7bb59c00d6e99c3ab557821fc06ccddbcbed6686b638b67868`.
- Full snapshot SHA-256:
  `1a63fba8f5c12b8cf13129b6dc290264c709d639e9a2b435d502d87d31ae7394`.
  Reused from attempt 031 inputs; no new policy is trained or exported.

## Execution record and cost

The single run completed without errors or restarts from 2026-11-15
19:01:18.584 UTC to 19:02:55.725 UTC (2026-11-16 03:01--03:02 Asia/Shanghai),
approximately 97 seconds. All outcomes are retained; no additional correction
search, training sample collection or parameter tuning occurred.

| Work | Episodes | Executed actions | Reset-certification actions | Fixed overrides (23/27) | Additional overrides (28) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary control/treatment | 2 | 100 | 82 | 4 | 1 |
| Independent verification | 2 | 100 | 82 | 4 | 1 |
| Total | 4 | 200 | 164 | 8 | 2 |

There are 364 total simulator actions including certification, ten externally
overridden episode actions and zero optimizer updates. Other episode actions are
chosen by the frozen policy. Previous demonstration collection and reference
validation costs belong to earlier attempts. No unseen-scene evaluation occurs.

## Evidence and limits

- [Summary and metrics](evidence/summary.json)
- [Pair audit and later reference disagreement](evidence/pair_audit.json)
- [Control trace](evidence/baseline.jsonl), [treatment trace](evidence/corrected.jsonl)
- [Independent control verification](evidence/verification_baseline.json),
  [independent treatment verification](evidence/verification_corrected.json)
- [Runtime/source provenance](evidence/provenance.json),
  [raw archive manifest](evidence/archive_manifest.json)
- [Artifact verification](artifact_verification.json): 11 raw evidence files
  (163599 bytes) match the run; 141 source/input hashes are unchanged and 14 raw
  artifacts match staged Git blobs byte-for-byte.

The result establishes step-28 intervention sufficiency conditional on the fixed
step-23 and step-27 corrections for one frozen model and known scene. It does not
show that all three corrections are individually necessary, that step 28 alone
rescues the unassisted policy, or that autonomous performance/generalization has
improved. The successful reload is a reproducibility check, not a second trained
model. Learning the required actions remains a separate, unperformed task.
