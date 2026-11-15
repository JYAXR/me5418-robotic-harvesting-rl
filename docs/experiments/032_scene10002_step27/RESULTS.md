# Attempt 032 results: adding the step-27 correction does not restore success

## Outcome and comparison

With step 23 fixed to `north_large` in both arms, additionally replacing step 27
with `north_small` does **not** restore scene-10002 success. The control reproduces
the previous 200-step blocked budget failure. The treatment instead fails from
damage after 58 steps, with a lower return. No further actions were corrected,
and the model and all training parameters remain unchanged.

| Frozen-policy arm | Overrides | Success | End condition | Steps | Return | Peak normalized deformation | Blocked steps | Stationary steps |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Control | Step 23: north_large | No | Budget exhausted | 200 | -41.612027 | 0.295240 | 174 | 171 |
| Treatment | Step 23: north_large; step 27: north_small | No | Damage | 58 | -62.610496 | 1.000319 | 0 | 0 |

Only the step-27 intervention differs between these arms. The control is the
attempt-031 corrected trajectory, not the unassisted policy. Its complete
200-transition trace reproduces the previous trace byte-for-byte, including Q
values, action overrides, positions, rewards, end flags and environment info.
Both new episodes were independently reloaded through the normal evaluator and
matched at every step.

## Fork and subsequent failure

The two arms share all transitions through step 26, including the common step-23
correction. They reach exactly the same encoded input, position and Q vector at
step 27. The treatment has exactly two overridden actions; all others remain
autonomous argmax decisions.

| Visited reference state | Frozen greedy action | Greedy Q | Demonstrated action | Demonstrated Q | Greedy minus demonstrated |
| --- | --- | ---: | --- | ---: | ---: |
| Step 27, common input | east_large (6) | 87.2107772827 | north_small (0) | 87.1775360107 | +0.0332412720 |
| Step 28, treatment trajectory | east_large (6) | 87.3921737671 | north_large (4) | 86.9300079346 | +0.4621658325 |

At step 27, the replacement moves the end effector from
(0.662228405, 0.542055786) to (0.662228405, 0.567055762). The executed trajectory
now matches the reference through step 27. At the resulting reference step-28
state, the policy immediately disagrees again: it turns east instead of taking
the demonstrated larger northward step. The demonstrated action is below both
`east_large` and `north_small` in this Q vector. Full eight-action values and
encoded hashes are retained in the evidence.

The treatment first contacts an obstacle at step 28, when normalized deformation
reaches 0.119987. Every action from steps 28 through 58 is `east_large`. It ends at
(0.785034895, 0.567055762), with 31 contact steps and target distance 0.164450139.
Peak normalized deformation reaches 1.000318521, exceeding the environment's
damage threshold of 1.0. There are no `collision_blocked` or stationary steps in
this trajectory. Those zero counts therefore do not imply safe passage.

In the control, contact/blocking begins at step 27 and stationary behavior lasts
from step 30 through 200. The extra correction changes this failure mode, but
does not complete the task. Its return decreases by 20.998470 in this deterministic
comparison. The step-28 disagreement is a candidate for a subsequent causal test;
no correction at step 28 or any other new location was tried here.

## Frozen-state and reproduction checks

- The [protocol](../032_scene10002_step27.md) and implementation were committed
  and pushed as `6ac5943` before simulator execution.
- Reused final attempt-030 inference policy SHA-256:
  `e236374791ce5b7bb59c00d6e99c3ab557821fc06ccddbcbed6686b638b67868`.
- Reused portable full-snapshot SHA-256:
  `1a63fba8f5c12b8cf13129b6dc290264c709d639e9a2b435d502d87d31ae7394`.
  The existing snapshot remains in `031_scene10002_rescue/inputs/final_td_state.pt`.
- All four physical episodes preserve online/target parameters, Adam moments,
  Adam per-parameter clock 30000, TD clock 10000, environment/DQN configuration,
  agent RNG and Torch RNG exactly. Gradients are disabled and optimizer stepping
  raises an error. No production training code or parameter was changed.
- The control trace SHA-256 is still
  `9f8e4bffb008be4b90f1acb1859cd51d53ca5d21c13994b2416d728acc41a5ef`,
  exactly matching attempt 031. Reset geometry, certification and metrics also
  match that source run.
- Both arms have fresh, independent full-snapshot reloads through
  `training.evaluate`. The audit compares every action, Q vector, input/successor
  hash, position, reward, end flag and info dictionary with the primary loop.
  Final metrics and reset-certification records also match exactly.
- The frozen model still selects `north_small` at step 23 and `east_large` at
  step 27. The externally corrected choices have not become learned behavior.

## Execution and validation record

The single run completed from 2026-11-15 18:42:56.185 UTC to 18:45:49.558 UTC
(2026-11-16 02:42--02:45 Asia/Shanghai), approximately 173 seconds. The runner
completed once without execution errors or restarts. There were no alternative
action searches or training updates.

| Work | Episodes | Executed actions | Reset-certification actions | Fixed step-23 overrides | Additional step-27 overrides |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary control/treatment pair | 2 | 258 | 82 | 2 | 1 |
| Independent pair verification | 2 | 258 | 82 | 2 | 1 |
| Total | 4 | 516 | 164 | 4 | 2 |

Total simulator actions: 680, including certification. Six executed actions are
overrides across the four episodes; all remaining episode actions come from the
frozen policy. There are zero new training samples and zero optimizer updates.
No complete scripted demonstration or unassisted episode was rerun; their
previous costs and successful reference validation remain in attempt 031.

All 20 new diagnostic tests passed on their first run; Ruff passed on both new
Python files. Tests cover the common correction, the added correction, changed
prefixes/Q vectors, extra interventions, greedy decisions, source integrity and
the optimizer guard. They perform no simulator episodes or optimizer updates.
New code, comments, protocols and experiment documentation are all in English.

## Evidence and scope

- [Summary and resource counts](evidence/summary.json)
- [Step-27 Q values](evidence/additional_fork.json)
- [Paired audit and step-28 disagreement](evidence/pair_audit.json)
- [Control trace](evidence/baseline.jsonl), [treatment trace](evidence/corrected.jsonl)
- [Independent control verification](evidence/verification_baseline.json),
  [independent treatment verification](evidence/verification_corrected.json)
- [Source/runtime provenance](evidence/provenance.json),
  [raw archive manifest](evidence/archive_manifest.json)
- [Artifact verification](artifact_verification.json): 11 raw evidence files
  (341756 bytes) match the run byte-for-byte; 142 source/input hashes are
  unchanged, and 14 raw artifacts match their staged Git blobs.

This is a conditional intervention on one already-observed scene and one fixed
model. The two verification episodes are replications, not independent models
or unseen-scene evidence. The result establishes that adding the step-27
correction while holding step 23 fixed is insufficient. It neither tests step 27
alone on the unassisted trajectory nor identifies the training mechanism behind
the Q rankings. No model repair or generalization improvement is claimed.
