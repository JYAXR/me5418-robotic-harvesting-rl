# Attempt 034: final autonomous recovery, with incomplete retention

## Main result

Doubling only scene-10002 step-23/27/28 supervision weights produces final
autonomous success on all three original scenes. Equal-budget unchanged-weight
continuation ends at 2/3 successes. Every evaluation uses zero action overrides.

| Scene | Control outcome | Targeted outcome | Control / targeted steps | Control / targeted return |
| --- | --- | --- | --- | --- |
| 10000 | Success | Success | 38 / 40 | 107.589934 / 107.565934 |
| 10001 | Success | Success | 47 / 47 | 107.282922 / 107.282922 |
| 10002 | Budget exhausted | Success | 200 / 43 | -41.614794 / 107.957967 |

Targeted 10002 finishes with zero peak deformation and zero blocked/stationary
steps, travel distance 1.704998456 and target distance 0.013752203. Control has
174 blocked and 171 stationary steps, with peak deformation 0.295497369.

The other two scenes retain success at all six sampled checkpoints (12/12 per
arm). However, targeted 10000 has a small final efficiency regression: two extra
steps, 0.119999886 extra travel and 0.023999977 less return; deformation stays
zero. Final 10001 metrics match source/control exactly, including peak deformation
0.101333319. Retained task success does not mean every performance measure is unchanged.

## Training outcomes and limits

| Additional TD updates | Control success / 3 | Targeted success / 3 | 10002 outcome: control / targeted |
| ---: | ---: | ---: | --- |
| 0 | 2 | 2 | Damage / damage |
| 10 | 3 | 3 | Success / success |
| 100 | 2 | 2 | Blocked timeout / blocked timeout |
| 500 | 2 | 2 | Blocked timeout / blocked timeout |
| 1000 | 2 | 2 | Blocked timeout / blocked timeout |
| 2000 | 2 | 3 | Blocked timeout / success |

Individual successes across checkpoints are 13/18 versus 14/18. Among the five
post-start checks, all-scene success is 1/5 versus 2/5. These correlated checkpoints
are not independent statistical replicates. The experiment supports final
acquisition, not uninterrupted retention. No checkpoint selection, budget
extension or additional weight search occurred. Longer training, multiple
training seeds, the other 30 training scenes and unseen scenes remain untested.

## Exact change and observed actions

The three states already exist in the unchanged 1440-transition, 33-scene dataset:
zero-based rows 98/102/103, labels 4/0/4. Their encoded hashes match the physically
visited correction states in attempt 033. No transitions were duplicated or added
to TD. The treatment changes only their action-margin weights from 1 to 2,
adding `(hinge[98] + hinge[102] + hinge[103]) / 1440` to the control objective.

Margin remains 0.8; existing row-40/55 weights remain 2 and the previous recovery
term retains its /117 normalization. TD coefficient 0.5, global supervision
coefficient 1, learning rate 0.0001, gamma, network, clipping, target interval,
dataset order and all other training settings remain unchanged. No MC term or
fresh optimizer is introduced. Both arms retain the exact source Adam moments
and advance TD clock 10000 to 12000 and Adam clock 30000 to 32000.

| Reference step | Demonstrated action | Control greedy / margin | Targeted greedy / margin |
| ---: | --- | --- | --- |
| 23 | north_large | north_large / +0.088455 | north_large / +0.219574 |
| 27 | north_small | east_large / -0.237892 | north_large / -0.045181 |
| 28 | north_large | east_large / -0.506989 | east_large / -0.194420 |

Margins are demonstrated Q minus the best competing Q, evaluated one input at a
time. The final policy has **not learned all three literal reference choices**.
Its successful physical trajectory executes north_large at steps 23, 27 and 28.
The larger step-27 move leads to a different step-28 input, where the policy
autonomously chooses north_large and eventually succeeds. Reference-label
agreement must therefore be distinguished from task success. The matched
comparison supports a final behavioral benefit of the supervision change,
without proving the internal mechanism or necessity of each label.

## Verification and execution record

The [protocol](../034_scene10002_supervision.md) and code were committed and pushed
as `119f647` before training. Both new tests passed on their first run and Ruff
passed. Tests verify the exact three-row gradient change, exact control equivalence
to the old updater, and independent treatment update/full-state equivalence.
They perform four optimizer updates and no simulator episodes, counted separately.

The single run completed without execution errors or restarts from 2026-11-15
19:15:10.957 to 19:34:38.438 UTC (2026-11-16 03:15--03:34 Asia/Shanghai), about
19 minutes 27 seconds. All 4000 main updates are nonzero. Independent replay of
the first/last ten updates per arm adds 40 updates and exactly reproduces metrics
and complete states. Each arm has ten target synchronizations; final online and
target networks match. All source/data/implementation hashes stay unchanged.

There are 36 primary autonomous episodes and six normal-evaluator endpoint
reloads. Reload metrics and reset certification match, with unchanged learner
states during evaluation. All six starting traces match attempt 030 byte-for-byte;
all reset geometries match per seed. Every primary trace action is its Q argmax.

| Work | Count |
| --- | ---: |
| Main optimizer updates | 4000 (2000 per arm) |
| Independent verification updates | 40 |
| Unit-test optimizer updates | 4 |
| Evaluation episodes, including reloads | 42 |
| Executed episode actions | 3084 |
| Reset-certification simulation actions | 1638 |
| Total simulator actions | 4722 |
| Forced overrides / new training samples | 0 / 0 |

Historical collection/training costs belong to earlier attempts. Final known-scene
recovery is supported; continuous retention and generalization are unestablished.

## Artifacts

Both fixed endpoints are retained as experimental candidates. Previous models
are unchanged. Portable resume states include Adam and RNG states.

- Targeted inference: `models/supervision_034_targeted.pt`, SHA-256
  `202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b`.
- Control inference: `models/supervision_034_control.pt`, SHA-256
  `a8a519b1944afec3bd5ac063b34db6c75136c1661c3c4f8fbc3412859e58e018`.
- [Targeted resume state](checkpoints/targeted_state_12000.pt),
  [control resume state](checkpoints/control_state_12000.pt).
- [Comparison and exports](analysis_summary.json), [summary](evidence/summary.json),
  [all evaluations](evidence/evaluations.json), [Q probes](evidence/probes.json).
- [Independent updates](evidence/update_verification.json),
  [endpoint reloads](evidence/reload_verification.json),
  [provenance](evidence/provenance.json), [archive manifest](evidence/archive_manifest.json).
- [Artifact verification](artifact_verification.json): 47 raw evidence files
  (5680586 bytes) match the run, 133 source/input hashes are unchanged, and 55
  staged artifacts match local bytes, including both inference and resume models.

```bash
python scripts/evaluate_dqn.py --checkpoint models/supervision_034_targeted.pt --episodes 3 --seed 10000
```
