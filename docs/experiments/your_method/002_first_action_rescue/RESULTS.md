# Own-route attempt 002: one correction rescues only one of three scenes

## Result

Replacing only the first demonstrated-action disagreement restores success on
290007, but is insufficient on 290012 and 290021. Both arms use the same frozen
attempt-034 targeted model; the baseline here is not the earlier control model.
The corrected action is east_large in all three cases. Every other action is
chosen autonomously from the state actually reached after the intervention.

| Scene | Corrected step | Autonomous outcome | One-correction outcome | Return: baseline / corrected |
| --- | ---: | --- | --- | ---: |
| 290007 | 8 | Damage at step 124 | Success at step 30 | -130.048293 / 101.646837 |
| 290012 | 19 | Budget exhausted at step 200 | Budget exhausted at step 200 | -29.168251 / -24.294699 |
| 290021 | 12 | Damage at step 46 | Budget exhausted at step 200 | -47.815925 / -48.759516 |

Primary success changes from 0/3 to 1/3 on these deliberately selected failed
training scenes. This is assisted rescue, not newly learned competence or an
unseen-scene success rate. No training data or network weights were changed.

For 290007, blocking falls from 90 steps to zero and stationary steps from 87
to zero. The intervention is sufficient for success in the specified frozen
episode. However, corrected peak normalized deformation is 0.964742, so the
successful trajectory is not deformation-free and is close to the damage
threshold of 1. The result does not establish robustness to perturbations.

For 290012, peak deformation decreases from 0.777321 to 0.445839, but blocking
remains high (161 to 163 steps), with 162 stationary corrected steps and final
target distance 0.340763. For 290021, damage is avoided, but the corrected run
has 172 blocked steps, 170 stationary steps and target distance 0.208419. Its
return is slightly lower despite avoiding damage. Neither episode succeeds.

## Controlled intervention and verification

The [protocol](PROTOCOL.md), runner and tests were committed and pushed as
`532d91c` before the single physical run. The run completed from 2026-11-16
08:18:40.781 to 08:20:18.768 UTC, approximately 98 seconds. No physical restart,
extra corrective action, parameter search or budget extension occurred.

- All three autonomous baselines exactly match attempt 001's episode metrics
  and every shared historical trace field.
- Baseline/corrected prefixes match for 7, 18 and 11 steps, respectively. The
  intervention inputs and all eight Q values match exactly at the fork.
- Each corrected primary episode contains exactly one forced action; baseline
  contains none. All other actions are greedy model choices.
- Six independent fresh-load runs through the normal evaluator reproduce every
  complete transition, certification record and endpoint metric exactly.
- Online/target weights, Adam moments, update counters, configuration and saved
  RNG states are unchanged. Gradients remain disabled and optimizer steps are
  blocked. TD clock stays 12000; Adam clock stays 32000.
- All pinned source/model/data hashes remain unchanged. Offline review verifies
  24 manifest files and independently recomputes primary metrics and override
  locations from the six saved trajectories.

| Work | Count |
| --- | ---: |
| Primary / independent verification episodes | 6 / 6 |
| Primary / verification forced actions | 3 / 3 |
| Episode simulation actions, including verification | 1600 |
| Reset-certification simulation actions | 588 |
| Total simulator actions | 2188 |
| Certification calls | 12 |
| Optimizer updates / new training samples | 0 / 0 |

All 12 override-audit tests and Ruff pass. The preparation record includes the
initial three test failures caused by expecting the wrong exception type; the
underlying rejection checks worked and no simulation was involved. See
[PREPARATION.md](PREPARATION.md). The offline review also runs without simulation.

## Later disagreements: observation, not another intervention

Offline comparison of corrected trajectories with the stored demonstration
locates the next disagreement at an identical encoded input:

| Scene | Next step | Model action | Demonstration action | Model Q minus demonstration Q |
| --- | ---: | --- | --- | ---: |
| 290007 | 14 | north_small | north_large | 0.165123 |
| 290012 | 26 | north_small | north_large | 0.193405 |
| 290021 | 17 | south_large | east_small | 1.168007 |

Scene 290007 still succeeds despite its later disagreement. Thus the later
differences in the two failed scenes are diagnostic candidates, not proven
failure causes. No later action was replaced in this attempt.

A possible next controlled experiment would keep the already tested correction
fixed in both arms, then separately test the next disagreement on 290012 and
290021. That experiment has not been executed. Supervised training should not
assume these later labels are sufficient recovery actions until tested.

## Reproduction and evidence

Run [run_rescue.py](run_rescue.py) in a checkout where this attempt's evidence
directory does not already exist, or use a separate copy of the experiment
directory. The runner refuses to overwrite retained evidence. Existing evidence
can be reviewed with [review_evidence.py](review_evidence.py) in a copy where
its review.json output does not already exist.

- [Summary](evidence/summary.json), [primary episodes](evidence/episodes.json),
  [paired checks](evidence/pair_checks.json), [independent replays](evidence/replays.json).
- [Provenance](evidence/provenance.json), [status](evidence/status.json),
  [manifest](evidence/manifest.json), [offline review](review.json).
- Exact frozen model and full snapshot hashes are specified in the protocol.
