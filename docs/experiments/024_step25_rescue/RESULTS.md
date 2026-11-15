# Attempt 024 results: one off-reference recovery action

Completed on 15 November 2026. The [protocol](../024_step25_rescue.md), runner
and intervention tests were committed before execution in `ecc89e4`.
The protocol file remains unchanged so its recorded source hash stays valid.

## Outcome

All five frozen failures recover after replacing only scene 10001's one-based
step 25 with `north_large` (action 4). Every unassisted control reproduces its
attempt-023 failure exactly. The first 24 complete transitions and the step-25
observation, proposed action and Q values are identical within each pair.

| Frozen model | Unassisted outcome | One-action correction | Q(north_large) - Q(east_large) at step 25 |
| --- | --- | --- | ---: |
| TD 2031 | Budget exhausted, 200 steps | Success, 42 steps | -0.168976 |
| TD 2226 | Budget exhausted, 200 steps | Success, 42 steps | -0.063408 |
| TD 2344 | Budget exhausted, 200 steps | Success, 42 steps | -0.044746 |
| TD 2345 | Budget exhausted, 200 steps | Success, 42 steps | -0.081451 |
| TD 2346 | Budget exhausted, 200 steps | Success, 42 steps | -0.093033 |

All five pairs share these episode metrics:

| Metric | Unassisted | Corrected |
| --- | ---: | ---: |
| Successes among the five selected failures | 0/5 | 5/5 |
| Undiscounted episode return | 2.4432439163 | 107.6253690034 |
| Episode steps | 200 | 42 |
| Collision-blocked steps | 177 | 1 |
| Stationary steps | 176 | 0 |

Step 22 remains `north_small` (action 0), exactly as chosen by each model;
it is not changed to the demonstration's `north_large`. At step 24,
`east_small` makes partial progress but is blocked. All five models then
encounter the same encoded off-reference observation at step 25. Their
unassisted policies choose `east_large`, and every subsequent action is
an eastward small or large move that leaves the robot stationary through
step 200.

The correction moves the robot from approximately `(0.3296593, 0.4872217)`
to `(0.3296593, 0.5472216)`. The model then independently chooses a feasible
`east_large` at step 26 and finishes at step 42. The corrective action's
immediate reward is slightly negative (-0.0121472120), despite its useful
longer-term outcome. This observation alone does not explain why training
ranked it below the blocked action; no new gradient or reward ablation was
performed here.

## Verification and actual work

Each arm reloads the same complete frozen snapshot and resets scene 10001.
No network, target, Adam state, learning rate, loss coefficient, reward,
environment setting, budget or training data changes. Before and after every
episode, full saved training state, including counters and agent/Torch RNG
states, matches exactly. Reset reference-route certification stays enabled.
There are zero optimizer updates and no held-out scenes.

Ten additional episodes reload all five models and repeat both arms using
the original episode helper with an independently spelled-out step-25 policy
adapter. They match every recorded decision observation, Q value, proposed
and executed action, complete robot path and final outcome. This second loop
verifies the trajectory and aggregate metrics; it does not separately compare
every individual transition's reward/termination/motion fields. The main
paired audit does compare all retained fields for the first 24 transitions.

| Formal work | Main pairs | Independent verification | Total |
| --- | ---: | ---: | ---: |
| Episodes / resets | 10 | 10 | 20 |
| Evaluated policy actions, including overrides | 1210 | 1210 | 2420 |
| Copied-simulator certification actions | 410 | 410 | 820 |
| Total simulator actions | 1620 | 1620 | 3240 |
| Forced actions, included above | 5 | 5 | 10 |
| Optimizer updates | 0 | 0 | 0 |

The run completed once, without a formal execution failure, from
04:38:35.998533 to 04:42:38.710659 UTC. Unit tests passed 55 cases in 9.90 s;
after final guard additions, the 12 intervention-specific tests passed again
in 3.87 s. Ruff passed. Unit-test simulator work is excluded from the formal
experiment counts above. Source dependencies, archived historical evidence
and the five model files retain their recorded hashes.

## Interpretation and next question

The intervention is sufficient to rescue these five known frozen failures
while leaving their step-22 divergence intact. Thus, that earlier divergence
does not make failure unavoidable: the policy still reaches a recoverable
state, but its step-25 ranking favors a blocked eastward action.

This is an external diagnostic correction. The learned policies remain
unchanged and still fail without it. The five models were selected because
they failed on one known training scene; 5/5 assisted success is not a
population success rate, evidence of generalization, or proof of a unique
underlying failure cause. No new evaluation of scenes 10000 or 10002 was
needed to answer this intervention question, and none was run in this attempt.

A possible next controlled experiment is to teach the recovery action at the
actual off-reference observation while preserving the current TD settings,
then test autonomous behavior over the full training interval and all three
known scenes. The original demonstration's action at time index 25 must not
be used as the label for this different state. This experiment is not performed
or claimed in attempt 024.

## Reproduction and evidence

From the repository root, with the verified attempt-023 artifacts and its
five full model snapshots available:

```powershell
.venv/Scripts/python.exe -u docs/experiments/024_step25_rescue/run_step25_rescue.py
```

The runner refuses to overwrite its output or archive. Historical snapshots
remain under `results/training/neighbor_rollouts_20261114/models`; attempt 023
records their deterministic reconstruction. The main run output is
`results/training/step25_rescue_20261115`. The immutable pre-run protocol and
runner remain bound to the recorded hashes.

The [archive manifest](evidence/archive_manifest.json) binds 18 JSON evidence
files (1,495,644 bytes), including both complete traces for every model,
[paired outcomes](evidence/pairs.json), [summary and work counts](evidence/summary.json),
[ten independent verifications](evidence/verification.json), reset certification,
[post-run analysis](evidence/analysis.json), [validation](evidence/validation.json)
and source provenance. Analysis and validation were added after the successful
run; the original 16 runner-produced JSON files were retained byte-for-byte.
All code and comments are English. No report or production default changed.
