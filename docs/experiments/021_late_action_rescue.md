# Attempt 021: a single-action rescue of the late failures

## Protocol declared before execution

Use the exact frozen TD-9955 and TD-9956 full snapshots reconstructed and
verified in attempt 020. For scene 10001, independently restore the same model
and reset seed for each arm: autonomous greedy control, then override only
one-based step 21 with action 0 (`north_small`). The model still selects action
4 (`north_large`) at this state. Immediately resume its own greedy decisions
after the override. Do not force any other action, retrain, recalibrate, reset
Adam, modify targets, environment, reward, budget or network configuration.

Reuse the tested attempt-015 trace and paired-prefix verifier. Require exact
reproduction of the original control trajectories, identical first 20 complete
transitions, and identical fork observation and Q values. Count exactly one
forced action per corrected episode. Preserve the entire training snapshot,
including optimizer moments, counters and RNG, before and after every episode.
Keep reset route certification enabled and count its simulator actions separately.

Run four main episodes, then four fresh verification episodes with a separately
written execution loop. Compare every transition and final outcome with the main
traces. Preserve all outcomes regardless of success. There are zero optimizer
updates and no new model training; the diagnostic intervention is not a deployed
policy improvement. Success would establish that this one external correction
is sufficient for these two deterministic failures, not that all failures or
unseen layouts are fixed.

Record full trajectories, fork Q values, blocked/stationary behavior, returns,
source/model hashes, verification and actual-work accounting. Archive compact
JSON evidence and English notes in Git; keep existing reports unchanged.
Results pending at the pre-execution commit.

## Completed results: 14 November 2026

The single prescribed correction rescued **both** frozen failed models.
The formal run completed once, with no execution failure and no alternative
interventions attempted. Source hashes were checked again after all episodes.

| Model | Execution | Outcome | Actions | Return | Blocked | Stationary |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| TD 9955 | Original greedy policy | Budget exhausted | 200 | 2.479857 | 178 | 177 |
| TD 9955 | Only step 21 corrected | Success | 41 | 107.585050 | 0 | 0 |
| TD 9956 | Original greedy policy | Budget exhausted | 200 | 2.479857 | 178 | 177 |
| TD 9956 | Only step 21 corrected | Success | 41 | 107.585050 | 0 | 0 |

Neither arm incurred damage. In each paired comparison, all first 20 complete
transition records were identical. At step 21, position, encoded observation,
all eight Q values and the model-selected action were identical. The original
policy selected `north_large`; the corrected arm executed `north_small` instead.
Every action after that was again selected by the unchanged network. The
corrected action reached the next demonstrated state, and the remainder of the
executed route matched the demonstration without further overrides.

The step-21 signed Q gap, Q(north_small) minus the best alternative, was
-0.0278167724609375 at TD 9955 and -0.0157470703125 at TD 9956. These values
were not altered by the intervention. The network therefore continued to
prefer the failing action at the original fork; execution alone was corrected.

This controlled replay supports a specific causal conclusion: overriding that
single decision is sufficient to turn each of these two deterministic failures
into success while leaving the subsequent policy unchanged. It does not show
that this action is the sole possible failure mechanism, that all negative-gap
updates can be rescued similarly, or that the trained policy has improved.
The two unassisted models still fail on this scene. No production action
exception, training change, or report update was introduced.

## Verification and accounting

A separate execution loop independently reproduced all four main episodes,
comparing actual observations, Q values, proposed and executed actions, rewards,
termination flags, movement and final trajectories. It exactly reproduced both
failures and both rescues. Reset certification also matched on each reload.
The entire model/optimizer snapshot, including Adam moments, counters, target
weights and RNG, matched before and after every episode.

| Newly executed work | Main comparison | Independent verification | Total |
| --- | ---: | ---: | ---: |
| Episodes / environment resets | 4 | 4 | 8 |
| Policy-loop actions, including overrides | 482 | 482 | 964 |
| Reset certification simulator actions | 164 | 164 | 328 |
| Total simulator actions | 646 | 646 | 1292 |
| Forced actions | 2 | 2 | 4 |
| Optimizer updates | 0 | 0 | 0 |

All 38 existing single-action rescue/verification tests passed in 4.78 seconds.
Ruff passed for the new runner. The paired checks and independent transition
replays are additional integration validation of this run. These totals exclude
unit-test execution and reused historical training. There was no new scene
collection or held-out evaluation.

## Reproduction and evidence

With the verified attempt-020 source artifacts present, run from the repository
root:

```powershell
.venv/Scripts/python.exe -u docs/experiments/021_late_action_rescue/run_late_action_rescue.py
```

The runner refuses an existing output directory. Source models remain in the
original local attempt-020 run. Complete step records for every main episode,
paired outcomes, source hashes, reset certification, independent verification
and test validation are retained in the [evidence manifest](021_late_action_rescue/evidence/archive_manifest.json).
The [summary](021_late_action_rescue/evidence/summary.json) contains the outcome
and work counts; [pairs](021_late_action_rescue/evidence/pairs.json) records the
common-prefix and intervention checks. No model binary is required in this new
archive because no model was trained or modified.
