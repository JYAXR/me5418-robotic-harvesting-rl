# Attempt 023: complete episodes at step-22 ranking reversals

## Protocol declared before physical evaluation

Freeze attempt-022 models (TD coefficient 0.5, row-55 supervised weight 2,
all other weights 1, learning rate 0.0001, retained Adam, no MC objective).
Choose four negative-gap intervals using only the existing dense reference
log: first occurrence, global minimum gap, longest contiguous interval
(earliest on ties), and last occurrence. Include each entire interval and one
adjacent update on both sides. Do not select models based on new episode outcomes.

The recorded selection is 2031--2032, 2225--2229, 2344--2346 and 9847.
The global minimum is TD 2345 (-0.31208038330078125); the longest interval
contains five updates. Including surrounding checks gives 19 models and
57 autonomous episodes on scenes 10000, 10001 and 10002. Eleven selected
models have negative step-22 reference gaps; the other eight are positive-gap
neighbors. This samples 11 of 690 reversals, not every negative model.

Restore the full source snapshots at TD 2000, 2200 and 9800. Replay the three
200-update segments to TD 2200, 2400 and 10000, requiring exact raw update
logs, all seven reference Q vectors and full endpoint states. Independently
replay the same segments using the separately implemented objective and
compare all 19 selected full states: 1200 reconstruction updates total.
These recover already-trained models and do not extend training or change any
hyperparameter. Preserve all new full snapshots locally.

Run unchanged greedy episodes with no forced actions, recording all decisions,
encoded-state hashes and Q values. At actual steps 21 and 22, compare reached
states and Q vectors with the original demonstrations and dense records.
Separate first divergence, actual failure, successful alternate routes and
unvisited reference states. Preserve reset certification and count its copied
simulation actions separately. Require unchanged full training state after
all evaluations. Reload the four anchor models (2031, 2227, 2345, 9847) on all
three scenes using the original uninstrumented rollout helper: 12 additional
episodes, with exact trajectory and reset-certification comparisons.

Record failures, recoveries and limitations honestly. No action rescue or new
training remedy is part of this attempt. Results are targeted diagnostics on
known scenes, not an unbiased success rate or generalization test. Keep code
and comments English, leave reports unchanged, and commit the evidence to Git.
Results pending at the pre-execution commit.

## Completed results: 14 November 2026

The selected reversals include genuine autonomous episode failures, but a
negative reference gap is not sufficient to predict failure. The 19 evaluated
models produced **52/57 successful episodes**, with all three scenes successful
at **14/19 models**. All scene-10000 and scene-10002 episodes succeeded.
Scene 10001 failed at **TD 2031, 2226, 2344, 2345 and 2346**.

Among the 11 selected negative-gap models, scene 10001 failed at five and
succeeded at six. All eight positive-gap neighboring models succeeded. These
are targeted diagnostic counts for 11 of the original 690 reversals, not an
estimated failure rate over all training updates.

| TD update | Step-22 reference gap | Scene 10001 | Steps | Return |
| --- | ---: | --- | ---: | ---: |
| 2030 | 0.10669708 | Success | 41 | 107.585050 |
| 2031 | -0.00981140 | Budget exhausted | 200 | 2.443244 |
| 2032 | -0.02498627 | Success | 42 | 107.625369 |
| 2033 | 0.13679504 | Success | 41 | 107.585050 |
| 2224 | 0.04055786 | Success | 41 | 107.585050 |
| 2225 | -0.11327362 | Success | 42 | 107.625369 |
| 2226 | -0.06587219 | Budget exhausted | 200 | 2.443244 |
| 2227 | -0.01306152 | Success | 42 | 107.625369 |
| 2228 | -0.02016449 | Success | 41 | 107.585050 |
| 2229 | -0.07618713 | Success | 41 | 107.585050 |
| 2230 | 0.05725861 | Success | 41 | 107.585050 |
| 2343 | 0.09975433 | Success | 41 | 107.585050 |
| 2344 | -0.13024139 | Budget exhausted | 200 | 2.443244 |
| 2345 | -0.31208038 | Budget exhausted | 200 | 2.443244 |
| 2346 | -0.22211456 | Budget exhausted | 200 | 2.443244 |
| 2347 | 0.14500427 | Success | 41 | 107.585050 |
| 9846 | 0.08782196 | Success | 41 | 107.585050 |
| 9847 | -0.00579071 | Success | 42 | 107.625369 |
| 9848 | 0.01316071 | Success | 41 | 107.585050 |

## Actual-state and trajectory evidence

Every evaluated policy reached the original encoded reference observation at
step 22 after the same demonstrated action prefix. For all 11 negative-gap
models, the actual eight-action Q vector matched the recorded reference probe
exactly. Each selected `north_small` (action 0) instead of demonstrated
`north_large` (action 4), making step 22 the first reference-action divergence.
The monitored step-21 behavior was preserved before this fork. Thus, these
reversals occurred on visited policy inputs, not merely hypothetical states.

All five failures exhausted the 200-step budget without damage. Each recorded
177 blocked and 176 stationary steps, with return 2.443243916332722. The six
successful negative-gap models followed two patterns: TD 2032, 2225, 2227 and
9847 succeeded in 42 steps, with one blocked step and return 107.6253690034151;
TD 2228 and 2229 succeeded in 41 steps without blocking, with return
107.58505000025033. Every positive-gap neighbor completed the original 41-step
route without blocking. A larger negative gap therefore cannot be treated as
a monotonic measure of physical failure severity.

A read-only comparison of adjacent models with different outcomes locates a
second decision split. TD 2031/2032, TD 2225/2226 and TD 2226/2227 executed
identical actions through step 24. At step 25 they had the same encoded
observation, but the failing model selected `east_large` (action 6) and the
successful model selected `north_large` (action 4). This state was already off
the demonstrated trajectory. The original demonstration's action at time index
25 is consequently not a valid expert label for this new observation.

These comparisons show why the step-22 reversal alone does not determine the
outcome: the subsequent policy can recover or enter a blocked continuation.
They do not isolate either step 22 or step 25 as the sole cause. No action was
forced in this experiment. A separate single-action intervention would be
required to test whether correcting a specified decision suffices to rescue
these frozen failures.

The conclusion strengthens the limitation recorded in attempt 022: its
selected step-21 improvement does not provide globally stable task retention.
The newly evaluated checkpoints reveal failures that its sparse physical
schedule had not measured. We did not evaluate the historical coefficient-one
supervision control at these exact checkpoints, so this experiment does not
establish a per-checkpoint causal comparison against that different training
arm. The earlier sampled results remain historical measurements, not revised
claims that every intermediate model succeeds.

## Verification and actual work

The main and independent reconstructions each replayed three 200-update
segments. All 1200 replayed update records and associated seven-state Q probes
matched the source, every segment reached the identical full endpoint, and
all 19 selected full states matched between implementations. This recovered
existing models; there were no new remedy-training updates or hyperparameter
changes. Source code, archived evidence and model hashes remained unchanged.

Twelve uninstrumented reload episodes at four anchors exactly reproduced the
complete recorded trajectories and reset-certification checks. This includes
the failures at TD 2031 and 2345 and the successful alternate routes at TD 2227
and 9847. Every main evaluation and reload preserved the full model, target,
Adam, counter and RNG snapshot. Reset route certification remained enabled.

| New formal work | Main evaluation | Independent reload | Total |
| --- | ---: | ---: | ---: |
| Episodes / resets | 57 | 12 | 69 |
| Policy actions | 3022 | 788 | 3810 |
| Copied-simulator certification actions | 2223 | 468 | 2691 |
| Total simulator actions | 5245 | 1256 | 6501 |

The 1200 reconstruction optimizer updates are separate from these action
counts. There were zero forced actions, new demonstrations, online training
steps or held-out scenes. Existing selection/update/reset tests passed **53
cases in 18.84 seconds**. After a style-only set-literal cleanup, all five
selection tests passed again in 7.74 seconds. Ruff passed for the final files.
Unit-test execution is not included in formal experiment counters. The main
run and analysis each completed once without a formal execution failure.

## Reproduction and evidence

Run from the repository root with the immutable attempt-022 local artifacts:

```powershell
.venv/Scripts/python.exe -u docs/experiments/023_neighbor_rollouts/run_neighbor_rollouts.py
.venv/Scripts/python.exe docs/experiments/023_neighbor_rollouts/analyze_neighbor_rollouts.py
```

Both tools refuse existing output destinations. The reconstructed full model
files remain under `results/training/neighbor_rollouts_20261114/models`.
The [archive manifest](023_neighbor_rollouts/evidence/archive_manifest.json)
retains full trajectories and actual decision Q values, selection criteria,
source/model hashes, reconstruction checks, reset accounting, validation and
reload evidence. The [analysis](023_neighbor_rollouts/evidence/analysis.json)
records all failures, actual reference-state visits and the three paired
step-25 splits. [Work accounting](023_neighbor_rollouts/evidence/accounting.json)
and [verification](023_neighbor_rollouts/evidence/verification.json) distinguish
historical reconstruction from physical evaluation.

All code and comments are English. No report or production default changed.
