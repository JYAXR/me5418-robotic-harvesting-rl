# Attempt 014: continue preserved-Adam TD from 2000 to 10000 updates

## Protocol recorded before execution

Attempt 013's learning rate 0.0001 treatment retained all three training scenes
at its 18 scheduled checkpoints through TD update 2000. The baseline at 0.0003
failed five times on seed 10001. This attempt asks whether the lower rate truly
extends sampled retention or merely postpones degradation over a longer horizon.
It is one continued training branch, not another learning-rate sweep.

Restore the exact attempt 013 **treatment full training snapshot** at TD update
2000, including online weights, target weights, Adam moments, optimizer internal
steps, the TD update clock and RNG state. Verify the committed attempt 013
evidence, binary checkpoint hash and the complete chain back to the unchanged
attempt 011 source/configuration/data. Require exact full-state equality after
restoration and exact re-evaluation of the stored update-2000 observation before
any new update. Inference-only loading is not a valid resume mechanism.

Do **not** call a phase-boundary reset, recreate fresh Adam, zero the TD clock or
synchronize the target at resume. The target happens to equal the online network
at update 2000 because it is a synchronization point; the implementation must
still preserve the saved target exactly. Synthetic tests resume at an update
that is not a synchronization point to detect accidental resets or copies.

| Quantity | Start | End or fixed value |
| --- | --- | --- |
| Absolute TD update | 2000 | 10000 |
| New TD updates in this attempt | 0 | 8000 |
| Adam internal parameter step | 22000 | 30000 |
| Historical updates on this branch | 42000 | Reused, not rerun |
| Total updates on the branch, including supervision/calibration | 42000 | 50000 |
| Learning rate / demonstration weight / margin | 0.0001 / 1 / 0.8 | Unchanged |
| Target synchronization interval / clipping norm / gamma | 200 / 10 / 0.99 | Unchanged |

Reuse the same ordered 117 demonstration transitions: 35/41/41 from training
seeds 10000/10001/10002. Keep the 341/256/256/8 network, encoder, environment,
rewards, actual discounted demonstration returns and equally weighted
per-transition full-batch objective unchanged. No new data collection, recovery
examples, online exploration, held-out scenes or added seed identifiers are
allowed. Every update remains `TD_Huber + demonstration_margin_loss`.

## Fixed evaluation and dense Q probes

Physically evaluate all three scenes at these **absolute TD update counts**:

- 2000, to verify the resumed policy exactly;
- 2001 through 2010, to inspect the first ten continuation updates;
- 2200 and 2201, to inspect the next normal target synchronization and its next update;
- 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000,
  8500, 9000, 9500 and 10000.

This is **29 checkpoints and 87 newly executed episodes**. The 500-update
spacing alternates the phase relative to the 200-update target period. All
rollouts use the actual greedy neural policy. No forced action, planner fallback,
rollback, early stopping or success-based checkpoint selection is permitted.
Continue to update 10000 even if an intermediate model fails, and retain every
failure and recovery. Report extension retention relative to the successful
update-2000 boundary, as well as the historical combined TD sample record.

The combined attempt 013 plus 014 history contains **46 distinct absolute
checkpoints and 138 scene checks** after removing the duplicate update-2000
observation. Its inherited 54 scene checks are not counted as new evaluation
work. This extension still executes the duplicate boundary's three episodes,
included in its new total of 87. Supervision and calibration are outside this
combined TD retention history; their earlier temporary failures are not erased.

Record **batch-of-one Q probes after every new update**, plus the initial resumed
state: 8001 probe records at updates 2000--10000. Use seed 10001's fixed
demonstration states at steps **6, 18, 22 and 24**, retaining all eight Q scores,
the greedy/demonstrated action and the signed gap
`Q(demonstrated) - max(Q(other))`. Steps 6/18 cover the earlier missed turns,
step 22 covers the later step-size disagreement in attempt 012, and step 24 is
a known successful-deviation control. Restore inference mode flags after probes;
they must not alter weights, optimizer state, RNG or the training data.

These probes do not execute the environment. Report negative-gap intervals,
greedy-action disagreements, ties and minimum gaps separately from physical
episode outcomes. A negative gap at an unvisited reference state is **not** a
measured episode failure; a zero gap may or may not change argmax because of
tie ordering. Do not trigger additional physical evaluations or modify training
when a gap changes sign. Complete gap/action series and interval summaries will
be archived, while full raw Q-vector logs remain locally with recorded hashes.

Save inference and complete training snapshots at every physical checkpoint.
Also retain the otherwise unsaved target-source training states at 2400, 3400,
4400, 5400, 6400, 7400, 8400 and 9400, so every scheduled checkpoint's target can
be compared against its recorded source. Log all 8000 updates with both absolute
TD counts and local extension counts. Normal target synchronizations occur at
2200, 2400, ..., 10000: **40 new synchronizations**, without a special resume sync.

## Verification and accounting

Independently reconstruct the resume and replay **updates 2001 through 2201**:
201 verification updates covering the first ten continuation steps, the target
synchronization at 2200 and its following update. Compare all available saved
online/target/Adam tensors, all replayed metrics and all replayed Q probes.
Inspect every later snapshot, stored target-source identity, raw log and counter,
without claiming to numerically rerun all 8000 updates.

Reload policies at **2000, 2001, 2200, 2201 and 10000** and evaluate each on all
three scenes: **15 additional verification episodes**, with actual actions
counted separately. This fixed selection remains unchanged if other checkpoints
fail. A future forensic replay would be a separately recorded activity.

New experiment work is **8000 TD updates and 87 scheduled physical episodes**.
Historical work is **42000 updates** (20000 supervision, 20000 calibration and
2000 TD), counted once. Dense Q probes add zero environment steps or optimizer
updates. Independent verification adds 201 updates and 15 episodes separately.
Planner/reset feasibility probes are excluded from policy-action counts as in
earlier attempts. No reference-data collection or held-out work is added.

Predeclare and commit the implementation before running it. Record source/helper
hashes, launch commit, exact state provenance, resolved settings and all failures.
Keep full model/data/raw-log artifacts in ignored results, with complete sampled
outcomes, compact dense-gap traces, update samples and this record in Git. Code,
comments and runtime messages remain in English. Earlier evidence, reports and
production defaults are outside the edits for this experiment.

## Interpretation limits

Physical success at all 29 extension checkpoints would extend sampled retention,
not establish success at every intervening update or indefinitely. Dense Q probes
can expose action-rank reversals between checkpoints but do not replace closed-loop
evaluation. The data still contain only three known privileged routes from one
initialization; no generalization or off-route recovery claim is supported.
Any later failure is evidence that the 2000-update result did not persist over
this longer sampled horizon, even if the final model subsequently recovers.

## Reproduction

Run from the repository root with the configured Python environment:

```powershell
python docs/experiments/014_td_extension/run_td_extension.py --reference-run results/training/td_learning_rate_preserved_20261112 --archive-dir docs/experiments/013_td_learning_rate/evidence --output-dir results/training/td_extension_10000_20261112
python docs/experiments/014_td_extension/verify_td_extension.py --run-dir results/training/td_extension_10000_20261112 --output results/training/td_extension_10000_20261112/reload_checks.json
python docs/experiments/014_td_extension/analyze_td_extension.py --run-dir results/training/td_extension_10000_20261112 --output results/training/td_extension_10000_20261112/q_analysis.json
python docs/experiments/014_td_extension/plot_td_extension.py --run-dir results/training/td_extension_10000_20261112
python docs/experiments/014_td_extension/archive_td_extension.py --run-dir results/training/td_extension_10000_20261112 --destination docs/experiments/014_td_extension/evidence
```

The local execution uses `.venv/Scripts/python.exe`. The raw attempt 013
checkpoint and original attempt 011 data must be available; compact Git evidence
does not include model binaries. The runner rejects a reused output directory.
The implementation and protocol were committed and pushed before execution as
`105a034582558764595b4d1a22660752f15ca09c`.

## Execution and observed outcomes

The run completed from 2026-11-11 17:59:41.508421 UTC to 18:06:07.366578 UTC
(12 November, 01:59:41--02:06:07 in Asia/Shanghai), approximately 6 minutes
26 seconds. The saved initial full state and the three re-evaluated update-2000
episodes exactly matched attempt 013. Adam parameter steps advanced from 22000
to 30000; the absolute TD clock advanced from 2000 to 10000, with 40 ordinary
target synchronizations and no resume reset. All 8000 updates had nonzero
parameter displacement.

**Strict sampled retention fails.** The lower learning rate did not eliminate
later degradation: scene 10001 fails at absolute TD updates **2009 and 10000**.
The first observed continuation failure appears after only nine new updates.
The policy recovers at 2010 and succeeds at every scheduled checkpoint from
2200 through 9500, then fails again at the final checkpoint. This is intermittent
loss of learned behavior, rather than a monotonic collapse.

| Evaluation scope | All-three-success checkpoints | Successful individual scene checks | Failure updates |
| --- | ---: | ---: | --- |
| Inherited attempt 013, TD 0--2000 | 18/18 | 54/54 | None at scheduled checks |
| First ten new updates, TD 2001--2010 | 9/10 | 29/30 | 2009, scene 10001 |
| All attempt 014 checks, including repeated TD 2000 | 27/29 | 85/87 | 2009 and 10000, scene 10001 |
| Combined distinct TD checkpoints, boundary counted once | 44/46 | 136/138 | 2009 and 10000, scene 10001 |

Scenes 10000 and 10002 succeed at all 29 new checkpoints in 35 and 41 steps.
Scene 10001 succeeds at 27/29 checks in 41 steps and exhausts its 200-step budget
at the other two. There is no observed damage. The final policy therefore
achieves **2/3 scene success and 115/117 demonstration action matches**, compared
with **3/3 success and 116/117 matches** at the resume boundary. High aggregate
action agreement does not guarantee a successful closed-loop route.
The final update log's `demonstration_action_accuracy=1.0` is measured before
that update; it must not replace the final observation's post-update 115/117.

The main experiment executes 87 episodes and **3711 actual policy actions**.
Historical evaluation actions remain inherited evidence; the combined
deduplicated history contains 5700 actions, not 5700 newly executed actions.
There is no new demonstration collection, online exploration or held-out test.
The 42000 inherited optimization steps are reused once; this branch finishes
with 50000 total steps across supervision, calibration and TD.

## Both failures begin at a new step-size decision

At both failed checkpoints, the first 20 actions of scene 10001 match the
demonstration. At **step 21**, the policy selects `north_large (4)` instead of
`north_small (0)`. The recorded state and action prefix are checked against the
demonstration, so the Q comparison is at the actual shared state before the
first deviation.

| TD update | Q(north_small), demonstrated | Q(north_large), selected | Demonstrated minus strongest other | Episode outcome |
| --- | ---: | ---: | ---: | --- |
| 2009 | 84.930313 | 84.947029 | -0.016716 | Budget exhausted, 200 steps |
| 10000 | 84.809303 | 84.869194 | -0.059891 | Budget exhausted, 200 steps |

Both failed rollouts record return 2.479857, 178 blocked steps and 177 stationary
steps. Their first deviation is a large-versus-small movement choice, rather
than the earlier step-6 or step-18 missed turns. The original calibrated policy
already had only a **+0.007416** demonstrated-action gap at this step-21 state.
This identifies another fragile action boundary. No forced-action rescue was
performed in this attempt, so this evidence locates the first deviation and its
subsequent failure; it does not independently prove that correcting that single
action would rescue the episode.

All four predeclared sentinel gaps are positive at both physical failures.
Step 21 was outside the fixed probe set, and reference states after the fork
need not be visited by the deviating policy. This is a concrete limitation of
monitoring only previously problematic states. The probe set was not changed
after seeing either failure.

## Dense Q rankings and optimization diagnostics

The complete 8001 probe records include the resumed boundary and every new
update. The compact archive retains every gap and greedy action; raw eight-score
vectors remain locally with their hashes and sampled vectors in Git.

| Reference step in scene 10001 | Negative gaps among 8000 new updates | Minimum signed gap | Update at minimum |
| --- | ---: | ---: | ---: |
| 6 | 394 | -0.323555 | 2784 |
| 18 | 0 | +0.105278 | 2047 |
| 22 | 0 | +0.224068 | 2410 |
| 24, successful-deviation control | 102 | -0.211525 | 2053 |

Step 6 first reverses at TD 2041 and has 267 negative-gap intervals, all entirely
between physical checkpoints; its last negative gap is at 9950. These are
**reference-state ranking reversals, not 394 measured episode failures**.
Step 24 has 103 negative records when the inherited TD-2000 boundary is included,
across 68 intervals, with its last at 3655. None of the four series has an exact
zero-gap tie. Positive sampled gaps and successful sampled episodes therefore
do not establish stability at every intervening update.

Mean pre-clipping gradient norm is 17.855495; 7149/8000 updates exceed the fixed
clip threshold of 10. Mean parameter displacement per update is 0.001959, with
nonzero minimum 0.000811. Final net parameter displacement is **2.266949** from
the resumed model and **2.911456** from the calibrated model. The parameters
continue adapting; apparent retention is not caused by a frozen optimizer.
These diagnostics do not by themselves establish the cause of either failure.

Final demonstration-return RMSE is **0.499196**, versus **0.461398** at resume.
Final action-margin loss is **0.044419**, versus **0.063221** at resume.
Thus the aggregate supervision loss can improve while a critical action ranking
and actual episode outcome deteriorate. Average losses over this continuation
are descriptive because the target network changes during training.

## Verification and retained evidence

- The full suite passed **473 tests in 95.79 seconds**. One independent replay
  integration test added after collection passed separately in 5.87 seconds,
  bringing checked coverage to 474 distinct tests. Ruff and whitespace checks
  passed. All new code, comments and runtime messages are English.
- Independent verification passed all 8000 raw update records, 8001 sentinel
  records, 29 physical observations and 67 model artifacts. Saved target tensors
  match the correct historical online snapshot, including the otherwise unsaved
  target generations at 2400, 3400, ..., 9400.
- Independent replay reproduced **201 updates, TD 2001--2201**, with exact
  numerical update logs, Q probes, online/target/Adam tensors and both RNG states
  at all available compared checkpoints. Later updates are audited through
  recorded hashes, counters and target lineage, not claimed as independently
  retrained.
- The fixed reload selection at 2000, 2001, 2200, 2201 and 10000 reproduces all
  stored outcomes, including the final failure. This verification adds
  **15 episodes and 744 actions**, separately from the main experiment. It does
  not add an adaptive replay of update 2009.
- Read-only Q analysis verifies 69 model files, including the two source
  inference models, and reproduces all 117 initial state Q vectors exactly.
  It adds no optimizer updates, resets or environment actions.
- The black-and-white outcome table and dense-gap figure were visually checked.
  Every failure is retained in the tables, raw observations and analysis.

See [complete outcome table](014_td_extension/evidence/extension_outcomes.png),
[dense Q gaps](014_td_extension/evidence/sentinel_gaps.png),
[Q and retention analysis](014_td_extension/evidence/q_analysis.json),
[independent verification](014_td_extension/evidence/reload_checks.json), and
[archive manifest](014_td_extension/evidence/archive_manifest.json).
The manifest retains hashes for all archived evidence and complete raw logs.
Git receives 170 sampled update records and 171 sampled raw Q records alongside
the complete compact gap/action series and all 87 scheduled outcomes.
Models and full raw logs remain in the ignored run directory; no report or
submission archive is modified.

The launch provenance records a dirty working tree because of the pre-existing,
untracked report document. Experiment code was committed before launch, and
the recorded source hashes and independent source checks passed.

## Conclusion and next question

The 0.0001 learning rate with supervision weight 1 and preserved Adam improves
the earlier sampled transition, but **does not stably retain all three scenes
through 10000 TD updates**. The previous all-success result covered a shorter
sampled horizon. This extension exposes both a newly observed step-21 failure
and reference-state rank fluctuations between successful checkpoints.
The final failed model is retained without rollback or replacement.

A next separately recorded diagnostic should test whether correcting only the
step-21 action rescues these two failures before changing the training objective.
Broader state-margin monitoring would also cover fragile decisions that were
absent from the four fixed probes. Neither activity is performed here. All
results remain restricted to these three known demonstration scenes and one
inherited initialization.
