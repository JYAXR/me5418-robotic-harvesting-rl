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
Results are pending at the pre-execution commit.
