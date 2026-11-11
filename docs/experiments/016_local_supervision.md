# Attempt 016: double supervision for one fragile decision during continued TD

## Protocol recorded before execution

Attempt 014 kept learning rate 0.0001 and Adam state after return calibration,
but scene 10001 failed at absolute TD updates 2009 and 10000. Attempt 015
reproduced those failures and rescued both by replacing only step 21 with
`north_small`. The models then completed the route autonomously in 41 actions.
This motivates a controlled training test of the supervision at that state;
the diagnostic intervention itself did not improve either learned policy.

Create two fresh branches from the **same complete attempt 013 treatment
snapshot at absolute TD update 2000**. Restore online and target weights,
Adam moments and internal parameter steps, the TD update clock and both RNG
states exactly. Do not reset Adam, resynchronize the target or rerun the
supervised/return-calibration stages. The inherited branch has 42000 updates:
20000 supervision, 20000 return calibration and 2000 TD. Adam's internal
parameter step is 22000.

## The only training factor

Use the same ordered 117 demonstration transitions and original returns from
training seeds 10000, 10001 and 10002. Scene 10001 occupies zero-based rows
35--75. Its **one-based step 21 is zero-based row 55**, demonstrated action
`north_small (0)`. Check the row identity, encoded state hash and action
against the committed source evidence before training.

Let `h_i` denote the existing action-margin hinge loss for demonstration row
`i`, with margin 0.8. The objectives are:

- **Baseline:** `L_TD + sum(h_i) / 117`.
- **Treatment:** `L_TD + (sum(h_i) + h_55) / 117`.

Equivalently, the treatment gives supervised row 55 coefficient 2 and every
other row coefficient 1. The denominator remains **117**, not the sum of
weights 118. Thus each other supervised coefficient remains unchanged.
The global supervision multiplier remains 1. Do not duplicate transitions,
upweight that row's TD loss, change the sampling distribution or add examples.
At a shared parameter/target state, the only objective difference is
`h_55 / 117`; resulting later optimizer and target differences are consequences
of this factor.

Keep learning rate **0.0001**, gamma **0.99**, gradient clip **10**, target
synchronization interval **200**, network **341/256/256/8**, environment,
encoder, rewards, action definitions and full-batch training unchanged.
Implement the experimental weighted update separately from immutable production
and previous experiment code. Test its all-ones equivalence to the original
updater and its weighted objective/gradient against an independent formulation.

## Fixed budget and baseline reproduction

Train **8000 new TD updates per arm**, from absolute TD 2000 through 10000.
This is **16000 new main optimizer updates**, with the 42000-step shared history
reused once, not counted twice as new work. Each branch ends with 50000 total
updates and Adam parameter step 30000. Target synchronization remains on the
absolute clock at 2200, 2400, ..., 10000: 40 new synchronizations per arm.

Run the baseline first using the original updater. Require exact reproduction
of attempt 014's 8000 update records, 29 physical observations and complete
training snapshots before beginning the treatment. Require exact payload
equality for all 67 full/inference checkpoint files, including metadata,
without requiring identical serialization bytes. Preserve the original baseline evidence and record equivalence
in this new run. A failed equivalence check invalidates the comparison and must
be recorded; it is not a reason to change the treatment to obtain success.

For each arm, physically evaluate all three scenes at:

- 2000, confirming the restored initial behavior;
- every update 2001--2010, explicitly including the known failure at 2009;
- 2200 and 2201, the next ordinary target synchronization and following update;
- 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000,
  8500, 9000, 9500 and 10000.

This gives **29 checkpoints and 87 episodes per arm**, or **174 main episodes**.
All actions come from the greedy neural policy: no forced step-21 correction,
planner fallback or action override is used during evaluation. Retain every
failure and recovery. Continue through 10000 even if an intermediate model
fails; do not stop early, roll back or select a successful checkpoint as the
final result.

## Neighbor monitoring and retained diagnostics

After every update, plus the restored TD-2000 boundary, record batch-of-one Q
probes on scene 10001's fixed demonstration states at steps
**6, 18, 20, 21, 22, 23 and 24**. There are 8001 records per arm, 16002 total,
containing 112014 individual state forward passes. Steps 20--23 test the target
decision and its neighbors; steps 6, 18 and 24 retain earlier diagnostic/control
states. Save all eight Q values, demonstrated and greedy actions, and the signed
gap `Q(demonstrated) - max(Q(other))`. The probes must preserve weights, Adam,
RNG and module mode flags, and add no environment actions or optimizer updates.

The local intervention should not be judged only by step 21's gap. In particular,
check whether favoring `north_small` damages neighboring decisions that require
`north_large`, or whether failures move to another state or scene. Report each
scene's actual outcomes and first actual reference-action divergence, alongside
negative-gap intervals, minima, ties and action-rank changes in all seven probes.
Reference-state reversals, including unvisited states after a trajectory fork,
are **not measured failed episodes**.

Save inference and full training snapshots at the 29 physical checkpoints, plus
otherwise unsaved target-source states at 2400, 3400, ..., 9400. This retains
67 model files per arm, including the final full-state alias. Keep all 8000
update diagnostics and a separate supervision log per arm. Baseline update logs
retain their original schema for exact source comparison. Treatment
`demonstration_loss_before/after` denotes the weighted mean; supplementary logs
separately retain the unweighted mean hinge and selected-row hinge. These
quantities must not be conflated when comparing loss curves.

Record actual gradient norms, clipping, parameter changes, return-calibration
error and demonstration agreement. The update log's action accuracy is measured
before the update; final policy accuracy comes from its saved observation.
Keep the complete compact gap/action series and all physical outcomes in Git.
Raw full Q logs and model/data binaries stay locally with recorded hashes.

## Independent verification and accounting

Independently replay updates **2001--2201 for both arms**, reproducing weighted
updates through an independent objective implementation. This adds
**402 verification updates**, separately from 16000 main updates. Compare logs,
Q probes and saved online/target/Adam/RNG states across the first ten steps and
the target synchronization at 2200. Inspect all later model hashes, counters
and recorded target lineage without claiming a numerical replay of all 16000.

Reload policies at **2000, 2009, 2200, 2201 and 10000** from both arms, on all
three scenes: **30 additional verification episodes**, with their actual actions
counted separately. The selection is fixed regardless of which new checkpoints
fail. Read-only analysis and plots add no episodes or optimizer updates.
Reset feasibility probes are excluded from policy-action counts as before.

Commit and push the code/protocol before execution, then preserve all results,
failures, resolved settings, source/helper hashes and exact launch commit.
The code, comments and runtime messages remain English. Previous attempts,
production defaults, reports and submission archives are outside this change.

## Interpretation limits

This is a targeted mechanism test on three known training scenes from one
initialization. The target row was selected using earlier diagnostic failures;
it is not held-out validation. Improved target-state ranking alone does not
establish robust retention. All-scene success at every scheduled checkpoint is
the primary sampled-retention criterion, while dense probes describe additional
rank stability and neighbor tradeoffs. Neither establishes success at every
unobserved state or on unseen scenes. A failed treatment remains useful evidence
and will not be hidden by its final checkpoint or mean loss.

## Reproduction

Run from the repository root using the configured Python environment:

```powershell
python docs/experiments/016_local_supervision/run_local_supervision.py --reference-run results/training/td_extension_10000_20261112 --archive-dir docs/experiments/014_td_extension/evidence --output-dir results/training/local_supervision_20261112
python docs/experiments/016_local_supervision/verify_local_supervision.py --run-dir results/training/local_supervision_20261112 --output results/training/local_supervision_20261112/reload_checks.json
python docs/experiments/016_local_supervision/analyze_local_supervision.py --run-dir results/training/local_supervision_20261112 --output results/training/local_supervision_20261112/q_analysis.json
python docs/experiments/016_local_supervision/plot_local_supervision.py --run-dir results/training/local_supervision_20261112
python docs/experiments/016_local_supervision/archive_local_supervision.py --run-dir results/training/local_supervision_20261112 --destination docs/experiments/016_local_supervision/evidence
```

Local execution uses `.venv/Scripts/python.exe`. The original local binary
checkpoints and demonstration data are required; compact Git evidence does not
include those files. The runner refuses to overwrite an existing output
directory. Results are pending at the pre-execution commit.
