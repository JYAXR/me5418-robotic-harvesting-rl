# Attempt 017: retain actual-return calibration during TD learning

## Protocol recorded before execution

Attempt 016 protected scene 10001 step 21 by doubling its supervised sample
coefficient, but sampled three-scene retention fell from 27/29 to 26/29.
New failures first diverged at step 6, and neighboring step 22 acquired 63
negative Q gaps between physical checkpoints. Its final demonstration-return
RMSE also increased from the baseline's 0.499196 to 0.630023. These observations
motivate testing a global value constraint rather than choosing another
individual sample. They do not establish that value drift caused the failures.

The new treatment retains the existing TD objective and adds a persistent
constraint toward the actual discounted returns of the original demonstrations.
The comparison uses the verified **attempt-016 baseline**, where every
supervised coefficient is 1. The locally reweighted treatment from attempt 016
is background evidence, not the matched control for this single-factor trial.

## One additional objective term

For the same ordered 117 transitions, define:

- Historical control: `L_TD + L_margin`.
- New treatment: `L_TD + L_margin + 1.0 * L_MC`.
- `L_MC = mean(SmoothL1(Q(s_i, a_demo_i), G_demo_i))`.

Each term uses its original mean over 117 rows. The total is not divided by
three, and the TD and MC targets are not blended into one target. All 117
supervised margin coefficients remain 1, with margin 0.8; there is no selected
row weighting. The MC coefficient is fixed at 1 before execution and will not
be tuned in response to intermediate outcomes.

The targets `G_demo` are the complete actual discounted demonstration returns
already used for calibration in attempt 011, computed separately for scenes
10000, 10001 and 10002 with gamma 0.99. Reuse their exact recorded tensor order
and values. They remain detached and fixed throughout TD; do not regenerate
them from a target network or accumulate rewards across episode boundaries.
They represent returns of the demonstration policy, not proven optimal Q
values. An MC constraint could also resist improvements beyond that policy.

The general use of demonstration supervision alongside TD has precedent in
[Hester et al., Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732).
This experiment's fixed full-trajectory MC anchor is a project-specific
addition; it does not implement the full DQfD algorithm or establish the same
performance claims. No external implementation is copied.

## Exact continuation and historical control

Restore the same successful full TD-2000 snapshot used in attempts 014 and
016: online and target networks, Adam moments and parameter steps, both RNG
states and the absolute TD clock. The inherited history is 20000 supervision,
20000 return calibration and 2000 TD updates. Adam starts at parameter step
22000. Do not rerun either pretraining stage or resynchronize the target at
the boundary.

Keep learning rate 0.0001, gamma 0.99, gradient clipping at 10, target
synchronization interval 200, network 341/256/256/8, full-batch data order,
environment, rewards, encoder and action definitions unchanged. Updates occur
in the existing experimental framework; production and earlier experiment
implementations remain immutable.

The historical control is attempt 016's completed baseline at commit
`e51effb2f5e0e05fb70486088d38ab235a395106`. Before new training, verify its
committed archive, source chain, raw logs, complete model hashes, exact
TD-2000 state and the earlier exact-reproduction gate against attempt 014.
Label it explicitly as reused historical evidence. Do not report its 8000
updates, 87 episodes or 3711 policy actions as newly executed work.

Train the new treatment for exactly **8000 updates**, TD 2001 through 10000.
It ends with 50000 cumulative updates on its branch and Adam step 30000.
The ordinary target synchronizations remain at 2200, 2400, ..., 10000.
Do not stop early, roll back a failure, freeze the network, override actions
or replace the prescribed final model with an earlier successful checkpoint.

## Evaluation and diagnostic coverage

Use the same 29 physical checkpoints:

`2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
2200, 2201, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500,
7000, 7500, 8000, 8500, 9000, 9500, 10000`.

Evaluate all three known scenes at each checkpoint using only greedy network
actions, giving **87 new main episodes**. Check the restored initial behavior
against the source before updating. Record success, damage, budget exhaustion,
return, action count, blocked/stationary actions and the first actual
reference-action divergence. Retain all failures and subsequent recoveries.

After every update and at the restored boundary, record all eight Q values on
scene 10001's fixed demonstration states at steps **6, 18, 20, 21, 22, 23, 24**,
one input per forward pass. Save greedy/demo labels and the signed difference
`Q(demo) - max(Q(other))`, using the same numerical procedure as attempt 016.
There are 8001 new dense records and 56007 forward passes. These probes must
preserve module modes and training/RNG state and execute no environment actions.

Report every negative-gap interval, ties, minima, final gaps and paired
changes against the historical control. Inspect neighboring steps 20--23 and
the earlier fragile step 6 together. A fixed-state reversal does not constitute
a measured rollout failure, and later reference states may be unvisited after
an actual trajectory diverges.

Save the same 67 model files: 29 inference checkpoints, 29 complete training
snapshots, eight otherwise unsaved target-source snapshots, and the final
full-state alias. Keep all 8000 update diagnostics, separately identifying TD,
unweighted margin and MC losses, their sum, parameter change and pre-clipping
gradient norm. Before/after TD losses use identical frozen Bellman targets.
Also record the difference between those targets and fixed MC returns.
Final action accuracy must be measured after updating, not taken from the
raw log's pre-update accuracy field.

## Verification and accounting

Synthetic tests must establish exact original-updater equivalence at MC
coefficient 0, independent objective/gradient equivalence at coefficient 1,
proper target timing and Adam continuity, detached return targets and
validation before mutation.

After the run, independently implement and replay TD 2001--2201 for both the
historical control and the new treatment. Compare logs, probes and saved full
states, including the target synchronization at 2200. This is **402 additional
verification updates**, not a rerun of the historical 8000-step control or a
numerical replay of every main update. Audit all later snapshot hashes,
counters, finite diagnostics and target lineage.

Reload only the new treatment's policies at TD **2000, 2009, 2200, 2201 and
10000**, each on all three scenes: **15 verification episodes**, with actual
actions counted separately. Previous control reload evidence is reused and
must not be counted again. Read-only analysis and figures add no actions,
resets or optimizer updates. Reset feasibility probes are excluded from
policy-action counts, consistently with earlier attempts.

New main work is therefore 8000 optimizer updates and 87 physical episodes;
shared historical pretraining and historical control costs remain separate.
No new demonstration collection, online interaction for training or held-out
scene access occurs in this experiment.

## Decision rule and scope

The primary criterion is retaining success on all three scenes at **every**
scheduled checkpoint. Compare the whole trajectory, not only the final
checkpoint or the average loss. Dense neighboring ranks and return
calibration provide separate diagnostics; improved RMSE cannot substitute for
successful behavior.

This is one fixed-coefficient test on three known training scenes from one
initialization. Record a failed or mixed result without changing its budget
or selecting another coefficient within the attempt. Even if it passes,
multi-layout training and independent-scene evaluation remain necessary before
claiming generalization or promoting this as a broadly reliable policy.

## Reproduction

Run from the repository root in the configured environment:

```powershell
python docs/experiments/017_mc_anchor/run_mc_anchor.py --reference-run results/training/local_supervision_20261112 --archive-dir docs/experiments/016_local_supervision/evidence --output-dir results/training/mc_anchor_20261113
python docs/experiments/017_mc_anchor/verify_mc_anchor.py --run-dir results/training/mc_anchor_20261113 --output results/training/mc_anchor_20261113/reload_checks.json
python docs/experiments/017_mc_anchor/analyze_mc_anchor.py --run-dir results/training/mc_anchor_20261113 --output results/training/mc_anchor_20261113/q_analysis.json
python docs/experiments/017_mc_anchor/plot_mc_anchor.py --run-dir results/training/mc_anchor_20261113
python docs/experiments/017_mc_anchor/archive_mc_anchor.py --run-dir results/training/mc_anchor_20261113 --destination docs/experiments/017_mc_anchor/evidence
```

The local interpreter is `.venv/Scripts/python.exe`. Original local model and
data binaries are required; the Git evidence archive contains compact records
and hashes. Refuse to overwrite any earlier run. Commit the protocol and
implementation before execution, then record and push all outcomes.

Before execution, the full suite passed **655 tests in 77.40 seconds** and
Ruff passed. The 77 added focused tests cover update semantics, independent
verification and analysis. A read-only real-source preflight verified the
complete TD-2000 state, 117 transitions, fixed returns and historical control;
it performed no optimizer updates or environment actions.

Results are pending at the pre-execution commit.
