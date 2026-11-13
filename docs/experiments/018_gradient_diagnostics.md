# Attempt 018: gradient conflicts and Adam effects on fragile action gaps

## Question and scope

Attempt 017 improved actual-return calibration while reducing sampled route
retention. This attempt diagnoses the update mechanism before choosing another
training intervention. It asks whether TD, demonstration-margin and actual-return
gradients conflict, and whether current gradients, inherited Adam moments or
clipping reduce the action gaps at fragile decisions.

This is a **retrospective diagnostic** of existing trajectories. It does not
extend either trained policy, tune a coefficient, select a new best checkpoint,
or claim better success rates. All code, comments and diagnostic messages are
English. Production training code and earlier evidence remain unchanged.

## Fixed inputs and case selection

Use the exact attempt-016 baseline and attempt-017 MC-anchor trajectories,
identified through their committed archives and model/log hashes. Both use the
same 117 original demonstration transitions, learning rate 0.0001, margin 0.8,
gamma 0.99, total-gradient clipping at 10, and supervision coefficients of one.
The baseline has no MC loss; the anchor has MC coefficient one. Their states
after the first new update differ, so comparisons between arms are conditional
on each arm's own trajectory, not matched-state treatment estimates.

For both arms, inspect the following **incoming** TD updates:

`2001, 2008, 2009, 2010, 2018, 2152, 2158, 2161, 2200, 2201, 3000, 4500, 8500, 10000`.

The pre-state for incoming update `t` is the complete state after `t - 1`.
These cases cover the shared start, early failure/recovery, first step-6 reversal,
reported dense minima, target synchronization and its following update, later
failed checkpoints, and the endpoint. Selection uses already observed results;
these are not random samples or evidence of conflict frequency over all training.

Restore the nearest preceding complete snapshot, including online/target
networks, Adam moments and step counters, absolute TD clock, and both recorded
RNG states. Share overlapping historical prefixes. Every replayed update must
exactly reproduce its original loss log and seven-state Q record; encountered
saved full states must also match. Each inspected actual update must exactly
match the original updater's online, target and optimizer state.

The eight demonstration probes are scene 10001 steps 6, 18, 20, 21, 22, 23 and
24, plus scene 10002 step 37. These are reference inputs, not extra rollouts.

An additional probe reconstructs the **actual incoming step-40 state** in
scene 10002. The saved TD-7500, TD-8500 and TD-9000 MC policies have identical
logged actions and positions through step 39, but different choices at step 40.
Replay that common prefix once: one reset, 39 environment actions, no completed
episode. Verify all three policies against their recorded actions and positions
on the shared full observation, and record newly inferred Q values. Historical
rollout logs did not store these Q vectors. Use `south_small`, selected by the
two successful policies, as the reference action; this is **not** a new
demonstration label or an assertion of optimality. Compare the encoded state
with the original demonstration state rather than assuming equal positions
mean equal observations. No forced-action rescue is performed.

For this prefix reconstruction only, disable `verify_reference_route` in a local
copy of the environment configuration. This flag otherwise executes additional
physics on a copied environment during reset. Record the sole flag difference
and require the realized initial encoded observation to match the original
scene-10002 demonstration input byte-for-byte before replay. Retain all physical
parameters and fail if the accepted scene differs; original configurations and
training states remain unchanged.

## Gradient measurements

Compute all losses on the unchanged full batch of 117 transitions. Keep TD
targets frozen from the pre-update target network and MC targets fixed to the
original episodewise actual discounted returns. Retain the original computational
graph for the actual combined update: one forward for TD/chosen Q and a separate
forward for the supervised margin. Compute MC gradients even for the baseline,
but mark them inactive in its actual objective.

Record each component's gradient norm and pairwise dot products/cosines globally
and for each named parameter tensor. A negative cosine indicates opposed local
loss-descent directions; it does not by itself explain a particular action
ranking. Record numerical differences between independently summed component
gradients and the original combined backward pass.

For each probe, fix the highest-Q non-reference action **before** the update.
Differentiate the fixed-action gap `Q(reference) - Q(competitor)` and project
each component's negative gradient onto it. Also measure the actual post-update
gap against the strongest post-update competitor, which may differ. Keep scalar
Q-gap subtraction in float64 and network computation in its original float32.

## Adam decomposition and isolated one-step ablations

The actual optimizer clips the combined gradient once. It does not clip each
component separately. Decompose its parameter displacement into the inherited
first-moment contribution and current TD, margin and MC contributions, all using
the **same actual new second-moment denominator**, clip coefficient and bias
correction. Retain numerical remainder terms and compare the mathematical
displacement with the actual stored parameter difference. Project these vectors
onto each fixed-action gap gradient and report the finite-change residual.

This conditional decomposition is additive under a common denominator; separate
Adam steps for individual losses are not generally additive. ReLU/competitor
changes and floating-point rounding can also separate a local linear prediction
from the observed gap change. The formulas follow the installed PyTorch Adam
implementation and its [official documentation](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html).

Run these independent one-step branches from each unchanged pre-state:

| Branch | Change relative to that arm's actual update |
| --- | --- |
| actual | Original objective, Adam state and clipping |
| no_td | Omit TD only |
| no_margin | Omit supervised margin only |
| no_mc | Omit MC only; aliases actual for the baseline |
| td_only | TD gradient with inherited Adam state |
| margin_only | Supervised-margin gradient with inherited Adam state |
| mc_only | MC coefficient one with inherited Adam state, including baseline diagnostics |
| zero_current_gradient | Set every current gradient to zero; retain inherited Adam state |
| zero_first_moment | Zero inherited first moments only; retain second moments and step counters |
| reset_adam | Reset all Adam state; retain model, target and learning rate |
| no_clipping | Retain actual objective and Adam state, omit clipping only |

Each branch gets an independent model/optimizer copy. Never chain ablations,
feed them back into historical replay, or count them as new policy training.
Loss-removal branches also change their own clip scale and new second moments;
their outcomes are finite one-step counterfactuals, not pure additive attribution.
Zero gradients are explicit tensors, not `None`, so Adam still updates moments
and counters. A reset changes moments and bias correction together.

## Accounting, verification and execution

The prescribed shared historical replay performs 2790 original optimizer steps
across both arms: 2762 intermediate steps and 28 inspected actual-update checks.
The diagnostic clones add 294 one-step updates (10 per baseline case, 11 per
MC case). The only environment work is the common 39-action trajectory prefix.
Independent verification is counted separately. No policy continuation,
demonstration collection or held-out evaluation is introduced.

Meaningful tests cover component gradients, Adam decomposition, clipping,
source/optimizer/RNG immutability, independent ablations, changing competitors,
trajectory-prefix verification, restoration and accounting. An independent
verifier audits all case files and numerically reconstructs both arms at incoming
updates 2008, 2201 and 8500, including the actual, first-moment-zeroed and
fully-reset Adam branches. Its budget is 198 intermediate historical updates
plus 18 diagnostic steps, with no environment actions.

Run from the repository root:

```powershell
.venv/Scripts/python.exe -u docs/experiments/018_gradient_diagnostics/run_gradient_diagnostics.py --reference-run results/training/mc_anchor_20261113 --output-dir results/training/gradient_diagnostics_20261114
.venv/Scripts/python.exe -u docs/experiments/018_gradient_diagnostics/verify_gradient_diagnostics.py --run-dir results/training/gradient_diagnostics_20261114 --output results/training/gradient_diagnostics_20261114/verification.json
```

Preserve every selected case and every ablation, source hashes, exact replay
checks, the reconstructed probe, all accounting and any failed/interrupted
attempt. Do not retain only explanatory-looking cases. Record conclusions
separately from proposed remedies: a one-step gap improvement does not establish
long-term route retention or complete-episode success.

Before execution, 38 focused core/trajectory tests passed in 4.85 seconds and
Ruff passed for the diagnostic runner, core, prefix reconstruction, archive
tool and their completed tests. A read-only preflight checked 247 source hashes,
the 117-transition batch and the 1395-update replay schedule per arm; it performed
no optimizer updates or environment actions. Independent verification and
analysis tools are developed separately and must pass before results are accepted.

The pre-execution protocol and diagnostic implementation were committed as
`ce8658136b612fd4d7e442993e1aa4e071a8f381`. The completed results follow.

## Completed run and accounting

The main diagnostic ran on 14 November 2026, 00:50:32--00:51:03 in
Asia/Shanghai (13 November, 16:50:32.576538--16:51:03.706112 UTC), taking
31.13 seconds. All 28 cases completed. Every replayed original update matched
its historical log and seven-state Q record exactly; available full snapshots
and all 28 diagnostic actual-update states also matched. All 247 tracked source
hashes remained unchanged. The recorded launch commit is the protocol commit
above; provenance also retains the dirty-worktree flag and actual source hashes.

The run executed **2762 intermediate historical updates, 28 original-update
checks and 294 independent diagnostic steps**. These are replay and diagnosis,
with zero new policy-training updates. The sole physical reconstruction used
one reset and 39 actions, stopped before action 40 and completed no episode.
All eight failure outcomes discussed below are reused historical evaluations.
No demonstration was collected and no held-out scene was accessed.

## Loss geometry and fragile decisions

Both value-fitting objectives often oppose the action-margin objective in the
selected states. This pattern is visible in the gradient measurements, but a
global cosine does not specify which local action ranking will change.

| Global gradient pair | Baseline: negative / 14 | MC anchor: negative / 14 | Baseline cosine range | MC-anchor cosine range |
| --- | ---: | ---: | ---: | ---: |
| TD and margin | 14 | 11 | -0.9270 to -0.5739 | -0.8688 to 0.1367 |
| MC and margin | 14 | 11 | -0.5048 to -0.0419 | -0.6234 to 0.6567 |
| TD and MC | 0 | 0 | 0.3008 to 0.6701 | 0.0372 to 0.9453 |

MC gradients in the baseline are measured but inactive. These counts describe
28 retrospectively selected updates, not the prevalence of conflict throughout
training. In the MC arm, the output layer's TD--margin cosine is negative in
all 14 cases, including the three with a positive global cosine. Aggregation
across layers can therefore conceal a local disagreement.

Even the full-batch supervised gradient can reduce an individual probe's gap:
its raw negative-gradient projection is negative in 16/126 baseline and
19/126 MC probe-update pairs. Supervision of the other demonstrated states
shares the same parameters; the aggregate objective need not improve every
example simultaneously. At MC update 3000, this occurs at scene 10001 step 6.

The following table retains all eight selected historical failure contexts.
Gaps compare the reference action with the strongest other action. Each row
compares the state immediately before and after the indicated incoming update.
Scene 10001 step 21 uses `north_small`; step 6 uses `south_small`. The scene
10002 step-40 reference is described separately below.

| Arm | Incoming TD update | Probe | Gap before | Actual gap after |
| --- | ---: | --- | ---: | ---: |
| Baseline | 2009 | 10001, step 21 | 0.175179 | -0.016716 |
| Baseline | 10000 | 10001, step 21 | 0.059540 | -0.059891 |
| MC anchor | 2008 | 10001, step 21 | 0.122620 | -0.102173 |
| MC anchor | 2009 | 10001, step 21 | -0.102173 | -0.073845 |
| MC anchor | 2201 | 10001, step 6 | 0.154503 | -0.053696 |
| MC anchor | 3000 | 10001, step 6 | 0.107628 | -0.018959 |
| MC anchor | 4500 | 10001, step 6 | 0.079430 | -0.002144 |
| MC anchor | 8500 | 10002, actual step 40 | 0.047981 | -0.003868 |

MC update 2009 improves an already-negative gap without restoring the reference
action. Thus a failed checkpoint need not be an update that worsens that gap.
The remaining seven rows cross from positive to negative at the selected probe.
No new rollout was run to test these one-step diagnostic policies.

## Retained Adam moments and finite changes

The sign predicted by the raw combined negative gradient disagrees with the
observed fixed-gap change in **24/126 pairs in each arm**. Adam's retained
moments and coordinatewise scaling matter; raw gradient direction alone is
insufficient. For baseline TD 10000, the raw projection predicts an increase,
but the gap falls by 0.119431. Under the actual common denominator, inherited
first moments contribute -0.145689, current TD -0.064856, and current margin
+0.091121. The old moment outweighs the positive current net contribution.

At MC TD 2008, the corresponding step-21 projections are -0.129986 from old
moments, -0.125322 from TD, +0.064424 from margin and -0.033884 from MC.
At MC TD 2201, the step-6 values are -0.046468, -0.072925, +0.021165 and
-0.098285. These updates combine adverse inherited and current contributions.
By contrast, old moments contribute **+0.008212 at MC TD 3000** and
**+0.000871 at TD 4500**. Current TD and MC contributions reduce the gap in
both cases. Retained momentum is not uniformly harmful.

The largest mathematical-versus-stored parameter-displacement residual norm is
8.667e-7, with maximum coordinate residual 2.351e-7. First-moment rounding
remainders and final storage/arithmetic residuals are retained separately.
Nonlinear gap residuals are also measured: -0.011687 at MC TD 2201 step 6 and
-0.018757 at TD 3000 step 6. The additive projections therefore explain a
local linearized change, rather than exactly reconstructing every finite Q gap.

## What the one-step ablations establish

Across the eight failure contexts, zeroing only the inherited first moment
produces a positive local gap in 6/8 cases; omitting TD does so in 5/8, and a
full Adam reset in 2/8. These counts are local ranking outcomes, not recovered
episodes. Removing MC alone does not resolve the early MC step-21 failures:
its gaps remain -0.061646 at TD 2008 and -0.087967 at TD 2009. It does produce
positive gaps at the MC TD 2201, 4500 and 8500 failure probes, but not TD 3000.

Broader checks reveal the tradeoffs. The table counts reference-action changes
relative to each case's actual update over **126 probe-update pairs per arm**.
Recoveries and regressions use the greedy action, not just a larger scalar gap.

| Independent branch | Baseline recoveries / regressions | MC-anchor recoveries / regressions |
| --- | ---: | ---: |
| No TD | 5 / 1 | 8 / 0 |
| No margin | 1 / 22 | 1 / 19 |
| No MC | 0 / 0 (alias) | 6 / 0 |
| Zero current gradient | 5 / 8 | 7 / 9 |
| Zero inherited first moment | 5 / 4 | 10 / 5 |
| Reset all Adam state | 2 / 69 | 9 / 51 |
| No clipping | 1 / 2 | 9 / 15 |

None is a universal remedy. Even omitting TD leaves three baseline and ten MC
probe-update pairs with negative gaps. At MC TD 2009 and 3000, it makes the
failure probe's gap worse than the actual update, despite a negative TD term
in the common-denominator decomposition. Independent branches recompute their
own clipping and second moments, so they are not sums of those conditional
attributions. A full reset can improve one fragile state while damaging many
other reference decisions. Removing margin causes widespread local regressions.

## Scene 10002: the later fork is a distinct diagnostic

The reconstruction reproduced all logged actions and positions through step 39
for the unchanged MC policies at TD 7500, 8500 and 9000. Its complete encoded
incoming step-40 observation is **exactly equal to original demonstration
transition index 115 (zero-based)**, with zero differing features. At this shared
input, the newly measured
`south_small - east_large` gaps are +0.046295, -0.003868 and +0.116600,
respectively. The two historically successful policies choose `south_small`;
the failed TD-8500 policy chooses `east_large`.
The metadata field `fixed_comparison_action` identifies this historical fork;
each derivative probe still selects its own strongest pre-update alternative.

During incoming update 8500, this gap changes from +0.047981 to -0.003868.
Its common-denominator projections are -0.026865 from old moments, -0.113283
from TD, +0.140578 from margin and -0.052277 from MC. The step-37 deviation
also occurs in successful historical episodes, so its status as the first
reference divergence does not establish it as the cause of failure. The later
shared-state fork is informative but remains a diagnostic association: no
forced-action rescue or complete counterfactual rollout was performed.

## Verification, retained evidence and next hypothesis

The full repository suite, run before the verifier correction, passed
**743 tests in 78.79 seconds**.
After the verifier correction below, **37 focused tests passed in 5.42 seconds**,
including two new rounding-bound regressions. Ruff passed across the repository.
The first verifier attempt stopped at a structural residual threshold before any optimizer step:
its fixed 2e-7 coordinate limit was smaller than half a float32 ULP for an
observed parameter near 4.735. The observed residual, 2.35087e-7, is below that
half-ULP scale of 2.38418579e-7. That validation attempt performed three source
probe forwards and no environment actions. Its source, console output and
failure record are preserved in `verification_attempts/`. The successful retry
uses a coordinatewise half-ULP storage bound plus Adam arithmetic and float64
reconstruction allowances. The diagnostic core and main results are unchanged.

The independent verifier passed: all **28 records** were structurally audited,
and **six cases** were reconstructed numerically, at incoming TD 2008, 2201 and
8500 in each arm. It independently checked component gradients, gap derivatives
and Adam decomposition, and exactly reproduced actual, reset-Adam and
zero-first-moment steps. Verification added **198 historical replay updates
and 18 independent steps**, zero environment work, and three source-policy
forwards. It checked stored trajectory evidence and the final-state Q vectors;
it did not independently replay prefix physics or all prefix Q computations.
All source hashes remained unchanged.

The read-only analysis audit matched all 28 cases, 252 probes, 2772 resolved
branch-probe records and every consumed input hash. It added no optimizer or
environment work. Complete per-case measurements and every ablation are retained
with the [analysis](018_gradient_diagnostics/evidence/analysis.json),
[main summary](018_gradient_diagnostics/evidence/summary.json),
[independent verification](018_gradient_diagnostics/evidence/verification.json),
[trajectory reconstruction](018_gradient_diagnostics/evidence/trajectory_probe.json)
and [comparison figure](018_gradient_diagnostics/evidence/gradient_contexts.png).
The figure passed visual inspection. The archive contains **45 files including
its manifest, totaling 7837176 bytes**; all hashes, byte counts and source copies
passed validation. It retains the failed verification record, source and console
alongside the successful retry. Main execution and verification together used
**3300 optimizer steps**, all historical replay or isolated diagnosis, with zero
new policy-training updates and only the recorded one-reset, 39-action prefix.
Full local run files are under `results/training/gradient_diagnostics_20261114`.

One possible next controlled experiment is to reduce only the TD objective
coefficient from 1 to 0.5, using the same complete baseline TD-2000 snapshot,
margin coefficient 1, no MC term, retained Adam, learning rate 0.0001 and
unchanged data and target schedule. This would test relative loss balance,
with all three scene outcomes and neighboring gaps checked through TD 10000.
It has **not been performed**. The current evidence supports testing that
hypothesis, not adopting it, resetting Adam by default, or claiming generalization.
