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

Results are pending at the pre-execution protocol commit.

Before execution, 38 focused core/trajectory tests passed in 4.85 seconds and
Ruff passed for the diagnostic runner, core, prefix reconstruction, archive
tool and their completed tests. A read-only preflight checked 247 source hashes,
the 117-transition batch and the 1395-update replay schedule per arm; it performed
no optimizer updates or environment actions. Independent verification and
analysis tools are developed separately and must pass before results are accepted.
