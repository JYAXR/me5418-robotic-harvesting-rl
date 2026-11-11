# Attempt 013: lower TD learning rate with calibrated values and preserved Adam

## Protocol recorded before execution

Attempt 012 traced five original failures to consequential turn-action ranking
reversals. Doubling TD demonstration weight improved some early checkpoints but
introduced a simultaneous three-scene failure at update 50. This attempt keeps
**demonstration weight 1** and **preserves calibration Adam state**, changing only
the TD learning rate from **0.0003 to 0.0001**. The threefold reduction is fixed
before execution; no additional rate is selected after observing these results.

Both arms restore the identical, hash-verified attempt 011 calibration-end full
training snapshot. It follows 20000 supervised updates and 20000 actual-return
calibration updates on three known training scenes. Those historical updates are
reused once, not rerun or counted as new work. The same 117 ordered demonstration
transitions, episode boundaries and finite-horizon discounted returns are reused:
35 transitions from seed 10000, 41 from 10001 and 41 from 10002. No new collection,
off-route recovery examples, online exploration or held-out scenes are introduced.

| Setting | Baseline | Lower-rate treatment |
| --- | --- | --- |
| TD learning rate | 0.0003 | 0.0001 |
| Demonstration weight / margin | 1 / 0.8 | 1 / 0.8 |
| Adam moments and internal parameter steps | Preserved; step 20000 at boundary | Identical preserved state |
| TD phase updates | 2000 | 2000 |
| Effective batch | All 117 transitions | Same ordered 117 transitions |
| Target synchronization | Every 200 TD updates | Same schedule |
| Gradient clipping norm / gamma | 10 / 0.99 | 10 / 0.99 |

The architecture remains 341/256/256/8, and the environment, reward, task budget,
encoder and demonstration data remain unchanged. Both arms copy online weights
to frozen targets, clear gradients and restart only the TD phase clock at zero.
Changing the treatment learning rate updates the learner configuration, Adam
default and parameter-group rate together; it must not reset or alter any Adam
moment tensor or internal step. The initial policies and all Q values must match.
The objective remains `TD_Huber + demonstration_margin_loss` in both arms, with
an equally weighted mean over transitions rather than over scenes.

Use the already tested, immutable generic control from attempt 012 through a
strict attempt 013 entry point. Record both script hashes and the resolved
attempt-specific protocol. First rerun the baseline and require **exact equality
of all 2000 update records, all 18 observations and the final complete training
state** with attempt 011's preserved-Adam arm. Stop and retain a mismatch if this
gate fails; do not run treatment against an inconsistent baseline. The learning
rate intervention is applied only after independently restoring the same
calibrated state for treatment.

Evaluate all three scenes at **0 through 10, then 20, 50, 100, 200, 500, 1000
and 2000** TD updates. This produces 18 checkpoints per arm and 108 scheduled
episodes. Every rollout uses the actual greedy neural policy; no corrective
action, planner fallback or weight rollback is allowed. Keep every failure and
recovery. Run both fixed budgets without early stopping or choosing a successful
intermediate model as the reported final model.

The primary criterion is whether **all three acquired scenes remain successful
at every scheduled checkpoint**. Also report first-ten all-scene successes,
all-schedule all-scene successes, individual scene counts, each seed's failed
updates and final performance. These different denominators must remain explicit;
the previous experiment showed that they can move in opposite directions.
Sampled checkpoint fractions are not independent test success rates.

Analyze the signed Q gap `Q(demonstrated) - max(Q(other))` using batch-of-one
inference, matching greedy action selection. Predeclare seed 10001 reference
steps **6, 18, 22 and 24**: the original turn failures, attempt 012's later
step-size disagreement, and a successful-deviation control. Also inspect the
first actual disagreement in every scene at every checkpoint, including new
failures. A disagreement is not automatically a causal error; no new one-action
rescue experiment is included here. Scores on fixed reference states beyond an
actual trajectory's common prefix must be labeled accordingly.

Record unweighted TD and action-margin losses, pre-clipping gradient norms,
actual parameter changes, optimizer state and fixed demonstrated-return errors.
Because the first gradients and Adam moments are matched, compare the measured
first parameter change with the expected learning-rate scaling. Later gradients,
moments and target values may diverge, so subsequent parameter changes are not
assumed to remain in a fixed ratio. Verify nonzero total parameter displacement,
all 2000 actual updates and ten target synchronizations to distinguish retention
during learning from freezing the model.

## Accounting, verification and scope

Main-run budget: **4000 new TD updates**, the separately reported **40000 shared
historical updates**, and **108 scheduled evaluation episodes** with actual
actions counted. No new demonstrations, pretraining, calibration, online training
or held-out evaluation are performed. Reset feasibility and planner probes are
excluded from policy-action counts, consistently with earlier attempts.

After completion, independently replay the first ten TD updates of each arm and
compare exact metrics, online/target tensors and Adam state. This adds **20
verification updates**. Reload initial/update-1/final models from both arms on
all three scenes (**18 verification episodes**), recording their action cost
separately. Check every saved inference and full training snapshot. At checkpoint
500, the stored target is integrity-checked; the unsaved update-400 origin is not
independently reconstructed unless additional verification is explicitly recorded.
The strict verification wrapper writes `reload_checks_base.json` and
`reload_checks.json` for the same replay; their update and episode counts are
reported once, not added together.

Preserve raw runs locally under ignored results, with all observations, concise
update samples, hashes, failures and this record committed to Git. Code, comments
and runtime messages stay in English. Production defaults are not changed by
this diagnostic. Reports and submission archives are outside this task.

This is a retention test on three known training scenes from one initialization,
not an unseen-layout or online RL evaluation. Equal update budgets do not imply
equal adaptation progress: lowering the rate slows movement and may postpone
failure beyond the observed horizon. Even perfect sampled retention cannot prove
success at unsampled updates or indefinitely. Attempt 008's lower-rate failures
remain relevant history, but used uncalibrated single-scene weights and fresh
Adam; it did not test this calibrated, Adam-preserved multi-scene setting.

## Reproduction

```powershell
python docs/experiments/013_td_learning_rate/run_learning_rate_control.py --reference-run results/training/multiscene_retention_20261112 --archive-dir docs/experiments/011_multiscene_retention/evidence --output-dir results/training/td_learning_rate_preserved_20261112
python docs/experiments/013_td_learning_rate/verify_learning_rate_control.py --run-dir results/training/td_learning_rate_preserved_20261112 --output results/training/td_learning_rate_preserved_20261112/reload_checks.json
python docs/experiments/013_td_learning_rate/analyze_learning_rate_q.py --run-dir results/training/td_learning_rate_preserved_20261112 --output results/training/td_learning_rate_preserved_20261112/q_analysis.json
```

## Outcomes

Executed from the predeclared and pushed commit `28d26e0` on **12 November 2026,
01:10:45--01:38:11 Asia/Shanghai** (11 November 17:10:45--17:38:11 UTC).
Both arms completed all 2000 updates. The baseline passed exact comparison with
attempt 011 for every update record, all 18 observations and the final full
training snapshot before treatment began. No run was replaced or stopped early.

| Measurement | Baseline: 0.0003 | Treatment: 0.0001 |
| --- | --- | --- |
| All-three-scene successes during updates 1--10 | 6/10 checks | 10/10 checks |
| Individual scene successes during updates 1--10 | 26/30 | 30/30 |
| All-three-scene successes across all scheduled checks, including update 0 | 13/18 | 18/18 |
| Individual scene successes across all scheduled checks | 49/54 | 54/54 |
| Seed 10000 failed updates | None | None |
| Seed 10001 failed updates | 1, 6, 9, 10, 200 | None |
| Seed 10002 failed updates | None | None |
| Scenes retained at every scheduled checkpoint | 10000 and 10002 | All three |
| Final demonstrated action agreement | 117/117 | 116/117 |
| Final actual route lengths | 35/41/41 | 35/41/41 |
| Final demonstrated-return RMSE | 0.563084 | 0.461398 |
| Strict all-scene sampled retention | False | True |

The lower-rate policy completed every scheduled episode without damage or
budget exhaustion. This includes the baseline's five failed checkpoints and
update 50, where the weight-2 treatment in attempt 012 lost all three scenes.
Only the paired baseline/treatment comparison here isolates learning rate; the
cross-attempt observation is contextual, not an additional matched comparison.

The final lower-rate model differs from the demonstration only at seed 10001's
step **24**, choosing `east_large` rather than `east_small`, and still completes
the route in 41 steps. Its 116/117 action agreement therefore accompanies better
sampled task retention than the baseline's final 117/117. Action agreement alone
does not replace closed-loop task evaluation. Both final RMSE values are measured
against the same demonstrated returns; these targets are not a known optimal
value function, and both errors remain above the shared calibration-end 0.393086.

### Q rankings at the previously fragile turns

The following signed gaps use the same demonstrated states and batch-of-one
inference. Positive values mean the demonstrated action outranks every other
action. All five originally failing choices have positive gaps in treatment:

| TD update / seed 10001 reference step | Baseline signed gap | Lower-rate signed gap |
| --- | --- | --- |
| 1 / 6 | -0.016968 | +0.181679 |
| 6 / 6 | -0.103065 | +0.308067 |
| 9 / 18 | -0.343254 | +0.628349 |
| 10 / 18 | -0.113281 | +0.523529 |
| 200 / 6 | -0.036690 | +0.068130 |

Across all 18 treatment checkpoints, the minimum reference-state gaps were
**0.001541 at step 6** (TD update 8), **0.349113 at step 18** (update 500) and
**0.200668 at step 22** (update 4). The critical turn rankings never reversed at
the observed checkpoints, but the very small step-6 gap does not demonstrate
robustness to perturbations or a uniformly satisfied 0.8 training margin.
At the final checkpoint, gaps at steps 6/18/22 were 0.209717/0.422127/0.435524.
The step-24 gap remained slightly negative (-0.031105), consistent with the
successful step-size disagreement. All fixed-state scores and actual first
disagreements are retained, including harmless disagreements. No new
counterfactual correction was applied to any rollout.

### Actual updates and loss behavior

The initial objective, TD component, margin component and pre-clipping gradient
norm were exactly equal across arms. Both first gradients had norm **28.194744**.
The first parameter-change norm was **0.00452334** at 0.0003 and **0.00150778**
at 0.0001: a measured ratio of **0.3333338**, matching the intended threefold
reduction up to float32 rounding. Directly comparing saved parameter-displacement
vectors gave a maximum absolute proportionality residual of 0.0000000795; later
updates are not assumed to retain this ratio.

With identical initial targets and objective weights, the baseline's first
total loss rose from **0.176720 to 0.179635**, while the lower-rate loss fell to
**0.164435**. The baseline reduced its TD component further (to 0.092785 versus
0.101252), but increased the unweighted action-margin component more (to 0.086850
versus 0.063183). These measured changes are consistent with a smaller first
step preserving learned action ordering while still fitting TD targets.

Both arms performed **2000 nonzero parameter updates**, ten target
synchronizations and ended at Adam internal step 22000. Final parameter-vector
L2 displacements from the shared calibrated model were **1.346784** in the
baseline and **0.786078** in treatment. The successful treatment was not frozen
or replaced with the initial model.

Across the phase, mean pre-clipping gradient norms were **37.3744 and 15.8636**;
norms exceeded the unchanged threshold of 10 on **1996/2000 and 1684/2000**
updates. Mean parameter-change norms were 0.00562691 and 0.00202524. Mean
post-update unweighted margin losses were 0.0867137 and 0.0654937; corresponding
TD losses were 0.1082392 and 0.0818867. Later target networks differ between arms,
so these phase averages are descriptive rather than identical-target loss
comparisons. No change in clipping threshold, supervision weight or optimizer
state was introduced alongside the learning-rate intervention.

## Conclusion and limits

**Reducing only TD learning rate to 0.0001 preserved all three acquired routes
at every scheduled checkpoint through 2000 TD updates**, with supervision weight
1 and calibrated Adam state retained. The matched 0.0003 baseline reproduced its
five seed-10001 failures. This is positive evidence for the reduced-rate setting
within the tested pipeline and horizon, with real continuing parameter updates.

It is not proof of indefinite stability, success at every unsampled update,
convergence or generalization. The data remain three privileged training routes
from one initialization; off-route recovery and new layouts remain untested.
The narrow minimum step-6 gap further limits robustness claims. Unlike attempt
008, this run starts from return-calibrated multi-scene weights and retains Adam,
so it does not negate that earlier lower-rate failure. Production defaults remain
unchanged; the tested setting and final local checkpoint are available for a
separately declared longer-horizon or broader-scene experiment.

## Verification and retained evidence

The full suite passed **418 tests**, including 44 new focused checks for strict
protocol enforcement, the unchanged Adam boundary and read-only analysis.
Ruff passed. Code, comments and runtime messages are in English. The training
loop from attempt 012 and all 52 production source files remained unchanged.

Main-run work was **4000 new TD updates and 108 evaluation episodes / 5010
actions**: 2904 baseline actions and 2106 treatment actions. The shared 40000
historical supervision/calibration updates were reused and counted separately.
No new demonstration, online exploration or held-out action was collected.
Reset feasibility/planner probes are excluded consistently from action counts.

Read-only Q analysis verified all **74 saved model files**, identical initial
Q vectors on all 117 demonstrated states, source/input hashes, each stored
common-prefix action, the first displacement scaling and both final parameter
displacements. It performed zero optimizer updates, environment resets or
actions. The figure showing all scheduled outcomes was visually inspected.

Independent verification passed. It checked 52 source files, 56 reference input
files, the strict learning-rate-only boundary, all saved models and all 4000
update records. The first ten updates of each arm exactly reproduced logged
metrics, online/target tensors and Adam state. Initial/update-1/final policies
from both arms exactly reproduced metrics and trajectories on all three scenes,
including the baseline's update-1 timeout. This added **20 verification updates,
18 episodes and 861 actions**, counted once across the base and wrapper reports.
The extra boundary audit performed zero updates, resets and environment actions.
The previously stated limit on independently reconstructing the update-500
target remains applicable.

All scheduled outcomes and failures, first-divergence records, fixed-state Q
comparisons and first-ten/every-50th/last update samples are retained. Full
checkpoints and raw logs stay in the ignored run directory. The final treatment
inference model is `results/training/td_learning_rate_preserved_20261112/treatment/td/update_2000.pt`;
the matching complete state is `treatment/td/training_state.pt` within that run.
Continuing with Adam state requires the complete training snapshot; inference-only
loading does not restore optimizer history. Earlier experimental evidence and
the user's report files were not modified.

```powershell
python docs/experiments/013_td_learning_rate/plot_learning_rate_control.py --run-dir results/training/td_learning_rate_preserved_20261112 --output results/training/td_learning_rate_preserved_20261112/learning_rate_outcomes.png
python docs/experiments/013_td_learning_rate/archive_learning_rate_control.py --run-dir results/training/td_learning_rate_preserved_20261112 --destination docs/experiments/013_td_learning_rate/evidence
```

Evidence: [all scheduled outcomes](013_td_learning_rate/evidence/learning_rate_outcomes.png),
[paired summary](013_td_learning_rate/evidence/summary.json),
[Q rankings and parameter changes](013_td_learning_rate/evidence/q_analysis.json),
[exact baseline reproduction](013_td_learning_rate/evidence/baseline_equivalence.json),
[independent verification](013_td_learning_rate/evidence/reload_checks.json),
and [archive manifest](013_td_learning_rate/evidence/archive_manifest.json).
