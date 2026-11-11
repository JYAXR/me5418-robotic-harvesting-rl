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
