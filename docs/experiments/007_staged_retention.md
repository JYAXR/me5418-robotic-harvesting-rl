# Attempt 007: supervised pretraining followed by TD retention checks

## Protocol recorded before execution

Attempt 006 established that supervised fitting can execute the complete fixed
multi-route demonstration, while the jointly trained TD-plus-margin policy still
fails. The next question is whether introducing TD **after** acquiring that
behavior preserves it. First hold the training data fixed to isolate the loss
change from online exploration and new scene exposure.

Reproduce the margin-only control from attempt 006: seed-0 initialization, one
successful training-scene demonstration at seed 10000, 35 transitions, the
existing 341-input/256/256/8-output MLP, encoder version 1, full-batch updates,
Adam learning rate 0.0003, gradient clipping 10 and margin 0.8. Perform **20000
supervised updates**. Checkpoints are 0, 5000, 10000 and 20000. Require exact
agreement with the archived attempt-006 dataset, final weights, whole-dataset
action metrics and successful greedy trajectory before starting the comparison.
If reproduction fails, retain the failure record and stop the TD comparison.

Fork the acquired weights into three branches, each with **2000 additional
updates** and the same frozen demonstration batch:

| Branch | Adaptation objective | Purpose |
| --- | --- | --- |
| `margin_only` | Continue supervised action-margin fitting | Control for additional optimization and the shared phase reset. |
| `td_only` | Production one-step DQN Huber TD loss | Measure retention after removing supervision. |
| `td_margin` | Production TD loss plus the same margin loss, weight 1 | Test whether retaining the supervision protects the behavior. |

At the boundary all branches deliberately receive **fresh Adam optimizers** at
the unchanged learning rate. Copy pretrained online weights to the target
network, freeze the target and reset the phase update clock to zero. This is a
new optimization phase, not an optimizer-state resume. Preserve the 20000
pretraining updates separately; TD synchronization occurs every 200 phase
updates. Apply the same reset to the supervised continuation control, although
that branch does not read the target network. Never bootstrap from the random
target that existed before supervised pretraining.

Evaluate at additional updates **0, 1, 10, 50, 100, 200, 500, 1000 and 2000**.
Before each branch updates, its phase-zero fit and rollout must exactly match
the pretrained baseline. Use greedy neural actions without a planner fallback.
Record success, damage, budget exhaustion, first reference-action divergence,
return, movement and complete action/path traces. Also retain full-dataset
agreement, confusion matrices, TD/margin update losses and current score/target
ranges. Supervised action scores are not calibrated discounted returns.

There is no adaptive early stopping, learning-rate tuning or checkpoint selection
by outcomes. Preserve failures and recovery, not just the final successful
checkpoint. A reported first failure is the first **sampled** failed checkpoint;
states between evaluations are not tested. The total budget is **26000 updates**:
20000 shared supervised updates, 2000 supervised-control updates and 4000 TD
updates across the two TD branches. Reference collection and all neural
evaluation actions are counted separately, excluding planner search and reset
feasibility probes. No held-out/validation scene enters this diagnostic.

## Reproduction

From the project environment, use a new output directory:

```powershell
python docs/experiments/007_staged_retention/run_staged_retention.py --output-dir results/training/staged_retention_20261111
```

The reusable supervised update and explicit phase boundary live in
`src/me5418_harvesting/agents/staged_learning.py`. Existing training defaults and
the earlier diagnostic scripts remain unchanged. All experiment code, comments
and runtime messages are in English. Scripts, resolved settings, source hashes,
data hashes, checkpoints and observed outcomes are retained for reproducibility.

## Interpretation limits

This tests retention on one known training scene with **fixed-data TD updates**.
It does not yet test online replay, exploration, randomized scenes or final-model
generalization. Fixed-route memorization can exploit the remaining-budget input.
Action agreement is useful but not equivalent to task success: different actions
may succeed, and a small number of errors may cause damage. Any conclusion about
the TD objective is conditional on this dataset, optimizer reset, learning rate,
initialization and update budget. Retaining Adam moments is a separate untested
choice. A supervised-only checkpoint is not a fully trained RL submission model.

## Outcomes

Pending the predeclared run. Keep all three branches and every sampled outcome.
