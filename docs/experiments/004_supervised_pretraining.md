# Attempt 004: Supervised action margins during demonstration pretraining

**Status:** protocol prepared; results pending execution.

**Run identifier:** `multiroute_supervised_20261110`.

## Question and single training change

Attempt 003 added 2000 TD-only demonstration updates, but all greedy validation
checkpoints still had zero successes. This attempt adds a supervised action
margin to those same 2000 pretraining updates. It asks whether explicitly
preferring the demonstrated action improves the policy before and after online
interaction. The environment, reward, architecture, demonstration seeds,
pretraining sampler and online schedule remain unchanged.

For a demonstration action `a_demo`, the supervised objective is
`mean(max_a[Q(s,a) + margin(a_demo,a)] - Q(s,a_demo))`, where the margin is zero
for the demonstrated action and 0.8 for each other action. The combined
pretraining objective is `TD_Huber_loss + 1.0 * supervised_margin_loss`.
Weight 1.0 and margin 0.8 are fixed before the trial and are not claimed to be
optimal. The existing Adam optimizer, gradient clipping and target-network
synchronization schedule are retained.
The margin is a soft training penalty and does not guarantee the learned
Q values will satisfy that separation on every observation.

The large-margin objective follows the idea in Hester et al.,
[Deep Q-learning from Demonstrations (AAAI 2018)](https://arxiv.org/abs/1704.03732).
The objective is implemented in the project's existing PyTorch learner; no
external DQfD implementation is imported. This is a limited component experiment,
not the full DQfD algorithm: the code retains ordinary one-step DQN, uniform
replay, and no added n-step loss, prioritized replay or L2 regularization term.

Only demonstration pretraining uses the supervised loss. Online updates remain
ordinary DQN, including updates whose uniformly sampled batch happens to contain
demonstrations. Random exploratory actions are never used as expert labels.
If the post-pretraining policy improves but subsequently deteriorates, retain
that evidence rather than selecting only the favorable stage.

## Protocol fixed before the run

| Setting | Value |
| --- | --- |
| Environment | `configs/environment_multiroute.yaml` |
| Network | `configs/dqn_multiroute_pilot.yaml` |
| Schedule | `configs/training_multiroute_supervised.yaml` |
| Demonstration scenes | Seeds 10000--10009; expected 421 transitions |
| Pretraining | 2000 updates; batch size 64 |
| Supervised weight and margin | 1.0 and 0.8; pretraining only |
| Pretraining sampler | Independent SeedSequence `[0, 1]`, matching attempt 003 |
| Online training | 30 episodes; seeds 0--29; initialization/exploration seed 0 |
| Development validation | Seeds 100000--100002; greedy actions |
| Validation stages | Initial; pretrained; online episodes 10, 20 and 30 |

Regenerate demonstrations through actual environment transitions, compare their
records with attempt 003, and verify identical initial validation. The online
warm-up remains 256 actions, with one update every four actions and epsilon
decaying from 1.0 to 0.1 over 4000 actions. Pretraining does not advance either
online counter or consume the exploration/online replay RNG streams. Optimizer
and target-network states carry forward into online training.

This trial matches attempt 003's planned pretraining and online episode budgets.
It adds computation for the supervised term, and actual online action/update
counts may differ because policies terminate at different times. Report both
stages and the additional demonstration collection cost explicitly. Collection
actions exclude planner search and reset feasibility probes.

`pretrained.pt` records the policy after supervised pretraining; `best.pt`
continues to select only among online validation checkpoints, with success rate
then mean return determining the score. `latest.pt` is the final online policy.
All are inference checkpoints. These development validation scenes are not a
final held-out test set, and no final test is used in this attempt.

## Commands

Run from the repository root using the declared Python environment and a new
output directory. The actual local interpreter is `.venv/Scripts/python.exe`.

```powershell
python -u scripts/train_dqn.py --config configs/environment_multiroute.yaml --dqn-config configs/dqn_multiroute_pilot.yaml --training-config configs/training_multiroute_supervised.yaml --output-dir results/training/multiroute_supervised_20261110
python scripts/summarize_training.py --run-dir results/training/multiroute_supervised_20261110
python scripts/evaluate_dqn.py --checkpoint results/training/multiroute_supervised_20261110/pretrained.pt --episodes 3 --seed 100000
python scripts/evaluate_dqn.py --checkpoint results/training/multiroute_supervised_20261110/best.pt --episodes 3 --seed 100000
python scripts/archive_training_attempt.py --run-dir results/training/multiroute_supervised_20261110 --output-dir docs/experiments/004_supervised_pretraining/evidence
```

## Metrics and interpretation

Record every pretraining update's TD loss, supervised margin loss, combined
loss, gradient norm and pre-update agreement with demonstration actions.
The plot labels any trailing-window smoothing and preserves raw values in
the archived log. First/last 100-batch agreement is a training diagnostic on
reused demonstration states, not held-out accuracy or episode success.

Task success, damage, budget exhaustion and motion diagnostics are evaluated
at all declared stages. A lower supervised loss or higher action agreement
alone does not establish successful rollout behavior. Retain all unsuccessful
episodes and any online deterioration. One initialization and three validation
scenes cannot establish reliability or generalization.

## Results

Pending execution. Compare actual results against attempt 003 without changing
the supervised weight, margin or training budget after observing the outcome.
