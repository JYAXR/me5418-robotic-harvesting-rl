# Attempt 003: DQN pretraining on the same reference data

**Status:** protocol prepared; measured results will be recorded after execution.

**Run identifier:** `multiroute_pretrained_20261110`.

## Question and controlled change

Attempt 002 provided 421 successful reference transitions but did not produce
a successful greedy validation policy. This attempt asks whether additional
training on those demonstrations before online interaction helps the DQN.

The only added training stage is 2000 ordinary DQN TD updates on demonstration
data. The reward, architecture, batch size, target synchronization interval,
online exploration schedule, warm-up and 30-episode budget remain unchanged.
There is no supervised imitation term, prioritized sampling, curriculum or
new demonstration seed. This is a diagnostic TD-pretraining experiment, not
DQfD, and it does not guarantee imitation of the reference actions.

The hypothesis is that successful outcomes need more updates to influence
earlier decisions. An alternative explanation remains possible: maximizing
next-state Q values over actions absent from demonstrations can produce poor
action choices. Training loss alone cannot resolve this question.

## Protocol fixed before the run

| Setting | Value |
| --- | --- |
| Environment | `configs/environment_multiroute.yaml` |
| Network | `configs/dqn_multiroute_pilot.yaml` |
| Training schedule | `configs/training_multiroute_pretrained.yaml` |
| Demonstration scenes | Seeds 10000--10009, matching attempt 002 |
| Pretraining | 2000 updates, batch size 64, existing DQN TD loss |
| Pretraining sampler | Independent NumPy SeedSequence `[0, 1]` |
| Online training | 30 episodes, seeds 0--29, initialization/exploration seed 0 |
| Development validation | Seeds 100000--100002 |
| Validation checkpoints | Initial, after pretraining, after online episodes 10/20/30 |

Reference trajectories are regenerated through the same seeded environment and
physics. Their collection records will be compared against attempt 002 to
verify the same 421-transition data budget. The initial validation will also
be compared before any learning takes place.

Pretraining does not advance online action counters, consume the online replay
sampler or consume the agent's exploration RNG. Its optimizer and target-network
states continue into online training. Total optimizer updates include both
stages; each stage is also reported separately. The existing 256-action online
warm-up still applies. The demonstration data is ordinary uniformly sampled
replay during subsequent online training.

`pretrained.pt` preserves the policy after pretraining. To retain the existing
checkpoint-selection rule, `best.pt` compares only online validation checkpoints
at episodes 10/20/30; `latest.pt` is the last online checkpoint. Report the
pretrained and online results explicitly, including deterioration if observed.
These are inference checkpoints, not exact training-resume snapshots.

This run has the same online episode budget as attempt 002 but 2000 additional
optimizer updates. Actual online action and update counts can differ because
episode lengths depend on the policy. Demonstration collection actions are
reported separately and exclude planner search and reset feasibility probes.
No final test scenes are used or selected based on this trial.

## Reproduction commands

Run from the repository root in the declared Python environment. Use a new
output directory to avoid overwriting a previous attempt.

```powershell
python -u scripts/train_dqn.py --config configs/environment_multiroute.yaml --dqn-config configs/dqn_multiroute_pilot.yaml --training-config configs/training_multiroute_pretrained.yaml --output-dir results/training/multiroute_pretrained_20261110
python scripts/summarize_training.py --run-dir results/training/multiroute_pretrained_20261110
python scripts/evaluate_dqn.py --checkpoint results/training/multiroute_pretrained_20261110/pretrained.pt --episodes 3 --seed 100000
python scripts/evaluate_dqn.py --checkpoint results/training/multiroute_pretrained_20261110/best.pt --episodes 3 --seed 100000
python scripts/archive_training_attempt.py --run-dir results/training/multiroute_pretrained_20261110 --output-dir docs/experiments/003_demonstration_pretraining/evidence
```

The actual run uses `.venv/Scripts/python.exe`. Provenance records retain its
arguments, dependency versions, UTC launch time, Git revision and source hashes.
Pretraining progress is saved every 100 updates and on handled interruption;
completed per-update records are retained in `pretraining_metrics.jsonl`.

## Results

Pending execution. Preserve all checkpoints, failed cases and intermediate
validation results. Judge the effect on greedy task completion and damage,
not merely a decrease in TD loss. One training seed and three development
validation scenes do not establish convergence or generalization.
