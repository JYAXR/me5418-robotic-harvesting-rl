# Attempt 002: Initialize replay with reference trajectories

**Status:** running. Results will be appended after this run exits.

**Run identifier:** `multiroute_replay_seeded_20261110`.

## Question

Attempt 001 obtained no successful training episodes. Does providing a small
number of successful trajectories help the same DQN learn a useful policy?
This attempt changes the initial replay contents only. It does not change the
network, reward, online episode budget, exploration schedule or update schedule.
There is no behavior-cloning loss or pretraining weight update.

## Protocol fixed before observing results

- Environment: `configs/environment_multiroute.yaml`.
- Network: `configs/dqn_multiroute_pilot.yaml`.
- Schedule: `configs/training_multiroute_replay_seeded.yaml`.
- Online training: 30 episodes, scene seeds 0--29, network/exploration seed 0.
- Demonstrations: one reference attempt per seed 10000--10009, using a separate
  environment with the same reset and physics settings. Keep only successful
  complete trajectories, but retain a record of all attempted seeds.
- Validation and checkpoint selection: seeds 100000--100002, before training
  and at episodes 10, 20 and 30, matching attempt 001. No final test seeds used.
- Replay sampling remains uniform. Demonstrations are neither protected from
  later eviction nor forced into each minibatch. Capacity is 20000; no eviction
  is expected within this short pilot.
- Success, damage, budget exhaustion, final distance and movement diagnostics
  determine interpretation. Training loss alone does not establish success.

The comparison holds the *online episode budget* fixed, not the actual action
or optimizer-update count: earlier termination can change both. Demonstrations
are additional training data from a planner with geometry access. Report their
actions separately. Collection counts exclude planner search and reset-time
feasibility probes; they are not total simulator work or an equal-data comparison.

## Commands

From the repository root, with the declared Python environment activated:

```powershell
python -u scripts/train_dqn.py --config configs/environment_multiroute.yaml --dqn-config configs/dqn_multiroute_pilot.yaml --training-config configs/training_multiroute_replay_seeded.yaml --output-dir results/training/multiroute_replay_seeded_20261110
python scripts/summarize_training.py --run-dir results/training/multiroute_replay_seeded_20261110
python scripts/evaluate_dqn.py --checkpoint results/training/multiroute_replay_seeded_20261110/best.pt --episodes 3 --seed 100000
python scripts/archive_training_attempt.py --run-dir results/training/multiroute_replay_seeded_20261110 --output-dir docs/experiments/002_reference_replay/evidence
```

The actual run uses `.venv/Scripts/python.exe`. Its `provenance.json` records
UTC launch time, arguments, packages, Git base revision and source SHA-256 hashes.
The working tree includes the new initializer, so the base revision alone does
not identify the executed source. The matching changes are retained in Git.

## Implementation checks

Before launch, 28 focused tests passed for actual demonstration transitions,
final observations and end flags, failure/partial-collection records, replay
capacity protection, seed separation, unchanged weights and exploration RNG,
training/reload integration, and provenance/completion/failure recording.
Existing training and validation regression checks also passed (21 tests).

Each started training run records its status. Demonstration collection writes
progress after each completed trajectory and on handled interruption, preserving
previously collected evidence. Hard process termination can leave `running`
status and the last saved snapshot; it must not be described as completed.

## Results and interpretation

Pending completion. A completed process will not by itself be called a
successfully trained policy. The final interpretation must include failures and
the limitations of one training seed and three development validation scenes.
