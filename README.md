# ME5418 Robotic Harvesting RL

Course project scaffold for studying reinforcement learning and model-based planning for contact-aware motion through flexible branches.

## Research questions

1. Can deeper receding-horizon planning match or outperform DQN in a deterministic simulator?
2. How do planning and learned policies degrade under bounded execution noise?
3. Which method gives the best performance versus online-computation trade-off?

## Project structure

```text
configs/                      Experiment configuration
src/me5418_harvesting/envs/   Scene generation, branch physics, Gymnasium environment
src/me5418_harvesting/agents/ DQN configuration and network
src/me5418_harvesting/planners/ Planning baselines
src/me5418_harvesting/evaluation/ Metrics and summaries
scripts/                      Entry points for smoke tests, training, and evaluation
tests/                        Contract and reproducibility tests
results/                      Generated outputs, ignored except for .gitkeep
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
python scripts/run_smoke.py
pytest
```

Install the training dependencies when the DQN training loop is added:

```bash
python -m pip install -e ".[train]"
```

## Current scope

The repository contains a runnable environment scaffold with an eight-action interface, deterministic and noisy transition modes, padded observations for up to five branches, reward accounting, planner interfaces, and evaluation metrics. The branch-contact update is deliberately lightweight and must be replaced or calibrated before final experiments.

## Development order

1. Freeze the state, action, reward, termination, and noise contracts.
2. Validate branch deformation, collision handling, and scene feasibility.
3. Implement collision-free, myopic, and receding-horizon baselines.
4. Add DQN training with replay, target network, and epsilon-greedy exploration.
5. Run planning-depth and noise-level ablations with independent seeds.
6. Produce reproducible tables, plots, and failure-case visualizations.

