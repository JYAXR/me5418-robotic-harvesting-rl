# ME5418 Flexible Obstacle Interaction Environment

This repository contains Group 18's reinforcement-learning environment for a
two-dimensional robotic harvesting task. A circular end effector must reach a
fruit while choosing between detours and controlled contact with flexible
branches. The same environment supports deterministic transitions and bounded
execution noise.

## Clean submission

Run outputs are generated locally and are not included in the repository or
submission archive. The tracked `results/` directory contains only `.gitkeep`.
Running a demo creates its output directory and files as needed.

To build a submission containing source, configuration, tests, and the current
two-page English Word report:

```bash
python scripts/build_submission.py
```

Use `--output path/to/submission.zip` to choose another destination. Even after
running demos, the archive excludes their GIFs, images, traces, and metrics.
All code comments, docstrings, exception messages, and console output use English.

The default archive is `ME5418_Group18_Submission.zip`, with a single
`ME5418_Group18/` project folder. Its report is
`report/ME5418_Group18_Neural_Network_Report_Revised.docx`. Older reports,
the Chinese reference translation, local environments, and rendering previews
are excluded.

## Reports

- [English report (two pages)](report/ME5418_Group18_Neural_Network_Report_Revised.docx): the current submission report.
- [Chinese reference (three pages)](report/ME5418_Group18_Neural_Network_Report_Chinese.docx): an earlier network-demo report, not updated for the Learning Agent and excluded from the submission archive.

## Neural-network forward/backward demo

The Q network is a fully connected MLP with two 256-unit ReLU hidden layers
and eight action-value outputs. The input size is read from the environment:
125 for the default scene and 341 for the mixed canopy.

The Conda environment includes CPU PyTorch for Windows and Linux. Alternatively,
install the CPU wheel and the project in an existing Python 3.11+ environment:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,network]"
python scripts/run_network_demo.py
python scripts/run_network_demo.py --config configs/environment_complex.yaml
```

The demo collects 128 actual random-policy transitions and performs three Adam
updates with the settings in `configs/dqn.yaml`. It prints JSON containing a
sample observation, the eight Q values, a selected action, a Bellman target,
Huber loss before/after each update, gradient norm before clipping, and the
parameter-change norm. Both losses for an update use the same detached targets.
The frozen target network receives no gradients and is synchronized at the
configured update interval. Success, damage and exhaustion of the task's action
budget stop bootstrapping: their Bellman target is the immediate reward alone.
The budget is 100 actions by default and 200 in the mixed canopy; remaining
budget is part of the observation. Only an external interruption before task
termination, such as a shorter `TimeLimit` wrapper, continues bootstrapping from
the final observation before reset.
Finally, the updated network selects and executes one action in the environment.

Use `--seed 7`, `--batch-size 16 --transitions 32`, or `--updates 5` to vary the
demonstration. Defaults for both YAML paths are resolved relative to the project,
so the script also works when invoked from another directory. Output is printed
to the console; no checkpoints or result files are written.

This demonstrates inference, backpropagation, and parameter updates. It does not
establish a trained harvesting policy or a performance advantage. The short demo
samples a fixed collection of transitions; the Learning Agent below collects
experience continuously across training episodes.

## Learning Agent: training and evaluation

The Learning Agent follows the lecture structure: initialize the environment and
network, reset each episode, choose an action, step the environment, store the
transition, and update the network when enough experience is available. Episode
statistics reset between episodes; replay, exploration and update counters persist.
All execution is on CPU. The existing Conda environment includes the dependencies.

For a short check with real network updates, model saving, and greedy validation:

```bash
python scripts/train_dqn.py --smoke --output-dir results/training/smoke_default
python scripts/evaluate_dqn.py --checkpoint results/training/smoke_default/latest.pt --episodes 3
python scripts/train_dqn.py --smoke --config configs/environment_complex.yaml --output-dir results/training/smoke_complex
python scripts/evaluate_dqn.py --checkpoint results/training/smoke_complex/latest.pt --episodes 3
```

Choose a fresh output directory for each run; existing directories are rejected
to protect previous results. Without `--output-dir`, a new timestamped directory
is created under `results/training/`. The smoke preset uses three episodes, a
16-transition minibatch, 16-step warmup, a 512-transition replay buffer, a
200-step exploration decay, and target synchronization every 10 optimizer updates.
It preserves the actual 100/200-action task budgets. It checks the execution path,
not policy convergence or harvesting success.

For longer training:

```bash
python scripts/train_dqn.py --episodes 200 --seed 0 --output-dir results/training/default_run
python scripts/evaluate_dqn.py --checkpoint results/training/default_run/best.pt --episodes 20 --seed 1000000
```

`configs/dqn.yaml` controls the network, replay capacity, minibatch size, warmup,
epsilon schedule and optimizer. `configs/training.yaml` controls episode count,
training frequency, validation frequency and seed; it can also override warmup
and epsilon settings. Exploration decays linearly with total environment steps.
After warmup and once a full batch is available, one random minibatch is used
every four environment steps by default. Target synchronization counts optimizer
updates, not environment steps. A short run may finish before warmup; the CLI
explicitly reports when no updates occurred.

The agent normalizes positions and offsets by workspace dimensions, scales safety
limits by workspace size, and maps stiffness with `log(1 + stiffness) / log(1 + 1e6)`.
Contact flags, branch masks and remaining budget retain their meanings; unused
branch slots remain zero. These fixed transformations are identical in training
and evaluation and do not estimate statistics from evaluation data. Default and
complex scenes use separate models because their observation dimensions differ.

Each run writes only to its output directory:

- `settings.json`: resolved environment, network, schedule and encoder settings.
- `metrics.jsonl`: episode returns, losses, update counts, exploration, replay size,
  success, damage, budget exhaustion, travel and peak deformation; periodic validation.
- `latest.pt`: policy at the latest validation, including the final episode.
- `best.pt`: policy with the highest validation success rate, breaking ties by return.
- `summary.json`: final counters and validation results.

Validation uses a separate environment and fixed seeds immediately following the
training seed range. Evaluation disables exploration and performs no learning.
The standalone evaluation CLI defaults to seeds starting at 1000000; choose a
range disjoint from training and validation for final testing. No-success
evaluations report `null` for mean successful-episode length, not a fabricated zero.
Budget exhaustion can coincide with success or damage on the last action, so
these flags are reported independently.

Policy files include environment settings, network architecture, encoder version
and weights, allowing evaluation without manually matching input dimensions.
They are inference checkpoints, not exact training-resume snapshots: optimizer,
replay and random-generator states are not saved. The English report describes
the Learning Agent, scene validation, position randomization and easy scene;
the Chinese reference remains from the earlier network-demo stage.
Full performance experiments remain future work.

Generated logs and model files under `results/` are ignored by Git and excluded
from the submission archive. Source files and training configuration are included.

## Position randomization and introductory training

The default, inclined, and mixed-canopy YAML configurations now enable bounded
position offsets on each reset. The table gives uniform offset amplitudes in
workspace units; for example, 0.015 means a draw between -0.015 and +0.015.

| Scene | Start offsets (x, y) | Target offsets (x, y) | Gate x offset | Extra twig offsets (x, y) |
| --- | --- | --- | --- | --- |
| Default / inclined | (0.015, 0.030) | (0.015, 0.020) | 0.025 | None |
| Mixed canopy | (0.015, 0.030) | (0.015, 0.020) | 0.015 | (0.010, 0.010) |
| Easy wide passage | (0.015, 0.030) | (0.015, 0.020) | 0.040 | None |

These offsets are added to the existing sampled geometry. Existing randomness
in target height, mixed-canopy start height, gap height and small-twig positions
still applies. Each opposing branch pair translates together along x, preserving
the gap width, branch shape and passage type, including touching closed tips.
Extra twigs translate as complete polylines. Rest and current nodes move together,
so position randomization does not introduce initial deformation.

All offsets are applied before the existing geometry checks; invalid candidates
are resampled with the same seeded RNG. Set `scene.start_jitter`,
`scene.target_jitter`, and `scene.twig_jitter` to `[0.0, 0.0]`, and
`scene.gate_x_jitter` to `0.0` to disable the new randomization. The Python
generator defaults to zero offsets for compatibility with older checkpoints.
Randomization is controlled by the scene configuration, so evaluation reproduces
the distribution saved in each policy file. These remain structured layouts,
not arbitrary tree topologies.

`configs/environment_easy.yaml` provides an introductory navigation task: two
branches form one open passage, with a 0.24-0.30 gap compared with the robot's
0.07 diameter. The gate center height ranges from 0.48 to 0.52. There are no extra
twigs or closed passages. The eight actions, reward terms, contact solver, damage
checks and 100-action budget are unchanged. Five branch slots are retained, with
three zero-padded slots, giving the same 125-input network as the default scene.
This scene practices navigation; it is not a test of learned branch manipulation.

Run the short end-to-end check and load its saved policy:

```bash
python scripts/train_dqn.py --smoke --config configs/environment_easy.yaml --output-dir results/training/easy_smoke
python scripts/evaluate_dqn.py --checkpoint results/training/easy_smoke/latest.pt --episodes 3
```

For a longer introductory run, `configs/training_easy.yaml` reduces warmup to
128 environment steps and epsilon decay to 10000 steps:

```bash
python scripts/train_dqn.py --config configs/environment_easy.yaml --training-config configs/training_easy.yaml --output-dir results/training/easy_run
```

The default training command still selects `configs/environment.yaml`. Select
the easy configuration explicitly. These are separate training runs; automatic
curriculum transitions and transfer from an easy checkpoint are not implemented.
A geometric direct-to-target controller can verify easy-scene feasibility, but
its success must not be reported as DQN learning performance.

## Quick start without Conda (Windows PowerShell)

Run from the extracted project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\run_demo.py --policy push --seed 0
.\.venv\Scripts\python.exe scripts\run_demo.py --policy detour --seed 0
Invoke-Item .\results\demo\push_seed_0.gif
```

If you already have a working Python environment, use its Python executable
instead. The local `.venv_me5418` used during development is not included in
the submission archive.

## Quick start with Conda

Run these commands from the repository root:

```bash
conda env create -f environment.yml
conda activate me5418-harvesting
python scripts/run_network_demo.py
python scripts/run_network_demo.py --config configs/environment_complex.yaml
python scripts/run_demo.py --policy push --seed 0
python scripts/run_demo.py --policy detour --seed 0
pytest -q
```

The network demo prints its forward/backward and update diagnostics to the
console. The optional environment demos write animated GIFs to `results/demo/`.
A random-policy episode can be generated with:

```bash
python scripts/run_demo.py --policy random --seed 7
```

## Why the layout contains useful contact choices

The default scene has four branches forming two pairs of opposing free tips.
Each tip gap is 0.028-0.043 wide, while the robot diameter is 0.070. Gap height
varies independently between 0.40 and 0.60, so a policy must find and align with
the passages. Opposing branch normals separate the tips when the robot pushes
into a gap. Displaced tips move back toward their resting positions on release.

The first gate has a lower boundary anchor and a free-standing upper anchor;
the second reverses those roles. This prevents a single top or bottom highway
from bypassing both gates. A collision-free route remains available around the
interior fixed anchors. Thus the environment permits both a longer, rigid
geometry route and a shorter route that opens internal passages through contact.

Compare both approaches on exactly the same scene:

```bash
python scripts/run_demo.py --policy push --seed 0
python scripts/run_demo.py --policy detour --seed 0
python scripts/compare_demo.py --seed 0 --seeds 10
```

`push` is a handwritten geometric controller: it locates each gap from the
observed branch nodes, aligns before contact, and uses small movements nearby.
`detour` uses A* with the same eight actions, checking the entire swept route
against resting branch geometry. It executes a planned action sequence rather
than cutting corners between waypoints. `greedy` moves directly toward the
fruit without finding the tip gaps; `random` samples seeded random actions.
The demonstrations are environment checks, not trained RL results.

For diagnostics without GIF generation:

```bash
python scripts/run_demo.py --policy push --seed 0 --no-gif
```

No external simulator is required. The visualization uses Matplotlib and the
GIF writer provided by Pillow.

## Complex mixed-canopy layout

`configs/environment_complex.yaml` increases the scene from four to fourteen
branches. Four passage regions mix the following behaviours in the same scene:

| Passage | Geometry and behaviour |
| --- | --- |
| Pushable (two) | Soft opposing tips, gap 0.028-0.043, smaller than robot diameter 0.070 |
| Naturally open (one) | Gap 0.115-0.150, allowing passage without deformation |
| Closed (one) | Touching tips with stiffness 1e6, an effectively rigid barrier; select another route |

The closed and open middle regions swap positions across seeds. Six additional
short curved, diagonal, or horizontal twigs occupy entry areas, inter-passage
corridors, and the fruit neighbourhood. Branch lengths, stiffness, and safety
limits vary. Brown branches mark the rigid barrier; red indicates contact.
All collision checks, anchor constraints, strain limits, and damage checks are
active. The branch model still omits inter-branch mechanical coupling and is
not a calibrated plant model.

This layout uses fourteen observation slots (341 features rather than 125) and
a 200-action budget. Keep observation dimensions consistent when building a
network; do not reuse a 125-input model. The original eight actions are unchanged.

```powershell
python scripts/run_demo.py --policy hybrid --seed 0 --config configs/environment_complex.yaml --output-dir results/complex_demo
python scripts/run_demo.py --policy detour --seed 0 --config configs/environment_complex.yaml --output-dir results/complex_demo
python scripts/compare_demo.py --policies hybrid detour --seeds 10 --config configs/environment_complex.yaml --output-dir results/complex_comparison
python scripts/build_comparison_figure.py --contact-policy hybrid --seed 0 --demo-dir results/complex_demo --output-dir results/complex_comparison
```

`hybrid` is a handwritten environment demo, not RL. It identifies narrow soft
tip pairs from geometry/stiffness, uses A* to approach them and route around
closed barriers, then executes small contact actions through the actual physics.
Planning treats all branches as obstacles; no closed barrier is removed.
The old `push` policy cannot handle closed passages because it tries every gap.
Planning failure is recorded separately from damage rather than raising an
unhandled error or claiming arrival.

Run the commands above to generate local diagnostic files and compare the
controllers on matched scenes. Check planning failure separately from damage
and inspect the trajectories to see where contact or detours occur. These
controllers are environment demonstrations rather than trained policies.

## Experimental inclined-branch layout

`configs/environment_inclined.yaml` adds independently left/right inclined
branches, with anchor-to-tip angles of 12-22 degrees from vertical. Inclination
changes the actual resting/current nodes used by the contact solver, not just
the drawing. The tip gaps are still narrower than the robot diameter. With a
matched seed, target, gap heights, gap widths, stiffness, and safety limits are
identical to the upright layout; only horizontal branch geometry changes.
The original default configuration remains available.

Run the inclined contact controller and its same-scene detour:

```powershell
python scripts/run_demo.py --policy push --seed 1 --config configs/environment_inclined.yaml --output-dir results/inclined_demo
python scripts/run_demo.py --policy detour --seed 1 --config configs/environment_inclined.yaml --output-dir results/inclined_demo
python scripts/compare_demo.py --seeds 10 --config configs/environment_inclined.yaml --output-dir results/inclined_comparison
python scripts/build_comparison_figure.py --seed 1 --demo-dir results/inclined_demo --output-dir results/inclined_comparison --title "Inclined branches, seed 1"
```

Position offsets are drawn before tilt so upright and inclined candidates with
matched seeds retain the same base geometry. If only one candidate fails validity
checks, rejection sampling can break that pairing; compare attempt counts when
performing a paired experiment.

Inclination changes how horizontal pushing interacts with the branches. Use the
comparison command to inspect success, damage and timeout outcomes across seeds.
Contact physics and safety thresholds are the same as for the upright layout.
Generated metrics include clearance and segment-strain checks.

## Environment definition

### Initial-scene validity

Every `reset()` validates the generated geometry before returning an observation.
The checks reject non-finite coordinates or material parameters, zero-length
segments, initially deformed branches, geometry outside the workspace, overlapping
start/target clearance regions with branches, and a start already inside the
success region. The robot's contact radius determines start clearance; the larger
of the target radius and contact radius determines target clearance. Both clearance
regions must fit within the workspace.

Branch self-intersections, crossings, T-junctions and collinear overlaps are also
rejected. Adjacent segments may share their connecting node. Across branches,
only the final tips of a declared closed passage may touch, without overlapping;
this preserves the intended rigid closed passages in the mixed canopy.

Invalid samples are resampled using the same seeded random generator. The default
limit is 100 attempts, adjustable with `scene.max_attempts` in environment YAML.
Exhausting the limit raises an English error with the last rejection reasons.
`reset()` returns `info["scene_generation_attempts"]` for diagnostics. Identical
settings and seeds reproduce both the accepted scene and the attempt count.
Previously invalid seeds may now produce a different, valid scene after retries.

These are initial-geometry checks, not a proof of route feasibility or harvesting
success. No static-planner rejection is applied to passages that could be opened
through deformation. These checks do not bypass contact physics.

- **Observation:** end-effector position, target-relative position, current and
  resting branch nodes, stiffness, contact flag, safe deformation limit,
  branch-validity mask, and remaining action budget. Unused branch slots are
  zero padded.
- **Action:** eight discrete planar motions: north, south, east, or west with a
  small or large step size.
- **Reward:** target progress minus normalized deformation and travel cost,
  with a large damage penalty and a success bonus.
- **Termination:** success, unsafe deformation or exhaustion of the configured
  action budget terminates the task (`terminated=True`, `truncated=False`).
  `info["budget_exhausted"]` records whether the budget was used up. External
  wrappers may truncate an unfinished task earlier.
- **Randomness:** `reset(seed=...)` controls scene generation. The
  `execution_noise_fraction` configuration adds bounded, zero-mean action
  perturbations while preserving full state observability.

The lightweight branch model subdivides every commanded motion to prevent
discrete-step tunnelling, detects contact against complete line segments,
projects contact corrections along the local normal, and constrains segment
strain to 3 percent. Fixed anchors remain fixed. Stiffness limits how far nodes
can move in one substep and controls restoration. Difficult substeps are reduced
until contact and segment constraints agree; otherwise the robot is blocked.
Damage uses the peak deformation over all substeps, so a transient overshoot
cannot be hidden by later relaxation.

This is a geometric quasi-static surrogate. It omits mass, inertia, hysteresis,
permanent deformation, and branch-to-branch mechanical coupling. Its parameters
are in normalized workspace units and have not been calibrated to a real plant.

## Visualization and demo

`HarvestingEnv` supports three render modes:

- `ansi`: one-line textual status for debugging;
- `rgb_array`: a 600 x 600 RGB frame for notebooks, tests, and GIF generation;
- `human`: an interactive Matplotlib window.

Each graphical frame shows the target region, end effector, executed
trajectory, resting branch shapes, and current branch shapes. A branch turns
red while it is in contact. A zoom panel shows the nearest tip gap, and the
frame reports travel and peak normalized deformation. The demo saves initial,
contact (for `push`), and final PNG snapshots plus a JSON metrics file.
Recorded state traces also allow regenerating the comparison figure:

```bash
python scripts/build_comparison_figure.py
```

## Repository structure

```text
configs/                          Environment and experiment settings
src/me5418_harvesting/envs/       Scene generation, branch model, Gym environment
src/me5418_harvesting/agents/     Q network, replay, preprocessing and Learning Agent
src/me5418_harvesting/planners/   Collision-free A* and receding-horizon interface
src/me5418_harvesting/evaluation/ Episode metrics and summaries
scripts/run_demo.py               Graded graphical demonstration
scripts/compare_demo.py           Same-scene push versus detour comparison
scripts/run_smoke.py              Minimal text-only smoke test
scripts/run_network_demo.py       Q-network forward/backward and environment action demo
scripts/train_dqn.py              Episode training loop, validation and checkpoints
scripts/evaluate_dqn.py           Load a policy and evaluate on seeded scenes
tests/                            Interface, reproducibility, and render tests
results/                          Empty output directory; files generated locally
```

## Configuration

The demos read `configs/environment.yaml` by default; use `--config path.yaml`
to select another file. Its `scene` section controls gap height, tip separation,
stiffness, and safe deformation ranges. To construct an environment directly:

```python
from pathlib import Path

import yaml

from me5418_harvesting import HarvestingEnv

config = yaml.safe_load(Path("configs/environment.yaml").read_text(encoding="utf-8"))
env = HarvestingEnv(config, render_mode="human")
observation, info = env.reset(seed=0)
```

The network demo exercises DQN updates on real environment transitions.
The Learning Agent supplies full training and saved-policy evaluation entry points.
The experiment configuration documents the intended final-project comparison
with receding-horizon planning; convergence and comparative performance still
require dedicated experiments.

## Testing

```bash
python -m pip install -e ".[dev,network]"
pytest -q
ruff check .
```

The tests check both 100- and 200-action task boundaries, reward-only Bellman
targets after budget exhaustion, and continued bootstrapping after genuine
external truncation. Network checks also cover both observation sizes,
final-observation handling, frozen target-network parameters,
gradient clipping, parameter updates, seeded reproducibility, and the CLI demo.
CI runs the tests and both network-demo layouts on Windows and Linux.

The tests check the Gymnasium interface, seeded reproducibility, RGB rendering,
trajectory bookkeeping, narrow-gap geometry, fixed anchors, segment strain,
soft-gap opening, rigid-gap blocking, and elastic recovery after release.
Inclined-layout tests additionally check seed-matched geometry, tilt bounds,
invalid settings, angled rigid blocking, and soft-gap opening without penetration.
Mixed-layout tests check all three passage types, diverse branch geometry,
complete observations, rigid blocking, and the different actual contact
behaviour of natural openings versus soft gaps.

The comparison scripts can generate paired diagnostics across matched seeds.
Final experiments still require a trained DQN, held-out seeds, noise tests, and
sensitivity to gap width, stiffness, and damage thresholds.
