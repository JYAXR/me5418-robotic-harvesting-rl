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
- [Chinese reference (three pages)](report/ME5418_Group18_Neural_Network_Report_Chinese.docx): the matching translation for review, excluded from the submission archive.

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
configured update interval. Success/damage stops bootstrapping; time-limit
truncation uses the final observation before reset and continues bootstrapping.
Finally, the updated network selects and executes one action in the environment.

Use `--seed 7`, `--batch-size 16 --transitions 32`, or `--updates 5` to vary the
demonstration. Defaults for both YAML paths are resolved relative to the project,
so the script also works when invoked from another directory. Output is printed
to the console; no checkpoints or result files are written.

This demonstrates inference, backpropagation, and parameter updates. It does not
establish a trained harvesting policy or a performance advantage. Replay-buffer
training, warmup, and epsilon schedules remain future full-training work; the
short demo samples minibatches from a fixed collection of real transitions.

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

Inclination changes how horizontal pushing interacts with the branches. Use the
comparison command to inspect success, damage and timeout outcomes across seeds.
Contact physics and safety thresholds are the same as for the upright layout.
Generated metrics include clearance and segment-strain checks.

## Environment definition

- **Observation:** end-effector position, target-relative position, current and
  resting branch nodes, stiffness, contact flag, safe deformation limit,
  branch-validity mask, and remaining action budget. Unused branch slots are
  zero padded.
- **Action:** eight discrete planar motions: north, south, east, or west with a
  small or large step size.
- **Reward:** target progress minus normalized deformation and travel cost,
  with a large damage penalty and a success bonus.
- **Termination:** success or unsafe deformation terminates the episode;
  reaching the configured step limit truncates it.
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
src/me5418_harvesting/agents/     Q-network construction and DQN update utilities
src/me5418_harvesting/planners/   Collision-free A* and receding-horizon interface
src/me5418_harvesting/evaluation/ Episode metrics and summaries
scripts/run_demo.py               Graded graphical demonstration
scripts/compare_demo.py           Same-scene push versus detour comparison
scripts/run_smoke.py              Minimal text-only smoke test
scripts/run_network_demo.py       Q-network forward/backward and environment action demo
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
The DQN and experiment configurations also document the intended final-project
comparison with receding-horizon planning; full policy training and evaluation
remain outstanding.

## Testing

```bash
python -m pip install -e ".[dev,network]"
pytest -q
ruff check .
```

The network tests check both observation sizes, terminal versus time-limit
Bellman targets, final-observation handling, frozen target-network parameters,
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
