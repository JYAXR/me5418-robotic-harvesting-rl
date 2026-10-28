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

To build a submission containing source, configuration, tests, and the report:

```bash
python scripts/build_submission.py
```

Use `--output path/to/submission.zip` to choose another destination. Even after
running demos, the archive excludes their GIFs, images, traces, and metrics.
All code comments, docstrings, exception messages, and console output use English.

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
python scripts/run_demo.py --policy push --seed 0
python scripts/run_demo.py --policy detour --seed 0
pytest -q
```

The demo writes an animated GIF to `results/demo/`. A random-policy episode can
be generated with:

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

An earlier paired diagnostic covered seeds 0-9. Both hybrid and non-contact A* succeeded
on nine scenes; seed 2 remains a recorded planning failure for both. No checked
episode triggers damage. This does not prove that seed 2 is physically
unsolvable or establish learned-policy performance. Seed 0 demonstrates both
routes: the hybrid opens soft gaps and detours around the closed region, while
non-contact A* takes a longer route. Re-run the commands above to generate
local diagnostic files; previous run outputs are not bundled.

## Experimental inclined-branch layout

`configs/environment_inclined.yaml` adds independently left/right inclined
branches, with anchor-to-tip angles of 12-22 degrees from vertical. Inclination
changes the actual resting/current nodes used by the contact solver, not just
the drawing. The tip gaps are still narrower than the robot diameter. With a
matched seed, target, gap heights, gap widths, stiffness, and safety limits are
identical to the upright layout; only horizontal branch geometry changes.
The original default configuration remains available.

Run a successful inclined contact example and its same-scene detour:

```powershell
python scripts/run_demo.py --policy push --seed 1 --config configs/environment_inclined.yaml --output-dir results/inclined_demo
python scripts/run_demo.py --policy detour --seed 1 --config configs/environment_inclined.yaml --output-dir results/inclined_demo
python scripts/compare_demo.py --seeds 10 --config configs/environment_inclined.yaml --output-dir results/inclined_comparison
python scripts/build_comparison_figure.py --seed 1 --demo-dir results/inclined_demo --output-dir results/inclined_comparison --title "Inclined branches, seed 1"
```

The unmodified handwritten contact controller succeeds on 6/10 inclined scenes
(seeds 0-9), triggers damage on three, and times out on one. Collision-free A*
succeeds on all ten in the earlier diagnostic. Re-run the demo with seed 0 to
generate the failed contact example or seed 1 for the successful example.
Inclination exposes limitations of aligning to
resting tips and then pushing horizontally; these results are not trained RL
evaluation. Contact physics and safety thresholds were not relaxed for this
variant. Recorded metrics include clearance and segment-strain checks.

## Environment definition

- **Observation:** end-effector position, target-relative position, current and
  resting branch nodes, stiffness, contact flag, safe deformation limit,
  branch-validity mask, and remaining action budget. Unused branch slots are
  zero padded.
- **Action:** eight discrete planar motions: north, south, east, or west with a
  small or large step size.
- **Reward:** target progress minus normalized deformation and travel cost,
  with a large damage penalty and a success bonus.
- **Termination:** the episode ends after success, unsafe deformation, or the
  configured step limit.
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
src/me5418_harvesting/agents/     DQN configuration and Q-network builder
src/me5418_harvesting/planners/   Collision-free A* and receding-horizon interface
src/me5418_harvesting/evaluation/ Episode metrics and summaries
scripts/run_demo.py               Graded graphical demonstration
scripts/compare_demo.py           Same-scene push versus detour comparison
scripts/run_smoke.py              Minimal text-only smoke test
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

The DQN and experiment configuration files document the intended final-project
comparison with receding-horizon planning. This assignment submission focuses
on the environment, its reproducible interface, and the required visual demo;
it does not claim that the placeholder DQN network has already been trained.

## Testing

```bash
pytest -q
ruff check .
```

The tests check the Gymnasium interface, seeded reproducibility, RGB rendering,
trajectory bookkeeping, narrow-gap geometry, fixed anchors, segment strain,
soft-gap opening, rigid-gap blocking, and elastic recovery after release.
Inclined-layout tests additionally check seed-matched geometry, tilt bounds,
invalid settings, angled rigid blocking, and soft-gap opening without penetration.
Mixed-layout tests check all three passage types, diverse branch geometry,
complete observations, rigid blocking, and the different actual contact
behaviour of natural openings versus soft gaps.

The paired diagnostic set covers seeds 0-9. These checks demonstrate that both
routes are available in the chosen surrogate. They do not establish a learned
policy advantage; final experiments still require a trained DQN, held-out seeds,
noise tests, and sensitivity to gap width, stiffness, and damage thresholds.
