# Team handoff: 16 November 2026

## Shared baseline and independent work

The shared baseline includes attempt 034's trained candidates and attempt 035's
verified scene cache. The pre-handoff code/results commit is `c52016a`.
The teammate branch `collab/teammate-20261116` starts at the commit containing this
handoff. Work on that branch rather than directly changing main. The original
checkout remains on main; each teammate should use their own clone/worktree.

Existing clone, first checkout of the teammate branch:

```bash
git fetch origin
git switch --track origin/collab/teammate-20261116
```

New clone:

```bash
git clone --branch collab/teammate-20261116 https://github.com/JYAXR/me5418-robotic-harvesting-rl.git
cd me5418-robotic-harvesting-rl
```

Commit and push changes on the teammate branch, then open a PR into main. Before
starting overlapping work, agree on experiment ownership and reserve a unique
experiment/output directory. Do not overwrite earlier runs or force-push main.
Use ordinary merges from origin/main when updating a shared working branch;
do not rewrite commits already used by another teammate. Code, comments and
experiment records should remain in English.

## Environment and quick checks

The exact existing dependency specification is environment.yml. It includes an
editable install of this repository; do not silently upgrade packages during a
reproduction run.

```bash
conda env create -f environment.yml
conda activate me5418-harvesting
python -m pytest tests/test_scene_cache.py tests/test_scene_feasibility.py tests/test_environment_contract.py tests/test_training.py -q
python scripts/evaluate_dqn.py --checkpoint models/supervision_034_targeted.pt --episodes 3 --seed 10000 --timings results/teammate_evaluation_timing.json
```

The relevant regression suite passed 49 tests before attempt 035. Frozen evaluation
should reproduce successful 40/47/43-step episodes on seeds 10000/10001/10002.
Evaluation can create local output files, which should not be confused with new
training data or final submission artifacts.

## Latest model and limits

- Inference candidate: models/supervision_034_targeted.pt.
  SHA-256: 202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b.
- Matched control: models/supervision_034_control.pt.
- Full targeted resume state:
  docs/experiments/034_scene10002_supervision/checkpoints/targeted_state_12000.pt.
- Full control resume state:
  docs/experiments/034_scene10002_supervision/checkpoints/control_state_12000.pt.
- TD clock 12000; retained Adam clock 32000. Use full snapshots for continuation;
  inference files do not contain the historical optimizer state.

Attempt 034 doubled only the existing supervised weights at dataset rows
98/102/103 (scene 10002 steps 23/27/28, labels 4/0/4). Both arms continued the same
source for 2000 TD updates. The targeted endpoint achieved 3/3 original-scene
success; the control achieved 2/3, timing out on 10002. All evaluations used zero
action overrides. Forty independent update checks and six endpoint reloads passed.

Do not overstate these results:

- Both arms failed 10002 at updates 100/500/1000. Continuous retention is unproven.
- Targeted 10000 takes 40 steps versus 38 for source/control; success is retained
  with a small efficiency loss. Scene 10001's final metrics are unchanged.
- Targeted 10002 succeeds in 43 steps by an alternative valid turn, not by exactly
  reproducing all three reference actions.
- The latest model has not been evaluated on all 33 training scenes or fresh
  unseen scenes. The 13/100 independent-test result belongs to attempt 030,
  not the latest model.
- Earlier reports and submission archives have not been updated for this latest
  experimental state. They are not a final-project-ready handoff package.

## Cache and timing

The optional CertifiedSceneCache stores accepted seeded initial states and RNG
state. It preserves returned reset information. last_reset_profile separately
records actual executed certification work; a cache hit has zero new certificate
calls/actions even though historical reference_route_checks are retained.

Reuse one cache object across repeated evaluations in the same process. A single
CLI run with distinct seeds does not benefit from hits, and separate processes
do not share this cache. See attempt 035's RESULTS.md for API examples and limits.
The measured 2.84x warm / 1.80x cold-inclusive gains apply only to repeated
evaluation. Physics is the main remaining warm-evaluation cost.

Default evaluation and environment interfaces remain uncached. Historical runners
that require a fresh certification call every reset should remain uncached.
Hash-pinned historical experiments should be reproduced at their original Git
commits; current source includes the later caching/timing changes.

## Suggested next experiment, not yet performed

Freeze both attempt-034 inference candidates. Evaluate all 33 training scenes,
then a predeclared fresh independent test bank using matched scenes for both
models. No tuning, online learning or action replacement during evaluation.

Training seeds are 10000/10001/10002 and 290000--290029. Previously observed test
banks include 280000--280099, 390000--390099 and 490000--490099. Check the complete
experiment index before choosing any new test bank; do not reuse observed seeds
as a fresh holdout. Keep reset errors and all failure outcomes without replacement.
Reserve the next experiment number with the teammate before implementation.

Retain protocol, source/model hashes, settings, every outcome, actual simulation
cost, timing and validation evidence. Compare full training-state preservation
or trajectory equivalence as appropriate. Commit the protocol before execution,
then record/push all results, including unsuccessful attempts.

## Reading order

1. [Experiment index](experiments/README.md).
2. [Latest learning result](experiments/034_scene10002_supervision/RESULTS.md).
3. [Cache benchmark and usage](experiments/035_scene_cache/RESULTS.md).
4. [Model catalog](../models/README.md).
5. Attempts 030--033 for the expanded-data baseline and causal action diagnostics.

Tracked evidence and final models are already in Git. Local .venv directories,
routine results directories and submission ZIP files are intentionally not
synchronized. Do not commit another teammate's unrelated local documents.
