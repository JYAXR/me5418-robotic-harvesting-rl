# Attempt 028: the frozen repaired policy generalizes poorly

Completed on 15 November 2026. The policy from attempt 027 succeeds in **1 of
100 previously unused scenes (1%)** under its unchanged scene-generation
configuration. Fifty scenes terminate with damage and 49 exhaust the action
budget. The three training scenes still succeed before and after the assessment.
The repair therefore preserves the measured training behaviors, but does not
provide reliable performance on unseen scenes from the same generator.

The [protocol](../028_frozen_generalization.md), evaluator and focused tests
were committed and pushed before evaluation in `4fd772d`. No weight, training
setting, reward, scene parameter or action-selection rule was changed in this
attempt. The original attempt-027 checkpoint remains the candidate; no checkpoint
was selected using these results.

## Frozen model and scene split

- Checkpoint: [step6_repair_027.pt](../../../models/step6_repair_027.pt), absolute TD 10000.
- SHA-256: `06733fccb5422d10ad7246e5368bbd5ed9cc3367f0b413847544ec26a96b0c88`.
- Training scenes: 10000, 10001 and 10002, confirmed against checkpoint metadata
  and the attempt-011 dataset. The added recovery observation also comes from 10001.
- Test scenes: exactly seeds 280000--280099, declared before any of their evaluations.
- Network inference: greedy, epsilon zero; no scripted actions or corrections.
- Environment: the embedded multi_route configuration, 14 branches, 200-action
  budget, zero execution noise and the original bounded position randomization.

All 100 requested scenes initialize successfully and have distinct encoded
initial observations. None matches a training initial observation. The unchanged
environment certifies a collision-free reference route before accepting a scene;
all 100 main resets accept their first route-certified candidate. No requested
seed is removed or replaced. This is a within-distribution test of new realizations,
not a test of a different layout family or stronger randomization.

Reference certification witnesses feasibility using copied physics. Its 100/100
accepted-route success is a consequence of the environment's acceptance rule;
it is not an independently sampled, unbiased planner benchmark. The greedy
policy receives observations only, not the reference actions.

## Main results

| Outcome | Scenes | Fraction of 100 |
| --- | ---: | ---: |
| Success | 1 | 1% |
| Damage | 50 | 50% |
| Budget exhausted: persistent blocked stall | 45 | 45% |
| Budget exhausted: other temporal stall pattern | 4 | 4% |
| Reset or evaluation errors | 0 | 0% |

Success is 1/100 both over requested scenes and over initialized scenes. The
95% Wilson interval is **0.18%--5.45%**, describing scene-sampling uncertainty
for this fixed candidate, not variation across training initializations. Mean
return is -56.770951 and mean episode length is 118.52 actions.

The sole successful scene is **280046**: 36 actions, return 107.337836, no blocked
or stationary actions, and peak normalized deformation 0.054818. Its independent
normal-evaluator reload reproduces the metrics exactly.

All six known-scene control episodes succeed: seeds 10000, 10001 and 10002 both
before and after testing, in 35, 41 and 41 steps respectively. Before/after traces
are byte-identical. These repeated training-scene controls are not added to the
holdout numerator or denominator.

## Observed failure signatures

Damage occurs in 50 scenes, between actions 9 and 181 (mean 40.32 actions).
Their peak deformation exceeds the environment's safety threshold of 1. For
example, scene 280000 first makes contact at action 9 and terminates at action
12 with peak deformation 1.047680, despite having no collision-blocked steps.
Thus absence of blocking is not evidence of safe interaction.

All 49 budget failures terminate at 200 actions and have 113--196 blocked actions.
The 45 persistent-stall cases average 88.63% blocked and 87.96% stationary actions.
Scene 280001 starts blocking at action 25, then has 175 consecutive stationary
actions at the end; it finishes with 176 blocked actions and remains unsuccessful.
Its peak deformation is 0.636218, so this is a timeout without damage.

The predeclared classifier assigns the remaining four cases the machine label
`budget_while_moving` because the final stationary streak is shorter than 20.
This label must not be read as free or productive motion. Full traces show:

| Scene | Blocked actions | Stationary actions | Longest stationary run | Final stationary run |
| --- | ---: | ---: | ---: | ---: |
| 280064 | 196 | 183 | 16 | 3 |
| 280072 | 180 | 178 | 178 | 0 |
| 280076 | 195 | 184 | 20 | 4 |
| 280094 | 168 | 165 | 165 | 0 |

These cases also spend most of the episode blocked, but have intermittent small
motion or resume moving near the budget boundary. The original categories and
thresholds are retained; this finer interpretation is post-hoc trace analysis.
For example, 280064 keeps selecting east_small in all of its last ten actions
and ends 0.715915 from the target. No action is replaced to test a rescue.

These are measured failure modes and temporal signatures. The evaluation does
not establish a unique internal cause, identify an optimal replacement action,
or prove that a particular gradient/optimizer mechanism caused generalization
failure. Narrow training coverage is a plausible explanation, not an isolated
causal finding from this experiment. The measured conclusion is that the current
policy does not transfer reliably to these new scenes.

## Verification and work accounting

Online and target networks are in evaluation mode with gradients disabled.
Optimizer.step raises if called. Complete online/target tensors, optimizer state,
update count and agent/Torch RNG are compared around every evaluation. Checkpoint
bytes and bound source files are unchanged at completion. There are zero optimizer
updates, zero added training samples and zero forced policy actions.

Seven normal-evaluator reloads cover the first five test seeds and the lowest
seed in each observed outcome category, deduplicated: 280000--280004, 280046 and
280064. Each is freshly loaded from the same file and exactly matches its main
episode metrics and reset certification. Ten focused tests pass, including a
known-scene comparison with the previously delivered model. Ruff passes.

All 106 main/control traces (12086 actions) are audited independently for greedy
Q selection, top-Q gap, observation/position continuity, reward sum, termination,
blocking, stationarity and classification. Reloads use the existing normal
evaluate function and contribute additional actions separately.

| Formal work | Resets | Policy actions | Copied certification actions |
| --- | ---: | ---: | ---: |
| Main unseen scenes | 100 | 11852 | 4304 |
| Known-scene controls, before and after | 6 | 234 | 234 |
| Independent normal-evaluator reloads | 7 | 575 | 296 |
| Total | 113 | 12661 | 4834 |

The total is 17495 simulator actions. Unit-test simulation is excluded from this
formal accounting. There is no training, demonstration collection, policy tuning
or additional failed-case intervention.

## Evidence and reproduction

- [Per-scene CSV](evidence/per_scene.csv): outcome and core metrics for every requested seed.
- [Summary](evidence/summary.json) and [diagnoses](evidence/diagnoses.json): complete counts and failure signatures.
- [Episode records](evidence/episodes.json): initial geometry, reset checks, hashes and metrics.
- `evidence/scene_<seed>.jsonl`: every main action, Q vector, reward and transition information.
- [Verification](evidence/verification.json), [accounting](evidence/accounting.json),
  [provenance](evidence/provenance.json) and [archive manifest](evidence/archive_manifest.json).

The archive includes every main/control action trace, all requested scene records,
console output, tests, source hashes and reload results; failures are not omitted.
The frozen model itself was already committed in attempt 027.

To evaluate the frozen model through the normal entry point from the repository root:

```bash
python scripts/evaluate_dqn.py --checkpoint models/step6_repair_027.pt --episodes 100 --seed 280000
```

That command repeats the same now-observed scene set; it does not create a fresh
blind test. The archival instrumented run was executed with:

```bash
python docs/experiments/028_frozen_generalization/run_frozen_generalization.py
python docs/experiments/028_frozen_generalization/analyze_generalization.py
```

The archival scripts refuse to overwrite existing output/archive directories.
Focused validation:

```bash
python -m pytest -q tests/test_frozen_generalization.py
ruff check docs/experiments/028_frozen_generalization tests/test_frozen_generalization.py
```

## Implication for the next attempt

Training-scene retention and unseen-scene competence are different objectives.
Attempt 027 addressed the former; this frozen evaluation exposes a substantial
gap in the latter. This attempt makes no policy change. Any later repair should
be evaluated on another predeclared disjoint set; once these outcomes inform
design choices, seeds 280000--280099 become development evidence rather than a
blind final test. Expanding training coverage and testing recovery across varied
layouts are candidates for a separate controlled experiment, not changes made here.
