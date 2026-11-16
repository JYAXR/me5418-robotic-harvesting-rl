# Own-route attempt 002: one-action rescue of three regressed scenes

## Fixed question

For seeds 290007, 290012 and 290021, does replacing only the first action that
differs from the stored demonstration restore success? Both arms use the same
frozen attempt-034 targeted policy. The autonomous arm is not the attempt-034
control model. A successful correction would demonstrate intervention sufficiency
on that scene, not learned recovery or generalization.

Branch: `experiment/your-method`; preceding evidence commit: `1f0e468`.

| Seed | One-based intervention step | Autonomous action at that input | Replacement | Zero-based demonstration row |
| --- | ---: | --- | --- | ---: |
| 290007 | 8 | north_small (0) | east_large (6) | 430 |
| 290012 | 19 | east_small (2) | east_large (6) | 653 |
| 290021 | 12 | north_small (0) | east_large (6) | 1052 |

The locations were measured in own-route attempt 001. Verify each reference
prefix and encoded input against the unchanged 1440-transition demonstration
dataset before running; do not move the intervention based on new outcomes.

## Frozen inputs and conditions

- Inference model: `models/supervision_034_targeted.pt`, SHA-256
  `202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b`.
- Full snapshot:
  `docs/experiments/034_scene10002_supervision/checkpoints/targeted_state_12000.pt`,
  SHA-256 `fe8e29438165cd4d99c6e143f819a37a2febc5e704a47efb9a895bb1074c610f`.
- Restore all weights, optimizer moments, counters and RNG states from the full
  snapshot. TD clock remains 12000; Adam clock remains 32000. Disable gradients
  and reject any optimizer step. CPU, one Torch thread, greedy action selection.
- Preserve environment configuration, seeds, rewards and 200-step budget.
  Use ordinary uncached resets for every primary and independent replay. This
  matches attempt 001's verified reset/trajectory semantics; cache timing is not
  a comparison endpoint. Count every actually executed certification action.
- Baseline: no action replacement. Corrected: exactly one replacement at the
  specified step. Every subsequent action is the model's greedy decision at the
  newly visited state. No later corrections, online learning or extra examples.

## Execution and checks

Run baseline and corrected full episodes for each seed, then independently
reload and repeat both arms through the normal evaluator: six primary plus six
verification episodes, with zero optimizer updates. There are three primary
forced actions and three additional verification forced actions. No full
scripted-reference episode or untouched test scene is required.

Require baseline metrics and every shared historical trace field to match
attempt 001. Require exact equality of baseline/corrected prefixes and fork
inputs/Q values, exactly one forced action per corrected episode, and greedy
actions elsewhere. Require exact full-state preservation and independent replay
of complete transitions, certification and outcomes. Hash all source/input files
before and after execution. Preserve any failed execution and partial evidence;
do not silently rerun or replace seeds.

Report success, damage, timeout, steps, return, travel, deformation, distance,
blocked/stationary steps, correction Q gap, simulation/certification counts and
runtime for both arms. A negative result means this single correction is
insufficient; it does not establish that the action was harmless or that no
later correction could help. Do not add another correction within this attempt.

Commit and push protocol, runner and override-audit tests before execution.
Retain complete traces, results, independent replay evidence, provenance and
hash manifest, then commit and push the results to our branch only.

```bash
python -m pytest docs/experiments/your_method/002_first_action_rescue/test_rescue_audit.py -q
python docs/experiments/your_method/002_first_action_rescue/run_rescue.py
```
