# Attempt 031: frozen scene-10002 first-action correction

## Question and selection, before new simulator rollouts

Freeze the final attempt-030 model at TD update 10000. Locate its first departure
from the successful scene-10002 demonstration at an identical encoded state.
Test whether correcting that one action is sufficient to restore success.
No model fitting, parameter change, additional correction search, or held-out
evaluation is authorized by this diagnostic protocol.

Read-only inspection of the already archived attempt-030 trajectory identifies
step 23 (one-based), after 22 matching transitions. The greedy action is
`north_small` (0), Q = 80.40086364746094; the demonstrated action is
`north_large` (4), Q = 80.35923767089844. The policy-minus-reference gap is
0.0416259765625. This is initially an action disagreement, not proof that this
single decision causes the eventual damage failure at step 58.

## Immutable inputs

- Inference policy: `models/stages_030_td.pt`, SHA-256
  `e236374791ce5b7bb59c00d6e99c3ab557821fc06ccddbcbed6686b638b67868`.
- Full resume snapshot: a byte-for-byte copy of attempt-030
  `td/state_10000.pt` in `031_scene10002_rescue/inputs/final_td_state.pt`, SHA-256
  `1a63fba8f5c12b8cf13129b6dc290264c709d639e9a2b435d502d87d31ae7394`.
  Online/target networks, Adam moments and clock (30000), TD clock (10000),
  configuration, agent RNG and Torch RNG must remain exactly unchanged.
- Existing attempt-029 expanded demonstration arrays; scene 10002 uses rows
  [76, 117), a successful 41-action trajectory. No data collection or relabeling.
- Existing attempt-030 scene-10002 trace and metrics, including its failed outcome.

The network remains 341-256-256-8 with the existing encoder. Learning rate
0.0001, supervision weights, TD coefficient 0.5, and the no-MC TD recipe are
unchanged. These settings are historical provenance; this experiment performs
zero optimizer updates. Both networks disable gradients and optimizer stepping
raises an error.

## Predeclared evaluation

1. Reload the frozen full snapshot and replay the autonomous baseline. Require
   exact equality with all 58 historical transitions, Q vectors and final metrics.
2. Independently reload the same snapshot and seed. Replace only step 23 with
   `north_large`; every other action, including all later actions, is the current
   frozen policy's argmax. Compare the complete prefix and fork input/Q vector.
   Record success or failure without trying another intervention.
3. Replay all 41 stored demonstration actions once as a separate positive control.
   Verify every stored observation, next observation, float32 reward and end flag.
   This scripted trajectory is not autonomous policy performance.
4. Independently reload the baseline and corrected arms again and use the normal
   `training.evaluate` loop with an observation-only transition audit and exactly
   the same single override in the corrected arm. Compare every transition and
   final metric with the primary manual loop, including reset certification.

There are five physical episodes: two primary diagnostic episodes, one scripted
reference check, and two independent verification episodes. Count their executed
actions and reset-certification simulation actions separately. Exactly one
override is allowed in each corrected episode. There is no adaptive stopping
other than the unchanged environment termination/truncation conditions.

Save full Q vectors, encoded hashes, positions, rewards, termination flags and
environment info at every step. Record all attempts, including exceptions. Hash
all source/input files before and after execution, and archive the raw evidence
without newline conversion. The runner refuses to overwrite an existing run.

If the correction succeeds, it establishes intervention sufficiency only for
this frozen model and known scene. It does not mean the policy has learned the
action or that generalization has improved. If it fails, the first disagreement
alone is insufficient; no claim of policy repair should be made.

## Commands

Run from the repository root in the configured environment:

```bash
python -m pytest tests/test_scene10002_rescue.py -q
python docs/experiments/031_scene10002_rescue/run_rescue.py
```

Output: `results/evaluation/scene10002_rescue_031_20261116`.
The input full snapshot and final evidence are retained in Git for reproduction.
