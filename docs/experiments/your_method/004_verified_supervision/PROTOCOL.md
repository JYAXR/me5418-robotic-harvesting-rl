# Own-route attempt 004: reinforce five verified actions under unchanged TD

## Question and exact intervention

Can the policy autonomously recover the three regressed scenes after reinforcing
their verified correction labels, without losing competence elsewhere in the
33-scene training bank? Compare two equal-budget continuations of the same full
attempt-034 targeted snapshot. No forced actions are allowed during evaluation.

| Scene | One-based step | Existing zero-based row | Verified label |
| --- | ---: | ---: | --- |
| 290007 | 8 | 430 | east_large (6) |
| 290012 | 19 | 653 | east_large (6) |
| 290012 | 26 | 660 | north_large (4) |
| 290021 | 12 | 1052 | east_large (6) |
| 290021 | 17 | 1057 | east_small (2) |

Require exact encoded-state hashes and labels to match the successful corrected
rollouts from own-route attempts 002/003 before execution. These states already
exist in the 1440-transition dataset. Control retains weight 1 on these five
rows; treatment changes only their action-margin supervision weights to 2.
Thus the added loss is the sum of their five existing hinge terms divided by
1440. No transitions are appended or duplicated in TD; no reward/next-state or
termination target is altered. Previous weighted rows 40/55/98/102/103 stay at 2,
and the separate existing recovery term keeps its /117 normalization.

## Frozen starting inputs and unchanged training settings

Branch: `experiment/your-method`; previous evidence commit: `98ed9b9`.
Source snapshot:
`docs/experiments/034_scene10002_supervision/checkpoints/targeted_state_12000.pt`,
SHA-256 `fe8e29438165cd4d99c6e143f819a37a2febc5e704a47efb9a895bb1074c610f`.
Matching inference model: `models/supervision_034_targeted.pt`, SHA-256
`202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b`.

Both arms retain Adam moments, RNG and TD clock, and perform exactly 2000 new
full-batch updates, the same continuation budget used in attempt 034. TD advances
12000 to 14000; Adam advances 32000 to 34000. Preserve architecture 341-256-256-8,
learning rate 0.0001, TD coefficient 0.5, global supervision coefficient 1,
margin 0.8, gamma 0.99, gradient clipping 10, target synchronization every 200
updates, dataset order, all existing weights and environment configuration.
No MC term, optimizer reset, new scene collection or online interaction training.
CPU with one Torch thread. No outcome-driven budget or coefficient changes.

## Autonomous evaluation and retention criteria

The complete baseline is the frozen attempt-001 targeted evaluation (10/33,
including all three original scenes). Verify its model and all simulator/agent
source hashes, then freshly reproduce six baseline episodes: 10000/10001/10002
and 290007/290012/290021. Reuse the existing full-bank evidence without rerunning
the other 27 unchanged baseline scenes.

For each arm, evaluate those six scenes after 10, 100, 500 and 1000 additional
updates. At the fixed 2000-update endpoint, evaluate all 33 seeds: 10000--10002
and 290000--290029. Then independently reload both final inference exports and
re-evaluate the six diagnostic scenes through the normal evaluator. Planned
work: 120 primary episodes and 12 endpoint reloads, all with zero overrides.
Do not choose a best intermediate model. No unseen or final-test scenes.

Use the verified process-local certified scene cache across primary evaluations,
and require each initial reset record to match the historical seed exactly.
Cache remains confined to evaluation; include actual certification work and
timing. Reload checks use fresh uncached resets. Preserve training state and
Torch RNG across evaluation, including inference-model construction.

Report all outcomes and per-scene gains/losses relative to the frozen source
and matched control. Separately report original-three success, targeted-three
success, all-33 success, damage, timeout, return and successful-episode steps.
Do not call a higher aggregate count 'no regression' if earlier successes are
lost. Sampled retention is not uninterrupted stability or generalization.

## Verification, artifacts and stopping rule

Before the main run, test that only five supervised gradients change, that a
control update exactly matches attempt 034, and that a treatment update matches
an independent implementation, including full optimizer/RNG state. Count the four
unit-test optimizer updates separately. During the run, independently replay the
first and last ten updates of each arm (40 extra updates) and require exact
metrics and complete-state matches. Preserve pinned file hashes and validate
all final clocks, target synchronizations, greedy actions and endpoint reloads.

Retain all update logs, checkpoint probes (five labels and adjacent rows),
compressed full evaluation traces, outcomes, failures, provenance, segmented
timing and actual resource counts. Archive both final inference candidates and
complete resumable endpoints; do not overwrite the current candidate. Keep
intermediate full snapshots locally. Stop with partial evidence on unexpected
execution/verification failure; do not silently restart or substitute scenes.

Commit and push the protocol/implementation before running. Record and push
all results afterward, including unsuccessful learning. Code and experiment
records remain in English. Only our branch is updated.

```bash
python -m pytest docs/experiments/your_method/004_verified_supervision/test_focused_update.py -q
python docs/experiments/your_method/004_verified_supervision/run_training.py
```
