# Own-route attempt 003: add one correction with the first held fixed

## Fixed comparison

Keep the current targeted model frozen and retain the first correction from
attempt 002 in both arms. Add only the specified second correction in treatment.
After these scheduled interventions, all actions remain greedy model decisions
at the states actually reached. Do not add further corrections or train.

| Seed | Fixed correction in both arms | Only additional treatment correction |
| --- | --- | --- |
| 290012 | Step 19: east_large (6) | Step 26: north_large (4) |
| 290021 | Step 12: east_large (6) | Step 17: east_small (2) |

Steps are one-based. The control must exactly reproduce attempt 002's corrected
200-step timeout. At the new fork, policy choices should remain north_small (0)
and south_large (5), respectively, before overriding. This tests the additional
action's sufficiency conditional on the first correction, not either action
alone, learned recovery, or generalization.

## Inputs and unchanged conditions

Branch: `experiment/your-method`; source evidence commit: `2764dca`.
Inference model: `models/supervision_034_targeted.pt`, SHA-256
`202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b`.
Full source:
`docs/experiments/034_scene10002_supervision/checkpoints/targeted_state_12000.pt`,
SHA-256 `fe8e29438165cd4d99c6e143f819a37a2febc5e704a47efb9a895bb1074c610f`.

Restore full weights, Adam moments and RNG states. TD remains 12000 and Adam
remains 32000. Disable gradients and reject optimizer steps. Preserve environment,
rewards, observation/action definitions, seeds and 200-step budget. Use CPU with
one Torch thread and ordinary uncached resets, as in attempt 002. No new samples,
parameter tuning, test scenes or evaluation of scene 290007 is needed.

## Execution and required checks

Run two primary arms per seed, then reload and independently repeat both through
the normal evaluator: four primary and four verification episodes. Control has
one forced action per episode; treatment has two. Thus six primary and six
verification forced actions are planned. Count all episode and certification
simulation actions, including repeated resets.

Verify the second fork from unchanged demonstrations and historical corrected
traces before execution. Require exact historical control transitions, metrics,
geometry and certification; identical paired prefixes and fork inputs/Q values;
exact intervention locations/counts; and greedy actions everywhere else.
Independent replay must match every transition and endpoint. Full learner/Adam/
RNG state and pinned input hashes must remain unchanged. Preserve partial
evidence on any failure, with no scene replacement or silent retries.

Record success/failure type, return, steps, damage/deformation, travel, remaining
distance, blocked/stationary steps, resource counts and runtime. Later reference
disagreements may be described from saved traces but not corrected in this run.
A failed treatment means this additional correction is insufficient under these
conditions; a successful treatment is assisted rescue, not autonomous learning.

Commit and push protocol, implementation and audit tests before simulation.
Commit and push all outcomes and verification evidence afterwards, on our branch
only. Code and experiment records remain in English.

```bash
python -m pytest docs/experiments/your_method/003_second_action_rescue/test_rescue_audit.py -q
python docs/experiments/your_method/003_second_action_rescue/run_rescue.py
```
