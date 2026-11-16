# Attempt 007: first divergence after the fixed step-5 correction

Branch: `experiment/your-method`. Parent result: `d590929` (attempt 006).
Commit and push the protocol before running the diagnosis.

## Scope

Keep the attempt-004 targeted endpoint frozen. Run seed 290017 with exactly one
override: step 5 becomes south_large (action 5). All other actions are greedy.
Find the first subsequent action disagreement separately against:

1. The pre-training attempt-034 targeted source, successful in 35 steps.
2. The matched attempt-004 control, successful in 41 steps.

Use one fresh, uncached complete corrected episode and require exact agreement
with the archived attempt-006 corrected trace and metrics. Reuse the two existing
successful traces; do not rerun their simulations or force another correction.

## Frozen conditions and comparison

Use `models/your_method_004_targeted.pt`, SHA256
`ea42fd1a2d23317c9e15a960c58d90970a38edbcfb7724308a845ead92f6119f`,
and the matching full attempt-004 snapshot `resume/targeted_state_14000.pt`,
SHA256 `ed32355abf23c463a523df6bf00d6a9708b727bcd15fc04f357eaadd6f68fd2c`.
Keep TD clock 14000, Adam clock 34000, network/optimizer/RNG state and all
environment and training parameters unchanged. Use CPU, one Torch thread,
no gradients or optimizer steps, and the existing 200-step intrinsic budget.

Reuse the exact loader and intervention rollout from attempt 006 and the
first-policy-fork comparator from attempt 005. The forced step must be exactly
step 5; all other actions must match the current Q argmax. Before any reported
fork, require matching actions and transitions, raw and encoded observations,
position and RNG. At the fork, require matching pre-action inputs. Never align
different post-fork states merely because their step numbers match.

Require the first executed-action divergence to occur after step 5. Report its
one-based step, both actions, both full Q vectors, and explicit gaps:
Q_current(current action) - Q_current(successful-policy action), plus
Q_baseline(baseline action) - Q_baseline(current action).
If the two successful policies yield different forks, report both rather than
selecting whichever seems more favorable.

Pin historical trace/model/source hashes, record all costs and timings, verify
the full frozen state on close, and preserve partial evidence on any failure.
No new training samples, optimizer updates, extra intervention or held-out
evaluation are included. A disagreement is a candidate for later intervention;
this diagnostic alone cannot establish that correcting it restores success.

Run from the repository root with the project interpreter:
`docs/experiments/your_method/007_post_correction_divergence/run_diagnosis.py`.
The evidence directory must not already exist.
