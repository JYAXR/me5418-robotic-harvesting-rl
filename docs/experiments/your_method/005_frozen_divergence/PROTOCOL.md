# Own-route attempt 005: frozen first-divergence diagnosis

## Fixed scope

Diagnose the fixed attempt-004 targeted endpoint on seeds 290012, 290021,
290017 and 290022. The first two remain unsuccessful after supervised
reweighting; the latter two lost success relative to the frozen pre-training
source. No training, optimizer changes, forced actions or new recovery labels.

Branch: `experiment/your-method`; prior result commit: `0cd5a0a`.
Current policy: `models/your_method_004_targeted.pt`, SHA-256
`ea42fd1a2d23317c9e15a960c58d90970a38edbcfb7724308a845ead92f6119f`.
Full snapshot: `docs/experiments/your_method/004_verified_supervision/resume/targeted_state_14000.pt`,
SHA-256 `ed32355abf23c463a523df6bf00d6a9708b727bcd15fc04f357eaadd6f68fd2c`.
TD and Adam clocks stay 14000/34000. Retain all configuration and RNG state;
disable gradients and reject optimizer steps. CPU, one Torch thread.

## Comparisons

1. Against the unchanged stored demonstration: walk the common prefix, verify
   the encoded input hash at every step, and stop at the first action difference.
2. Against the pre-training attempt-034 targeted source: compare current and
   source trajectories only while their pre-action raw observations, encoded
   hashes, positions and environment RNG states agree. Stop at the first action
   difference. This directly describes the regression relative to earlier success.
3. Against the matched attempt-004 control: perform the same common-state
   comparison to distinguish continuation effects from the weight change.

For each fork retain the one-based step, common-prefix length, input hash,
chosen/reference actions, all eight Q values and their explicitly oriented
gap. Reference gap is Q_current(chosen) minus Q_current(reference). For model
comparisons, compute each model's preference using its own Q values at the
same input; never compare Q values across unequal post-fork states. A first
disagreement is not automatically an error or proof of failure causation.

## Frozen replay and evidence

Reproduce four full current-policy episodes from the complete snapshot using
fresh uncached resets and the normal evaluator with the audited trace wrapper.
Match the archived attempt-004 transitions exactly, including raw observations,
Q values, RNG states, actions, rewards, terminal outcomes and reset records.
Verify full snapshot preservation on close and inference/full-state agreement.
Reuse hash-verified source/control trajectories; no extra baseline simulations
are needed. Keep original environment, reward, 200-step budget and seed.

Archive all results, traces, model/data/source hashes, failure categories,
actual simulator/certification costs and time. Preserve partial evidence on
execution failure and do not silently restart or substitute seeds. No changes
to training parameters, no later action correction, and no unseen-scene tests.
Commit/push protocol and tested implementation before running; commit/push
outcomes afterward on our branch only. Code and records remain in English.

```bash
python -m pytest docs/experiments/your_method/005_frozen_divergence/test_diagnostics.py -q
python docs/experiments/your_method/005_frozen_divergence/run_diagnosis.py
```
