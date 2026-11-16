# Own-route attempt 001: frozen training-coverage audit

## Question and fixed scope

Does the final attempt-034 targeted policy retain competence beyond the three
original scenes? Compare it with the matched attempt-034 control on every one
of the 33 training seeds before selecting another training intervention.
This is a training-set diagnostic, not a generalization test or a comparison
with the teammate's future method.

Branch: `experiment/your-method`. Shared baseline: `809e7d8`.
No policy, environment, training hyperparameter, dataset or optimizer changes.
Zero optimizer updates and zero forced actions. Use CPU with one Torch thread.

## Models and scenes

- Targeted: `models/supervision_034_targeted.pt`, SHA-256
  `202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b`.
- Control: `models/supervision_034_control.pt`, SHA-256
  `a8a519b1944afec3bd5ac063b34db6c75136c1661c3c4f8fbc3412859e58e018`.
- Evaluate seeds 10000, 10001, 10002, then 290000 through 290029, once per arm:
  exactly 66 planned autonomous episodes. No fresh validation or final-test seeds.
- Alternate which arm runs first for each scene. Share one certified cache of
  capacity 33; verify identical complete reset records within every scene pair.
  Cache is an execution optimization, not a policy change or a speed benchmark.
- Use the existing normal evaluator and the audited attempt-035 trace wrapper.
  Retain raw observations, action Q values, actions, rewards, termination flags,
  geometry, RNG state, reset certification work and segmented timing.

## Verification and analysis

Require unchanged learner/optimizer/RNG state around each episode, greedy
argmax actions throughout, and unchanged input/model hashes after the run.
Require exact agreement with attempt-034 final metrics on all six original-scene
episodes. Preserve per-scene outcomes and paired success gains/losses. Report
original, added and combined training success; damage and budget failures;
successful episode length; reward; blocked/stationary steps; actual simulator
work; elapsed time. Timing includes trace instrumentation and is not a pure
inference latency measurement. Report descriptive stall signatures, not causal
claims. Stop and retain partial evidence if execution or verification fails;
do not replace scenes or silently rerun failures.

Compress each trace losslessly and verify decompression before retaining it.
Keep the full records, hashes and status in this attempt's evidence directory.
The script refuses to overwrite existing evidence. Commit and push this protocol
and runner before execution; commit and push results separately afterwards.

```bash
python docs/experiments/your_method/001_training_coverage/run_audit.py
```

## Decision rule for subsequent work

If many trained scenes fail, prioritize broad training competence and inspect
representative failure trajectories before another isolated action-weight fix.
If most trained scenes succeed, next measure fresh development-validation
coverage, keeping the eventual shared final test untouched. Any subsequent
training experiment needs its own predeclared protocol and matched comparison.
