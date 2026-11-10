# Attempt 008: lower learning rates during TD adaptation

## Protocol recorded before execution

Attempt 007 reproduced the learned supervised route, but TD+margin at learning
rate 0.0003 lost it after one update, recovered temporarily and then failed again.
The supervised continuation control remained successful. Test whether reducing
the TD-phase step size preserves behavior over the same adaptation horizon.

Use the **verified attempt-007 supervised checkpoint after 20000 updates**, and
its identical frozen 35-transition demonstration dataset. Check the reference
source, checkpoint, encoded data, parameter and reproduction hashes. This is an
explicit transfer of pretrained weights followed by a fresh optimizer, not a
resume of optimizer state. No additional demonstration collection or supervised
pretraining is performed; their original costs remain in attempt 007.

Two new branches use **TD+margin** (margin 0.8, weight 1), with phase learning
rates **0.00003** and **0.000003**. The already-recorded 007 TD+margin branch at
0.0003 is the historical baseline, not a newly executed branch. Its runner and
production source hashes must still match. Everything except phase learning
rate remains fixed: architecture, inputs, environment, reward, data, full-batch
updates, optimizer type, clipping and target synchronization every 200 updates.

Apply the new rate before constructing fresh Adam; record the resolved rate in
each branch's settings and saved model. Synchronize the target from pretrained
online weights and start the phase counter at zero. Keep the inherited 20000
supervised updates separate from new updates. Require phase-zero weights,
whole-dataset action metrics and the greedy route to exactly match the 007
supervised baseline before updating. Use the unchanged attempt-007 phase runner.

Each new branch receives exactly **2000 TD+margin updates**, evaluated at
**0, 1, 10, 50, 100, 200, 500, 1000 and 2000**. Retain all snapshots, including
temporary failures/recoveries. Success at every sampled checkpoint is the stated
retention criterion; it does not establish behavior at unobserved updates.
There is no adaptive tuning, early stopping or selection of a successful
intermediate checkpoint. Verify actual parameter changes and update counts so
retention is not mistaken for simply freezing the policy.

The new budget is **4000 optimizer updates**, zero new collection actions, zero
online exploration steps and zero held-out scenes. Separately count scheduled
neural evaluations and post-run reload verification, excluding reset feasibility
probes. Reusing the existing dataset does not make these policies independent
of privileged planner-generated training data.

## Reproduction

With the verified reference run available and a new output directory:

```powershell
python docs/experiments/008_td_learning_rate/run_learning_rate_retention.py --reference-run results/training/staged_retention_20261111 --output-dir results/training/td_learning_rate_20261111
```

The input manifest identifies the exact reference and its immutable hashes.
Both new rates, all failure cases and the previously recorded higher-rate
baseline will remain visible in the result comparison. This experiment tests
retention on one known scene over 2000 updates, not convergence, optimal learning
rate, improvement on new tasks, online learning or multi-scene generalization.

## Outcomes

Pending the predeclared run. Preserve both rates regardless of outcomes.
