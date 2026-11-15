# Attempt 034: targeted supervision of three existing reference states

Start both arms from the exact final attempt-030 full snapshot retained in
attempt 031 inputs (SHA-256
1a63fba8f5c12b8cf13129b6dc290264c709d639e9a2b435d502d87d31ae7394).
The three intervention states from attempt 033 are already present in the
1440-row demonstration dataset: scene 10002 steps 23/27/28 correspond to
zero-based rows 98/102/103 and labels 4/0/4. Verify their encoded hashes against
the visited states in the successful corrected trace before training.

Only change their action-margin weights from 1 to 2 in the treatment. The
control retains weight 1. Keep the 1440-row mean denominator, all TD samples,
the existing row-40/55 weights of 2, recovery term divided by 117, margin 0.8,
TD coefficient 0.5, learning rate 0.0001, architecture, target interval 200,
gradient clipping, gamma, Adam moments/clocks and random states unchanged.
No MC constraint, new demonstration trajectory or new training scene is added.

Predeclare a bounded 2000-update continuation per arm, with no early stopping,
extension or model selection based on results. TD clock advances 10000 to 12000;
Adam clock advances 30000 to 32000. Do not resynchronize/reset anything at the
start. Evaluate scenes 10000/10001/10002 at relative updates
0, 10, 100, 500, 1000 and 2000. Every evaluation is fully greedy, with zero action
overrides. This budget tests acquisition and sampled retention, not long-term
stability or generalization. The two unchanged-weight and targeted arms receive
identical update budgets and evaluation seeds.

Save every update's losses, Q-margin probes, checkpoint outcomes and full-state
checkpoint hashes. Verify the first/last ten updates of each arm with the
independently assembled objective, comparing metrics and complete end states.
Verify both final inference exports on all three scenes using the normal
evaluator. Count training updates, verification updates, episode actions and
reset-certification simulation separately. Preserve both experimental endpoints
without replacing the existing policy, including if targeted supervision fails.
No held-out scene is evaluated or reused for training.

Commit/push protocol and implementation before training, then retain and push
all results, including failures. Raw evidence must retain exact bytes. Reports
and submission archives are outside this attempt's scope. All code and comments
remain English.

```bash
python -m pytest tests/test_scene10002_supervision.py -q
python docs/experiments/034_scene10002_supervision/run_training.py
```

Run from the repository root. Output directory:
results/training/scene10002_supervision_034_20261116.
