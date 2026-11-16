# Independent method development

Branch: `experiment/your-method`. Shared starting point: `809e7d8`.
This route develops independently from `collab/teammate-20261116`.
Final method comparison will require a common untouched test bank and explicit
resource accounting. Development evaluations are not final test results.

| Attempt | Question | Status |
| --- | --- | --- |
| [001: frozen training coverage](001_training_coverage/RESULTS.md) | Does attempt-034 recovery extend across all 33 training scenes? | Targeted 10/33 versus control 9/33; both 7/30 on added scenes. Four gains and three losses; targeted has 16 damage and 7 budget failures. All 66 traces verified, no training updates. Broad training competence remains limited. |
| [002: first-disagreement single-action rescue](002_first_action_rescue/RESULTS.md) | Does correcting only the first reference disagreement rescue 290007, 290012 and 290021? | Only 290007 recovers (124-step damage to 30-step success). 290012 still times out; 290021 changes from damage to timeout. Six independent replays match, frozen full states unchanged, zero training updates. Later disagreements are untested candidates. |
| [003: second correction with the first fixed](003_second_action_rescue/RESULTS.md) | Does adding step-26 north_large on 290012 or step-17 east_small on 290021 rescue the one-correction failures? | Both recover: 200-step control timeouts become 62/46-step successes. Four independent replays match complete trajectories; full model/Adam/RNG state stays frozen. Conditional assisted rescue, with no learning or generalization claim. |
