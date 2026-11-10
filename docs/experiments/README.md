# Experiment record

This directory retains small, reviewable evidence for each development attempt,
including unsuccessful attempts. Model weights, full trajectories and console
logs remain under the ignored `results/` directory. These records are development
evidence for the final report, not a claim that a final trained model is ready.

| Attempt | Question | Observed outcome |
| --- | --- | --- |
| [001: multi-route DQN from scratch](001_multiroute_from_scratch.md) | Can the unchanged DQN and reward solve the new multi-route scene in a bounded pilot? | 0/30 training successes; 0/3 final validation successes; reference planner 3/3. |
| [002: reference replay initialization](002_reference_replay.md) | Do ten training-only reference trajectories help with the same online episode budget? | 421 demonstration transitions; 2/30 online training successes; 0/3 final validation successes, with damage in 1/3. No reliable improvement established. |
| [003: demonstration TD pretraining](003_demonstration_pretraining.md) | Do 2000 additional DQN updates on the same demonstrations help before online training? | Protocol prepared; results pending execution. |

For each new attempt, retain the motivation, exact commands, resolved settings,
training and validation seed ranges, source hashes, resource/data budget,
metrics, failure cases and interpretation. Keep failed attempts in the index.
Commit the implementation and evidence for each attempt separately where
practical, and push the commit after local verification. Avoid overwriting an
earlier run directory or rewriting earlier results after changing code.

Separate measured observations from explanations that still need testing.
Report any planner-generated training data and its simulation cost separately
from online agent interactions. Keep validation data out of replay and tuning
results separate from final held-out test results. A short successful smoke test
only verifies execution; it does not demonstrate reliable learning.

The first record was reconstructed from retained artifacts after its run.
Its provenance file explicitly identifies historical information that was not
captured at launch. Subsequent attempts should capture it automatically.
