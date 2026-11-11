# Experiment record

This directory retains small, reviewable evidence for each development attempt,
including unsuccessful attempts. Model weights, full trajectories and console
logs remain under the ignored `results/` directory. These records are development
evidence for the final report, not a claim that a final trained model is ready.

| Attempt | Question | Observed outcome |
| --- | --- | --- |
| [001: multi-route DQN from scratch](001_multiroute_from_scratch.md) | Can the unchanged DQN and reward solve the new multi-route scene in a bounded pilot? | 0/30 training successes; 0/3 final validation successes; reference planner 3/3. |
| [002: reference replay initialization](002_reference_replay.md) | Do ten training-only reference trajectories help with the same online episode budget? | 421 demonstration transitions; 2/30 online training successes; 0/3 final validation successes, with damage in 1/3. No reliable improvement established. |
| [003: demonstration TD pretraining](003_demonstration_pretraining.md) | Do 2000 additional DQN updates on the same demonstrations help before online training? | 2000 pretraining + 880 online updates; 3/30 training successes; post-pretraining and final validation both 0/3 successes and 2/3 damage. No benefit established; final return deteriorated. |
| [004: supervised demonstration pretraining](004_supervised_pretraining.md) | Does an action-margin objective improve the same pretraining and online schedule? | 2000 pretraining + 1033 online updates; 2/30 training successes; all greedy validation stages 0/3 successes. Final damage 1/3; no task-success benefit established. |
| [005: fixed-scene fitting diagnostic](005_fixed_scene_fit.md) | Can the existing encoder and MLP fit one fixed multi-route demonstration under TD+margin, margin-only or cross-entropy? | After 2000 updates each: 19/35, 25/35 and 25/35 action matches; all same-scene rollouts time out. Underfitting persists without TD. |
| [006: fixed-scene optimization budget](006_fixed_scene_budget.md) | Does ten times the update budget resolve the single-scene underfitting with otherwise identical settings? | At 20000 updates, margin-only and CE fit 35/35 actions and succeed in 35 steps. TD+margin fits 33/35 but fails by damage. Fixed-scene memorization demonstrated; generalization untested. |
| [007: staged supervision and TD retention](007_staged_retention.md) | After supervised fitting succeeds, do TD-only or TD-plus-margin updates preserve the known route? | Supervised control retains 35-step success throughout. Both TD branches fail after one update; TD+margin recovers temporarily but ends in damage. Retention is unstable at learning rate 0.0003. |
| [008: lower TD adaptation learning rates](008_td_learning_rate.md) | Can a smaller learning rate preserve the supervised behavior during the same TD-plus-margin phase? | Both 0.00003 and 0.000003 retain sampled success through 200 updates, but fail at 500, 1000 and 2000. Smaller steps postpone observed failure without resolving retention. |
| [009: return calibration before TD](009_return_calibration.md) | Does fitting actual discounted demonstration returns before TD reduce forgetting compared with equal-budget action supervision? | MC RMSE falls from 83.1175 to 0.2747. Calibrated TD still fails at updates 1/10, then succeeds at every sampled 50--2000 checkpoint; matched control ends in a timeout. Later deterioration improves, but immediate retention remains unresolved. |
| [010: Adam state at the TD boundary](010_adam_state.md) | Does preserving calibration Adam moments and step counts reduce the early TD failures compared with a matched reset? | Calibration and reset control exactly reproduce 009. Preserved Adam succeeds at all first-ten updates and all 18 sampled checkpoints through TD2000; reset succeeds at 2/10 early checks and 9/18 overall. Both end in 35-step success. Early retention improves in this one known scene; generalization remains untested. |
| [011: three-scene staged learning and retention](011_multiscene_retention.md) | Can one network acquire three fixed training routes, calibrate their actual returns and retain acquired behavior when TD preserves Adam state? | Supervision/calibration both finish at 117/117 action matches and 3/3 scene successes. Preserved Adam has all-scene success at 6/10 early and 13/18 total TD checks, versus reset 0/10 and 5/18. Seed 10001 still fails at preserved updates 1, 6, 9, 10 and 200; both final policies succeed on all three scenes, but neither retains all scenes throughout sampled TD checks. Calibration also has temporary failures; generalization untested. |
| [012: first-divergence diagnosis and TD supervision weight](012_divergence_control.md) | Do missed turns explain scene 10001 failures, and can doubling only TD action-supervision weight reduce them? | Correcting one first-divergent action rescues all five original failures. Weight 2 raises early all-scene success from 6/10 to 8/10 and total from 13/18 to 14/18, but individual scene success falls from 49/54 to 48/54 and all three fail at update 50. Both final models succeed; neither retains all scenes throughout. Stronger supervision is not a stable retention fix. |

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
