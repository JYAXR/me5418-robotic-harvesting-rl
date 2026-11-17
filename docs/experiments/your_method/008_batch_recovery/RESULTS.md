# Attempt 008: bounded batch recovery and supervised-only learning

Round 1 is complete. It preserves the original three scenes but does not yet
improve both training coverage and validation performance. The predeclared early
stop condition is not met, so the second and final round is authorized by the
protocol. No third round or single-step repair will be added.

| Endpoint | Original three | Training success | Validation success | Training damage / budget failures | Validation damage / budget failures |
| --- | --- | --- | --- | --- | --- |
| Source attempt-004 targeted | 3/3 | 10/33 | 3/30 | 11 / 12 | 14 / 13 |
| Round 1, 6000 supervised updates | 3/3 | 9/33 | 5/30 | 8 / 16 | 12 / 13 |

The source training results are reused from the previously audited 33-scene
evaluation; its validation bank was evaluated freshly here. All candidate
training/validation episodes are autonomous: no planner or action replacement
is used in evaluation. There are no reset errors in this round. Validation is
independent of training data but is a development/model-selection set, not a
final untouched test. Two additional validation successes do not establish a
reliable generalization improvement on this small bank.

## Round 1

Protocol/code commit: `0f3bfb7`. All 33 training scenes were processed. Of the
maximum 99 scheduled queries, 87 were reached; the remaining 12 were skipped
because the policy prefix had already terminated. Seventy recovery routes
succeeded under actual cloned-state physics, producing 1971 accepted state/action
labels. Seventeen queries did not yield accepted routes and remain in the
collection records. No validation labels were collected.

The supervised dataset consists of 1441 base examples (1440 original rows plus
one legacy verified recovery) and 1971 new recovery examples. Each update uses
128 base and 128 recovery samples with replacement. After 6000 updates,
base-label accuracy is 80.6384% and recovery-label accuracy is 69.1527%. These
are data-fitting measurements, not task success rates. They show that fitting
the available labels is itself still incomplete. The existing architecture and
training budget are not changed in response to these results.

Adam was reset once for the new cross-entropy objective and has 6000 steps at
the endpoint. The inherited TD clock stays 14000; the target network remains
unchanged. No TD or MC loss was evaluated. The output values are now action
scores, not calibrated Q estimates. The round-1 optimizer, sampling RNG and
weights are archived for exact continuation into the final round.

The run took 275.416 seconds overall, using four independent simulation workers.
It performed 1182 policy-prefix actions, 1971 recovery actions, 7454 candidate
evaluation actions, 3324 baseline-validation actions and 4629 certification
actions (18560 simulator actions total). There were 110 certification calls
and 16 cache hits. Cache assignment affects wall time, so this is not a
controlled speedup measurement.

Post-run verification checked 258 manifest files, 35 pinned inputs, all 93 fresh
evaluation episodes and 10778 greedy evaluation actions, all 1971 recovery labels,
and all 6000 supervised updates. Returns, terminal outcomes, blocked/stationary
counts, source hashes, target preservation and model/resume weights passed.
There was no experimental restart or execution failure. Preparation test failures
and fixes are disclosed separately in [PREPARATION.md](PREPARATION.md).

The round-1 endpoint is retained as experimental evidence but is not eligible
to replace the source under the predeclared retention rule: training success
is below the source's 10/33. The second round will collect from this endpoint,
aggregate both rounds' successful recovery routes, retain its Adam state and
apply exactly 6000 more supervised updates with unchanged settings.

Evidence: [protocol](PROTOCOL.md), [round-1 summary](round_1/summary.json),
[collection](round_1/collection.json), [training](round_1/training.json),
[evaluation](round_1/evaluations.json), [verification](verification_round1.json),
[manifest](round_1/manifest.json). All scene outcomes and compressed trajectories
are retained, including collection failures and policy regressions.
