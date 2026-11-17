# Attempt 008: bounded batch recovery and supervised-only learning

Both rounds are complete and the experiment has stopped at its two-round limit.
The supervised candidates improve observed validation successes from 3/30 to
5/30, but neither meets the predeclared retention rule. Round 1 loses training
coverage; round 2 loses original scene 10000. Retain the source endpoint as the
current baseline, while preserving both new models and all data for comparison.
No third round, additional single-step repair or TD stage was executed.

| Endpoint | Original three | Training success | Validation success | Training damage / budget failures | Validation damage / budget failures |
| --- | --- | --- | --- | --- | --- |
| Source attempt-004 targeted | 3/3 | 10/33 | 3/30 | 11 / 12 | 14 / 13 |
| Round 1, 6000 supervised updates | 3/3 | 9/33 | 5/30 | 8 / 16 | 12 / 13 |
| Round 2, 12000 cumulative supervised updates | 2/3 | 11/33 | 5/30 | 10 / 12 | 15 / 10 |

The source training results are reused from the previously audited 33-scene
evaluation; its validation bank was evaluated freshly here. All candidate
training/validation episodes are autonomous: no planner or action replacement
is used in evaluation. There are no reset errors in either round. Validation is
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
is below the source's 10/33. Its results, checkpoint and evidence were pushed
as `a536f4f` before the second round began.

## Round 2 and final selection

The second round collected from the round-1 policy under exactly the same
query schedule and training settings. Of 91 reached queries, 70 successful
routes yielded 2132 new samples; 21 planner failures yielded no labels. Eight
scheduled queries were skipped after early prefix termination. Round 1 had no
accepted recovery samples for 290024; round 2 had none for 290020. Every scene
still retains its original demonstration in the base pool.

The final dataset includes 1441 base examples and 4103 aggregated recovery
examples, for 5544 total rows. Sampling still draws half of each batch from
each pool. The second round performed another 6000 supervised updates, retaining
round-1 Adam moments and sampling RNG exactly. Adam reaches step 12000, while
the TD clock remains 14000 and target weights remain unchanged. Final fitting
accuracy is 79.5975% on base labels and 68.1209% on aggregated recovery labels.
These numbers do not by themselves explain a failure mechanism, but they show
that the network does not yet reliably reproduce the collected labels.

Round 2 raises training success to 11/33, but scene 10000 now exhausts its
200-step budget; 10001 and 10002 remain successful. Relative to the source,
seven training scenes gain success and six lose it, including 10000. Both
rounds gain four validation successes and lose two source successes; the net
gain of two should not conceal those regressions. Scene lists are preserved
in [comparison.json](comparison.json).

Round-2 validation also has 15 damage failures versus source 14 and round-1 12.
Its mean validation return is -33.106817, compared with source -31.086667 and
round-1 -27.524096. Thus the small success-count increase is not a consistent
improvement across metrics. Neither endpoint reaches the provisional 27/33
training and 9/30 validation targets.

Under the predeclared rule, a replacement must retain original 3/3 and at least
source 10/33 training successes before validation ranking. Round 1 fails the
training condition; round 2 fails original-scene retention. The selected
retained endpoint is therefore `models/your_method_004_targeted.pt`. Both new
models remain available as `models/your_method_008_round1.pt` and
`models/your_method_008_round2.pt`; their SHA256 values and selection eligibility
are in [comparison.json](comparison.json). This is a selection decision, not
a claim that the source is adequate for final submission.

## Combined resources, checks and limitations

Round 2 took 195.448 seconds. It performed 1187 policy-prefix, 2132 recovery,
6829 evaluation and 3740 certification actions, with 88 certification calls
and eight cache hits. Its audit verified 231 manifest files, 296 pinned inputs,
all 63 fresh evaluation episodes, 6829 greedy actions, 2132 new labels and all
6000 updates. The round-2 initial Adam summary exactly matches round-1 final
Adam state; inference exports match their saved resume weights.

Across both rounds there are 178 recovery queries, 140 successful routes, 4103
new recovery labels, 12000 supervised updates and zero new TD updates. All 38
failed recovery queries are recorded as planner-no-route, not task infeasibility.
There are 156 fresh autonomous evaluation episodes (30 baseline validation plus
63 per round), 32448 simulator actions including certification and recovery,
198 certification calls and 24 cache hits. Main run durations sum to 470.864
seconds (about 7.85 minutes), excluding preparation, post-run auditing and Git
synchronization. No physical experiment was restarted or silently discarded.

Using Asia/Shanghai client time, round 1 ran on 17 November 2026 from 15:12:35
to 15:17:11 and round 2 from 15:19:33 to 15:22:49. UTC timestamps are retained
in each round's status file. All evaluation failures, planner failures and
regressions are included in the corresponding denominators and records.

This bounded attempt demonstrates that removing TD and adding batch recovery
examples alone did not reliably preserve and improve the current policy under
these settings. It does not identify a unique cause: optimization, alternative
route labels, state coverage and the unchanged representation remain possible
contributors. Only one training random seed and 30 development-validation
scenes were used. No final untouched test set was consumed. Further work, if
requested separately, should reassess the broader data/model formulation rather
than resume an unlimited sequence of hand-selected action repairs.

Evidence: [protocol](PROTOCOL.md), [round-1 summary](round_1/summary.json),
[collection](round_1/collection.json), [training](round_1/training.json),
[evaluation](round_1/evaluations.json), [verification](verification_round1.json),
[manifest](round_1/manifest.json), [round-2 summary](round_2/summary.json),
[round-2 collection](round_2/collection.json), [round-2 training](round_2/training.json),
[round-2 evaluation](round_2/evaluations.json), [round-2 verification](verification_round2.json),
[round-2 manifest](round_2/manifest.json). All scene outcomes and compressed trajectories
are retained, including collection failures and policy regressions.
