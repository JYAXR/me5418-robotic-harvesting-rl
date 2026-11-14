# Attempt 020: complete episodes around the late Q-ranking reversal

## Protocol declared before execution

Evaluate every attempt-019 treatment model from TD 9946 through 9966 inclusive
on known scenes 10000, 10001 and 10002 (63 autonomous complete episodes).
The existing dense log identifies negative step-21 gaps at TD 9955 and 9956;
9957 is positive. This is a targeted retrospective diagnostic, not an unbiased
estimate of the frequency of failure or a held-out generalization test.

Restore the unchanged full TD-9500 snapshot and replay 500 updates through
TD 10000 with TD coefficient 0.5, supervised coefficient 1, learning rate
0.0001, retained Adam, original targets/counters/data and no MC constraint.
Compare every replayed loss record and all seven reference Q vectors with
attempt 019, and require full endpoint equality before physical evaluation.
Retain all 21 reconstructed full snapshots locally. A separate implementation
replays the same 500 updates and compares all 21 snapshots and the endpoint.
These 1000 replay updates reconstruct existing models; they are not a new
training remedy or additional updates on the original branch.

Use the original greedy rollout helper and unchanged environment settings,
including reset certification and the 200-step budget. Observe all actual
policy decisions, full encoded-state matches, and Q values without replacing
any action. Record the first action divergence and whether the actual step-21
state equals the demonstrated state. Preserve all complete trajectories,
returns, failures, damage, blocking and recovery outcomes. Separately count
policy and reset-certification actions. No online learning or action rescue.

Independently reload TD 9955, 9956 and 9957 on all three scenes (nine additional
complete episodes), compare exact trajectories and certification checks, and
verify evaluation does not change weights, targets, Adam, counters or RNG.
A failure after a matching-prefix divergence establishes a behavioral failure
at that model; it does not isolate the action's causal effect without an
intervention. Do not claim success at untested updates.

Record source hashes, local model hashes, full observations, verification and
work accounting. Keep binary checkpoints and raw logs local; commit compact
JSON evidence and this English record to Git. Existing code, results, reports
and production defaults remain unchanged. Results pending at protocol commit.
