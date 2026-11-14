# Attempt 022: local supervision with TD coefficient 0.5

## Protocol declared before execution

Change only the supervised coefficient on original demonstration row 55
(scene 10001, one-based step 21, action 0 north_small) from 1 to 2.
Use 0.5 * mean(TD Huber) + mean(weighted demonstration hinges), with all
other 116 coefficients fixed at 1, hinge margin 0.8, and denominator 117.
Do not renormalize by weight sum, alter margin, duplicate data, add MC loss,
force actions, reset Adam, change learning rate, or modify production defaults.

Restore the same complete TD-2000 snapshot as attempt 019 and retain online,
target, Adam moments/steps, RNG and absolute clock. Train 8000 updates through
TD 10000 with learning rate 0.0001, gamma 0.99, clipping 10 and target period
200. No early stopping or checkpoint selection. Reuse attempt 019 as the
historical coefficient-one control, augmented by its attempt-020 physical
checks at every update 9946--9966. Audit both committed source archives and
all referenced snapshot/data hashes before starting.

Evaluate the same 29 original checkpoints plus all 21 updates 9946--9966:
50 physical checkpoints, three known scenes each, 150 autonomous episodes.
Record complete trajectories, failure, blocking, return and demonstration fit.
Probe all seven existing scene-10001 reference states after every update;
record full-batch action agreement on all 117 reference states as an additional
monitor. Full-batch predictions are not physical outcomes and may differ from
batch-one inference at numerical ties. Retain every failure and ranking flip.

Check initial behavior/Q equality. Save full snapshots at every evaluation and
every target synchronization. Test exact equivalence to attempt 019 when the
selected coefficient is 1, and independently compute the coefficient-2 update.
Independently replay TD 2001--2201 (201 updates), including target sync, and
compare logs, Q probes and saved full states. Independently reload TD 2000,
2009, 2200, 2201, 9955, 9956 and 10000 on all three scenes (21 episodes),
requiring exact trajectory and reset-certification equality. Audit later state
hashes, counters and target lineage without claiming full numerical replay.

Reset certification remains enabled and its simulated actions are counted
separately. Historical control and inherited training are reused, not rerun.
No held-out evaluation or new demonstration collection. Stable sampled success
and stable reference ranking are separate criteria; neither proves performance
at unsampled physical states. Record English code, all evidence and results in
Git. The user reports remain unchanged. Results pending at protocol commit.
