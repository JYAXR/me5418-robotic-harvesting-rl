# Attempt 025: supervise one observed off-reference recovery action

## Protocol declared before execution: 15 November 2026

Resume the same complete TD-2000 snapshot used by attempt 022 through absolute
TD 10000 (8000 updates), preserving Adam moments/clocks, target lineage and RNG.
Keep learning rate 0.0001, TD coefficient 0.5, original supervision coefficient
1, original row 55 (scene 10001 step 21) weight 2, margin 0.8, denominator 117,
gamma 0.99, target interval 200 and clipping 10. No MC constraint, online TD data,
new pretraining, early stopping, or production-default change.

The only new objective term is hinge(Q(recovery_state), north_large) / 117.
Recover this actual encoded state by executing the frozen TD-2031 policy's first
24 actions on seed 10001 and checking every observation against attempt 024.
The state must match the step-25 hash shared by all five rescued failures.
Its label is action 4 from the successful intervention, not the demonstration's
action at time index 25. Preserve the original 117 TD transitions and their
loss reduction exactly. Do not divide the old supervised loss by 118 or add a
fabricated reward, next state or terminal flag for the recovery observation.

Compare with the immutable attempt-022 baseline, including attempt-023's 19
neighbor checkpoints: 69 matched checkpoints, three autonomous full episodes
per checkpoint. Replay a fixed 24-action prefix and then let the network choose
every action from step 25 onward at all 69 treatment checkpoints. These
conditional recovery probes are explicitly separate from autonomous episodes;
their scripted prefix is counted, and there is no forced recovery action.
Run the same conditional probe on baseline TD 2000, 2031, 2226, 2344, 2345,
2346 and 10000. The baseline's other full-episode results are historical, not
new simulator work. Probe the recovery-state Q values and the seven existing
reference sentinels at every update. Keep all successes and failures.

Before treatment, verify one zero-addition update exactly against attempt 022
and one treatment update against independent objective math. Then independently
replay treatment updates 2001--2201 and 9801--10000, checking logs and complete
saved states. Audit every saved target/Adam clock. Reload the seven baseline
probe checkpoints above and verify all three treatment full episodes plus its
conditional recovery trajectory (28 episodes). Preserve reset certification
and account separately for all collection, scripted-prefix, policy-tail,
verification and copied-simulator work. No held-out scenes are accessed.

This is a fixed-state training diagnostic on three known scenes, not evidence
of generalization or continuous episode success between sampled checkpoints.
A positive recovery-state Q gap is not by itself an episode-success claim.
Record English code, tests, provenance and complete outcomes, then synchronize
the attempt to Git. Reports remain unchanged. Results pending at protocol commit.
