# Attempt 023: complete episodes at step-22 ranking reversals

## Protocol declared before physical evaluation

Freeze attempt-022 models (TD coefficient 0.5, row-55 supervised weight 2,
all other weights 1, learning rate 0.0001, retained Adam, no MC objective).
Choose four negative-gap intervals using only the existing dense reference
log: first occurrence, global minimum gap, longest contiguous interval
(earliest on ties), and last occurrence. Include each entire interval and one
adjacent update on both sides. Do not select models based on new episode outcomes.

The recorded selection is 2031--2032, 2225--2229, 2344--2346 and 9847.
The global minimum is TD 2345 (-0.31208038330078125); the longest interval
contains five updates. Including surrounding checks gives 19 models and
57 autonomous episodes on scenes 10000, 10001 and 10002. Eleven selected
models have negative step-22 reference gaps; the other eight are positive-gap
neighbors. This samples 11 of 690 reversals, not every negative model.

Restore the full source snapshots at TD 2000, 2200 and 9800. Replay the three
200-update segments to TD 2200, 2400 and 10000, requiring exact raw update
logs, all seven reference Q vectors and full endpoint states. Independently
replay the same segments using the separately implemented objective and
compare all 19 selected full states: 1200 reconstruction updates total.
These recover already-trained models and do not extend training or change any
hyperparameter. Preserve all new full snapshots locally.

Run unchanged greedy episodes with no forced actions, recording all decisions,
encoded-state hashes and Q values. At actual steps 21 and 22, compare reached
states and Q vectors with the original demonstrations and dense records.
Separate first divergence, actual failure, successful alternate routes and
unvisited reference states. Preserve reset certification and count its copied
simulation actions separately. Require unchanged full training state after
all evaluations. Reload the four anchor models (2031, 2227, 2345, 9847) on all
three scenes using the original uninstrumented rollout helper: 12 additional
episodes, with exact trajectory and reset-certification comparisons.

Record failures, recoveries and limitations honestly. No action rescue or new
training remedy is part of this attempt. Results are targeted diagnostics on
known scenes, not an unbiased success rate or generalization test. Keep code
and comments English, leave reports unchanged, and commit the evidence to Git.
Results pending at the pre-execution commit.
