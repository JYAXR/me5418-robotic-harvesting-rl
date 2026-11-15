# Attempt 026: correct only scene 10001 step 6

## Protocol declared before execution: 15 November 2026

Use the frozen attempt-025 full snapshots at TD 2004 and 2009. Both fail scene
10001 after choosing east_small (2) instead of south_small (1) at the visited
reference state for step 6. Their later learned recovery capability remains
present. Compare a fresh greedy control with a single-action correction:
override only one-based step 6 with south_small, then immediately return to
the original neural policy. Do not force any other reference or recovery action.

Restore identical online/target weights, Adam moments and clocks, counters and
RNG state per arm. Keep seed 10001, environment, reward, action budget and all
training settings unchanged. Do not train or modify the frozen models. Require
exact reproduction of the historical full greedy outcome/actions/path, identical
first five full transitions and identical fork observation/Q/proposed action.
Check that the fork Q values match the archived dense reference probes.

Execute four main episodes and four independent reload episodes using the
original episode helper with a separately written single-action policy adapter.
Check every decision and full transition, full trajectory and final outcome.
The transition observer is installed after reset certification so copied reset
simulations remain unaffected. Count reset certification separately. Compare
complete training snapshots before and after every episode and recheck all
bound sources. Keep all results, whether correction succeeds or fails.

This is a frozen known-scene intervention, not an additional learned remedy,
a proof of a unique underlying cause or a generalization test. Record English
code, tests, full evidence and actual work, then synchronize to Git. Reports
and production defaults remain unchanged. Results pending at protocol commit.
