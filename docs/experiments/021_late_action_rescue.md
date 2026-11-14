# Attempt 021: a single-action rescue of the late failures

## Protocol declared before execution

Use the exact frozen TD-9955 and TD-9956 full snapshots reconstructed and
verified in attempt 020. For scene 10001, independently restore the same model
and reset seed for each arm: autonomous greedy control, then override only
one-based step 21 with action 0 (`north_small`). The model still selects action
4 (`north_large`) at this state. Immediately resume its own greedy decisions
after the override. Do not force any other action, retrain, recalibrate, reset
Adam, modify targets, environment, reward, budget or network configuration.

Reuse the tested attempt-015 trace and paired-prefix verifier. Require exact
reproduction of the original control trajectories, identical first 20 complete
transitions, and identical fork observation and Q values. Count exactly one
forced action per corrected episode. Preserve the entire training snapshot,
including optimizer moments, counters and RNG, before and after every episode.
Keep reset route certification enabled and count its simulator actions separately.

Run four main episodes, then four fresh verification episodes with a separately
written execution loop. Compare every transition and final outcome with the main
traces. Preserve all outcomes regardless of success. There are zero optimizer
updates and no new model training; the diagnostic intervention is not a deployed
policy improvement. Success would establish that this one external correction
is sufficient for these two deterministic failures, not that all failures or
unseen layouts are fixed.

Record full trajectories, fork Q values, blocked/stationary behavior, returns,
source/model hashes, verification and actual-work accounting. Archive compact
JSON evidence and English notes in Git; keep existing reports unchanged.
Results pending at the pre-execution commit.
