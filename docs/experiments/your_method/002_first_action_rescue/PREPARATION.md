# Preparation record

The first unit-test invocation had nine passes and three failures. All three
failures were in the deliberately changed-Q-value rejection cases: the reused
strict comparison correctly rejected the mismatch with ValueError, while the
new tests expected AssertionError. The test expectation was corrected to the
actual exception type and diagnostic message; no rollout or policy logic was
changed. These checks use synthetic traces and execute no simulator or optimizer
steps. The corrected test run and frozen-input checks are recorded before the
primary experiment. No physical experiment has been restarted by this change.

Final preparation: all 12 tests passed; Ruff passed. All three historical
first disagreements and the exact full-state restore/preservation check passed
with no environment reset, simulator step or optimizer update. The output
directory did not exist before execution.
