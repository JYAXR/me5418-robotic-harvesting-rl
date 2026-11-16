# Preparation record

Seven synthetic diagnostic tests passed on the first invocation (9.52 seconds).
They cover all four requested intervention locations, greedy control behavior,
rejection of an extra override, mismatched fork RNG and premature termination
before the intervention. Ruff passed on the first invocation.

The rollout instrumentation is adapted from attempt 035 and uses the exact
full-snapshot loader verified in attempt 005. Baseline traces retain the same
schema with two additional intervention-audit fields. Production training code,
models and prior evidence are unchanged. No physical episodes were run during
preparation.
