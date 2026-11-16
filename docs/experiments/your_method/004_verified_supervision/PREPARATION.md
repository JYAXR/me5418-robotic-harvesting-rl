# Preparation record

Both substantive tests passed on their first invocation. The gradient test
confirms that only rows 430/653/660/1052/1057 change. The exact-update test checks
the unchanged control against attempt 034 and treatment against an independent
loss implementation, including complete training state. It performs four
optimizer updates on disposable restored agents, with zero simulator actions.
All five correction labels and encoded input hashes match the successful traces.

The first Ruff invocation reported an unused dataset metadata variable. The
metadata is now included in provenance; this changed no objective or training
behavior. No main experiment was started or restarted during preparation.
