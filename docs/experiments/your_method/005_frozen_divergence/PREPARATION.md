# Preparation record

All five diagnostic tests passed on the first invocation. They check earliest
divergence, explicit gap orientation, rejection of unequal inputs/RNG states,
and refusal to align trajectories after their first differing action.

The initial Ruff finding requested a tuple-based startswith call. That
style-only change was made and Ruff passed. The full endpoint snapshot restored
and passed the exact unchanged-state close check. No simulator or optimizer
steps were executed during preparation, and the output directory was fresh.
