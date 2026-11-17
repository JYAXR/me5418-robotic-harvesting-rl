# Preparation and fixes before the experiment

The first test invocation had five failures and three passes. Three failures
came from a synthetic fixture with zero branches, which the existing physics
does not support; the fixture now includes one distant branch. Two exposed a
runner hashing error: tensors were sent directly to a JSON digest without the
existing canonical tensor conversion. The runner now applies that conversion.
No production environment or training code was changed for these fixes.

A code review also aligned the new evaluator's stationary-step threshold to the
production value (motion norm <= 1e-6). A production-evaluator equivalence test
checks the final metric definitions. The corrected suite passes all eight tests
(4.93 seconds), and Ruff passes. Tests use tiny synthetic/example scenes and
small network updates, not the 33/30 experimental split or an experimental fit.
No batch experiment or trained endpoint was run during preparation.
