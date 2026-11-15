# Attempt 027: strengthen the early step-six action while retaining recovery

## Protocol declared before execution: 15 November 2026

Attempt 026 showed that correcting only scene 10001 step 6 to south_small
rescues the two residual attempt-025 failures. The state/action already exists
as original row 40. Resume the same full TD-2000 state used in attempt 025,
and change only that row's supervised weight from 1 to 2 through TD 10000.
Keep row 55 (step 21) weight 2, all other original row weights 1, the same
single off-reference north_large recovery observation and its weight 1,
denominator 117, margin 0.8, TD coefficient 0.5, learning rate 0.0001, gamma
0.99, target interval 200, clipping 10, no MC term, original 117 TD transitions,
Adam moments/clocks, target lineage and RNG. No extra pretraining or early stop.
The recovery sample is reused byte-for-byte, with no new collection actions.

Use the immutable attempt-025 control at its 69 full-episode checkpoints and
69 conditional recovery probes. Add full/conditional treatment checks at every
update from 2000 through 2050: 105 treatment checkpoints in total, comprising
69 matched and 36 additional early checks. Keep the additional checks separate
from matched comparisons. At each checkpoint run all three seeds autonomously
and a separate fixed-prefix recovery probe. Only the probe's first 24 actions
are scripted; no recovery or normal-episode action is forced.

Track original label agreement, seven reference-state Q probes and recovery
Q values at every update. Verify that weight 1 exactly reproduces the old
update and state; independently implement weight 2 and compare complete states.
Replay treatment updates 2001--2201 and 9801--10000 independently, audit target
and Adam clocks, then reload TD 2000, 2004, 2009, 2031, 2345, 9956 and 10000
for all three full episodes plus conditional recovery. Count all simulation,
scripted prefixes, copied reset certification, gates and verification updates
separately. Preserve every failure and perform frozen post-hoc verification
of any new failed checkpoints before interpreting a repair.

No inference-time action override, new reward, changed environment, changed
model architecture, or held-out evaluation is introduced. Success on sampled
known scenes is not a proof of continuous success or generalization. Record
English code, provenance, tests and all outcomes; synchronize each attempt to
Git. Keep prior artifacts and reports unchanged. Results pending at protocol
commit. A failed attempt is evidence, not a completed repair.
