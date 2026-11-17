# Attempt 009: retention works, learning is mostly blocked

The single bounded experiment preserved all ten old successful training scenes
and improved development validation from 3/30 to 4/30. Training coverage stayed
at 10/33. The 400-update pilot failed its continuation gate, so the planned
2000-update extension was not run. No further round or manual correction was
performed. This is a small provisional candidate, not a broadly competent policy.

Protocol/code commit: `fc78fdd`, pushed before execution on
`experiment/your-method`. Main run: 17 November 2026, timestamps in
`evidence/status.json` (UTC). See [protocol](PROTOCOL.md),
[machine-readable summary](evidence/summary.json), and [audit](audit.json).

## Autonomous outcomes

| Endpoint | Original scenes | Training success | Training damage / budget failure | Development validation success | Validation damage / budget failure |
| --- | --- | --- | --- | --- | --- |
| Source attempt-004 targeted | 3/3 | 10/33 | 11 / 12 | 3/30 | 14 / 13 |
| Geometry + retained behavior | 3/3 | 10/33 | 12 / 11 | 4/30 | 12 / 14 |

Training mean return changed from -0.175589 to -0.279987; validation mean return
changed from -31.086667 to -24.820317. There were no reset errors or action
overrides. All source successful training scenes and all three source successful
validation scenes were retained. Scene 590014 improved from damage at step 88
(return -93.932473) to success at step 53 (return 90.584652). However, training
scene 290002 changed from budget failure to damage. This is not an across-the-board
improvement in safety. Thirty of the 33 complete training trajectories remained
exactly identical to the source, including observations, actions, rewards,
termination, information fields, and RNG states.

The candidate meets the predeclared replacement eligibility rule: retain all
ten old training successes, retain original 3/3, and exceed 3/30 validation.
It is recorded as the provisional own-route candidate; the source checkpoint
remains intact as a comparator. One added validation success is weak evidence,
and this previously inspected development bank is not a final held-out test.
No component-specific or statistically reliable generalization claim is made.

## What was implemented

- Complete source-policy trajectories, with risk and time-coverage queries,
  rather than training mostly on successful planner suffixes.
- Forty explicit commanded-path geometry features and a 381--128--8 residual
  policy correction network; the original DQN weights remain immutable.
- Recorded failed planner queries and physics-based immediate adverse-action
  labels; inference has no hard collision mask or oracle action override.
- A 454-action retention bank from ten successful source trajectories, anchor
  KL and margin terms, and exact batch-one action checks after each update block.
- A separate versioned loader and evaluation CLI; existing DQN checkpoints and
  training code remain compatible and unchanged.

Collection yielded 167 actual visited states: 84 risk and 83 coverage queries.
Sixty queries occurred after step 40, including 14 after step 100. There were
108 successful planner continuations, each contributing only its first/root
action. All 59 failed queries were retained with unknown action labels: 54
planner-no-route cases and five budget failures. Seventy-five states contributed
usable immediate adverse-action labels. The original 1440 demonstration rows
were retained. There were no validation rows in training.

## Pilot and objective conflict

The verified-root accuracy stayed at 31/108 (28.7037%), below the required
three-percentage-point improvement. Of twenty proposed 20-update blocks, only
the first was accepted, at interpolation scale 0.125. The other nineteen were
rolled back together with their Adam state. Thus there were **400 attempted
updates but only 20 surviving Adam steps**, not 400 accepted updates. No new TD
updates were performed. All 454 protected actions were retained.

Post-run offline auditing found 42 query inputs exactly overlapping the protected
success bank. Fourteen have planner action labels different from the already
successful source action. Both continuations succeed, but single-label
cross-entropy favors one action while exact retention requires the other.
There is also one adverse-action mask marking an action used on a verified
successful source trajectory: temporary no-motion can be part of eventual
success. The collection rule exempted the planner's verified action, but not
every independently verified successful source action.

These are concrete conflicts in the new objectives. They do not prove that all
rejected updates were caused by those rows: shared network parameters, very
small action margins, and block-level backtracking may also constrain learning.
The audit did not replay or attribute every rejected proposal. Since almost all
proposals were rejected, this run also cannot establish that geometry features
are ineffective. It shows that the current combination preserves behavior but
largely prevents learning.

The next change should resolve supervision semantics before increasing training:
use sets of physics-verified successful actions at overlapping states, avoid
penalizing a protected successful action merely because it differs from one
planner route, and exclude verified successful actions from the adverse mask.
Audit those conflicts before any subsequent pilot. Keep the same bounded
evaluation discipline; do not return to scene-by-scene manual fixes. These
follow-up changes were **not** applied in this run.

## Cost and verification

Main wall time was 752.52 seconds (12.54 minutes); the fit/export stage took
44.19 seconds. Four worker processes each used one Torch thread. Summed worker
time was 1373.07 seconds for collection and 1213.77 seconds for evaluation;
these overlap and must not be interpreted as wall time. Recovery-witness planner
and physics time summed to 158.61 and 179.41 seconds respectively.

| Main-run physics work | Actions |
| --- | ---: |
| 33 complete collection episodes | 3343 |
| Planner recovery witnesses, including failed attempts | 2656 |
| Eight one-step probes at 167 query states | 1336 |
| 63 final autonomous evaluations | 6709 |
| Scene certification | 3779 |
| Total | 17823 |

There were 89 certification calls and seven cache hits. Process-local cache
placement limited reuse; this is not evidence of an end-to-end speedup.

Preparation passed 12 relevant tests and Ruff, preserved all 454 historical
actions before training, and completed a separate two-step synthetic worker
smoke test. Post-run auditing checked 306 evidence files and 47 pinned inputs,
recomputed all 6709 final greedy decisions from the exported model, verified
all root labels and adverse masks against archived witnesses, confirmed no
validation leakage, and checked immutable inherited weights, resume parameters,
Adam step counts, all retained actions, and exact paired initial states. The
audit performed no optimizer updates or new physics steps. The separate CLI
recheck of the three original scenes is archived in `cli_verification.json`;
all metrics exactly match the main evaluation. Its 128 repeated episode actions
and unprofiled certification work are outside the main-run cost table and do
not select a different checkpoint. [Selection metadata](selection.json) records
the provisional candidate and its tradeoffs.

## Artifacts

Inference checkpoint: `models/your_method_009_geometry.pt`

SHA-256: `7a4100d51d1df74f25081593407683e6d637a0654ebeba82925d5ae60326d969`

Optimizer/RNG snapshot: `evidence/resume.pt`

SHA-256: `75744f01d89cd239335456d2925401bbd77c5a9d5e3817268f8713b440b5ba53`

The evidence directory includes collection and evaluation traces, original and
derived datasets, every attempted loss, block decisions, input hashes, and an
artifact manifest. `audit.py` performs offline verification; `run.log` retains
execution output. No historical experiment or teammate branch was overwritten.

```text
python scripts/evaluate_recovery_policy.py --checkpoint models/your_method_009_geometry.pt --seed 10000 --episodes 3
python docs/experiments/your_method/009_geometry_retention/audit.py
```
