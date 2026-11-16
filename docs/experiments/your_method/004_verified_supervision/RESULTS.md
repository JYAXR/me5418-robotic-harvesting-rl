# Own-route attempt 004: five-label reweighting does not achieve reliable repair

## Main result

The predeclared targeted continuation completes, but does not meet the repair
and retention objective. Its fixed endpoint succeeds on only 1/3 targeted
scenes and 10/33 training scenes, versus 2/3 and 12/33 for matched unchanged
continuation. Every evaluation is autonomous, with zero action overrides.

| Metric | Frozen source | Matched control | Targeted supervision |
| --- | ---: | ---: | ---: |
| Original scenes 10000/10001/10002 | 3/3 | 2/3 | 3/3 |
| Targeted scenes 290007/290012/290021 | 0/3 | 2/3 | 1/3 |
| All training scenes | 10/33 | 12/33 | 10/33 |
| Damage failures | 16 | 13 | 11 |
| Budget failures | 7 | 8 | 12 |
| Mean return across all 33 scenes | -6.328920 | 4.484419 | -0.175589 |

The targeted endpoint retains all three original scenes, but loses previously
successful training scenes **290017 and 290022**, while gaining 290007 and
290015. The unchanged aggregate 10/33 hides this exchange. Relative to matched
control, treatment gains 10002 but loses 290017, 290021 and 290022. Control itself
loses source success on 10002 while gaining 290007/290015/290021.

The lower damage count is accompanied by more timeouts. These results do not
support promoting the targeted candidate as a reliable replacement. Both
experimental endpoints are retained, and the original candidate is preserved.

## Autonomous outcomes on the three repaired scenes

| Scene | Matched control | Targeted supervision | Targeted return |
| --- | --- | --- | ---: |
| 290007 | Success in 30 steps | Success in 30 steps | 101.646621 |
| 290012 | Timeout at 200 steps | Timeout at 200 steps | -24.194670 |
| 290021 | Success in 39 steps | Timeout at 200 steps | -50.751326 |

Both arms have identical final metrics on 290007, including zero blocked steps
and peak normalized deformation 0.964742. Its recovery therefore cannot be
attributed specifically to the new weighting. Targeted 290012 has 163 blocked
and 162 stationary steps; targeted 290021 has 172 blocked and 170 stationary
steps. Full per-scene metric changes are retained in the artifact verification.

## Intervention and fixed training conditions

The five physically verified correction states exactly match existing dataset
rows 430/653/660/1052/1057, with labels 6/6/4/6/2. Their margin-supervision weights
alone change from 1 to 2 in treatment. No new transition or forced-action rollout
is inserted into TD. The 1440-transition dataset and its order stay unchanged.
Previous row weights 40/55/98/102/103 remain 2; the separate recovery loss retains
its /117 normalization.

Both arms start from the same complete attempt-034 targeted snapshot, retaining
Adam and RNG state. Each receives exactly 2000 additional full-batch updates.
TD coefficient 0.5, global supervision coefficient 1, margin 0.8, learning rate
0.0001, gamma 0.99, clipping 10, target interval 200 and network 341-256-256-8
are unchanged. No MC term or optimizer reset is introduced. Final TD and Adam
clocks are 14000 and 34000. No budget extension or checkpoint selection occurs.

## Checkpoint behavior

Only the six diagnostic scenes are evaluated at intermediate checkpoints;
all 33 are evaluated at each fixed endpoint. These correlated checks are not
independent training replicates or evidence of uninterrupted stability.

| Added updates | Control original / 3 | Control targeted / 3 | Treatment original / 3 | Treatment targeted / 3 |
| --- | ---: | ---: | ---: | ---: |
| 10 | 2 | 3 | 2 | 2 |
| 100 | 3 | 0 | 3 | 1 |
| 500 | 3 | 0 | 2 | 2 |
| 1000 | 3 | 1 | 3 | 2 |
| 2000 | 2 | 2 | 3 | 1 |

Treatment temporarily recovers 290021 at updates 100/500/1000, then loses it
at 2000. Original scene 10002 also fails treatment checks at 10 and 500. The
fixed endpoint and complete history are reported rather than selecting an
earlier favorable checkpoint.

## Fixed-state Q probes

Treatment improves all five reference-action margins relative to control, but
both endpoints still choose only two of the five literal reference labels.
Margins below are reference Q minus the best competing Q, computed on one
encoded reference input at a time.

| Dataset row | Reference label | Control margin | Treatment margin | Treatment greedy action |
| --- | --- | ---: | ---: | --- |
| 430 | east_large | 0.145790 | 0.230766 | east_large |
| 653 | east_large | -0.430435 | -0.180031 | east_small |
| 660 | north_large | -0.196602 | -0.007698 | north_small |
| 1052 | east_large | 0.186089 | 0.315742 | east_large |
| 1057 | east_small | -1.062820 | -0.930748 | south_large |

Better fixed-state margins are not sufficient evidence of better rollout
behavior. In particular, control succeeds on 290021 despite its negative
reference margin at row 1057. These probes describe the result; they do not
establish a complete causal explanation for the new failures.

## Execution and verification

Protocol/implementation commit: `7080498`, on `experiment/your-method`. The single
main run completed from 2026-11-16 10:17:42.091 to 10:29:21.480 UTC, approximately
11 minutes 39 seconds. Measured update computation was 73.0 seconds for control
and 61.3 seconds for treatment; this difference is not an algorithmic speed
comparison. The remaining work includes evaluation, verification and I/O.

Both substantive tests passed on their first invocation: the gradient changes
only on the declared five rows, control matches the old objective exactly, and
treatment matches an independently assembled objective including complete
training state. Four disposable unit-test optimizer updates are accounted for.
The initial unused-variable Ruff finding and its provenance-only fix are in
[PREPARATION.md](PREPARATION.md); final Ruff passes.

The first and last ten updates of each arm are independently replayed: all 40
metrics and full-state comparisons match exactly. Each endpoint has the expected
TD/Adam clocks and synchronized online/target networks. Evaluation preserves
training state and Torch RNG even while loading temporary inference models.
Initial scene records match historical evidence exactly. All twelve uncached
endpoint reloads match primary metrics, with unchanged learner states. All
pinned source/data/model hashes remain unchanged.

| Work | Count |
| --- | ---: |
| Main training updates | 4000 (2000 per arm) |
| Independent replay / unit-test updates | 40 / 4 |
| Primary autonomous episodes / endpoint reloads | 120 / 12 |
| Executed episode simulation actions | 10860 |
| Actual reset-certification actions / calls | 1968 / 45 |
| Total simulator actions | 12828 |
| Forced actions / new training samples | 0 / 0 |

The full frozen source bank is reused from attempt 001 after model/source hash
checks; six source scenes are freshly reproduced. Historical baseline costs are
not charged again. Certified-scene caching is evaluation-only. No fresh holdout
or final-test scenes are used. This is one continuation from one saved training
state; neither robust learning nor generalization is established.

Offline review verifies 132 manifest files, all 120 compressed trajectories,
four exported model/resume files and twelve reload metrics. It independently
recomputes reward totals, greedy actions, terminal metrics, resource counts and
per-scene success gains/losses, with no additional training or simulation.

## Artifacts and next decision

- [Summary](evidence/summary.json), [all evaluations](evidence/evaluations.json),
  [Q probes](evidence/probes.json), [source bank](evidence/source_baseline.json).
- [Independent updates](evidence/update_verification.json),
  [endpoint reloads](evidence/reload_verification.json),
  [provenance](evidence/provenance.json), [status](evidence/status.json),
  [manifest](evidence/manifest.json), [artifact verification](artifact_verification.json).
- Inference candidates: `models/your_method_004_control.pt` and
  `models/your_method_004_targeted.pt`. Exact hashes are in
  [model exports](evidence/model_exports.json).
- Complete continuation states: [control](resume/control_state_14000.pt) and
  [targeted](resume/targeted_state_14000.pt). Intermediate full snapshots remain
  in the local run directory; both final endpoints are versioned.

The next experiment should diagnose the remaining autonomous failures and the
two lost training successes before assuming stronger local weighting is enough.
No additional weight search or follow-up training has been run in this attempt.
