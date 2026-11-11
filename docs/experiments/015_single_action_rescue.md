# Attempt 015: one-action rescue of the two extended-TD failures

## Protocol recorded before execution

Attempt 014 continued the calibrated, preserved-Adam policy at learning rate
0.0001 and demonstration weight 1 through TD update 10000. Scene 10001 failed
at updates **2009 and 10000**. Both first departed from the demonstration at
**one-based step 21**, choosing `north_large (4)` instead of `north_small (0)`.
The demonstrated-action Q gaps were -0.016716 and -0.059891, respectively.
This attempt tests whether changing that one action is sufficient to restore
success for each of those two frozen models.

Use the exact attempt 014 full training snapshots and the unchanged scene,
encoder, rewards, step budget, seed and neural action selection. Verify the
committed attempt 014 archive and original model/data/source hashes, including
the provenance chain back to attempts 013 and 011. Do not retrain, reset Adam,
change network weights, collect demonstrations, select another checkpoint or
change the intervention after observing an outcome.

For each model, perform these two episodes from a freshly restored copy of its
full snapshot and an environment reset with seed **10001**:

1. **Unmodified greedy reproduction.** Run the original policy throughout.
   Require exact agreement with the attempt 014 recorded actions, positions,
   return and terminal outcome.
2. **Single-action intervention.** At step 21 only, execute action 0,
   `north_small`, while recording the network's own proposed action and all
   eight Q scores. Every other step, including step 22 onward, executes the
   current greedy output of the same frozen neural network.

The first 20 actions and the state before step 21 must be identical across the
paired episodes and the demonstration. The original proposed action at the
fork must be `north_large (4)`; the substituted action must be
`north_small (0)`. Verify exactly one forced action and verify that it reaches
the demonstrated step-22 state. Do not force any later demonstration action or
use a planner to complete the episode. A later reference mismatch must not be
treated as another instruction to intervene.

This is **four new physical episodes and zero optimizer updates**. The two
greedy reproductions are expected to use 400 actions in total, but their
agreement is checked rather than assumed. Count actual intervention actions
and all completed/partial work separately. A failed or unsuccessful rescue is
retained; the episode is not retried with another action. Reset feasibility
probes are excluded from policy-action counts, consistently with earlier work.

Restore the saved online/target networks, Adam moments and parameter steps,
absolute update count, and both RNG states before each arm. Compare the complete
training state before and after inference. These models inherit 42009 and 50000
branch updates respectively; this shared history is not added together or
counted as new training. The stored learning rate remains 0.0001; the
supervision weight of 1 describes their source training and is not an active
loss in this evaluation-only experiment.

## Evidence and verification

Record every step's actual/proposed action, intervention flag, position, reward,
Q vector, encoded-state hash, motion, blocked/stationary status and termination.
Retain paired fork comparisons, success/damage/budget outcomes, returns, episode
lengths, full-state invariance checks and exact source identifiers.
Write the protocol and implementation to Git before executing the diagnostic.
Use a new run directory and keep all previous attempts immutable.

Independent verification is **read-only**, with no additional episodes, resets
or optimizer updates. Audit the archived inputs, source and model hashes,
complete action logs, baseline reproduction, paired prefix and the unique
step-21 override. Recompute model Q values on the known shared demonstrated
prefix through step 21; validate stored Q-vector argmax and action-selection
rules at every subsequent step. This does not claim an independent physics
replay of later off-demonstration states. Keep main episode costs separate from
read-only verification.

## Interpretation limits

If a corrected episode succeeds, one action replacement was sufficient for
that model on this deterministic known scene. It does not mean the neural
policy has learned the corrected decision, that its weights have improved, or
that unassisted evaluation now succeeds. If it fails later, record the next
deviation without adding a second correction.

The two checkpoints belong to one training lineage and one scene; they are
not independent generalization trials. Counterfactual success will be labeled
as assisted diagnostic success and never substituted for the original failed
policy outcomes. A useful next training change can be considered only after
these fixed comparisons are recorded.

## Reproduction

Run from the repository root with the configured environment:

```powershell
python docs/experiments/015_single_action_rescue/run_single_action_rescue.py --reference-run results/training/td_extension_10000_20261112 --archive-dir docs/experiments/014_td_extension/evidence --output-dir results/training/single_action_rescue_20261112
python docs/experiments/015_single_action_rescue/verify_single_action_rescue.py --run-dir results/training/single_action_rescue_20261112 --output results/training/single_action_rescue_20261112/verification.json
python docs/experiments/015_single_action_rescue/plot_single_action_rescue.py --run-dir results/training/single_action_rescue_20261112
python docs/experiments/015_single_action_rescue/archive_single_action_rescue.py --run-dir results/training/single_action_rescue_20261112 --destination docs/experiments/015_single_action_rescue/evidence
```

Local execution uses `.venv/Scripts/python.exe`. Original binary checkpoints and
the immutable demonstration data are required locally; the Git evidence archive
does not contain those binaries. The runner refuses to overwrite a run directory.
The protocol and implementation were committed and pushed before execution as
`c9754713ad0ee008c3537d019e0b0b9075a80c7e`.

## Execution and results

The diagnostic completed from 2026-11-11 18:27:43.796083 UTC to
18:30:24.641948 UTC (12 November, 02:27:43--02:30:24 in Asia/Shanghai).
Both unmodified greedy episodes reproduced the attempt 014 failures exactly.
**Replacing only step 21 with north_small rescued both frozen models.**

| Source TD update | Episode | Success | Actions | Return | Blocked / stationary steps | Forced actions |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2009 | Original greedy | No, budget exhausted | 200 | 2.479857 | 178 / 177 | 0 |
| 2009 | Step 21 corrected | Yes | 41 | 107.585050 | 0 / 0 | 1 |
| 10000 | Original greedy | No, budget exhausted | 200 | 2.479857 | 178 / 177 | 0 |
| 10000 | Step 21 corrected | Yes | 41 | 107.585050 | 0 / 0 | 1 |

All four episodes have zero damage. Both corrected episodes execute the same
41-action sequence as the demonstration, with no later reference-action
divergence. This agreement is observed after the intervention; later actions
are not supplied from the demonstration. The network independently selects each
action from step 22 onward.

Each pair has identical first-20-step transition records and identical state,
proposed action and Q scores at step 21. The unchanged models still propose
`north_large (4)` at the fork. The demonstration-minus-competitor Q gaps remain
**-0.0167160034** at TD 2009 and **-0.0598907471** at TD 10000; the diagnostic
does not change those scores. Only the executed action becomes `north_small (0)`.
The corrected next state matches the demonstrated step-22 state, and at step 22
both models autonomously select and execute `north_large (4)` without another
override.

The physical sequence explains why this decision matters. At step 21, the
original large northward action is not itself blocked; it moves about 0.06,
instead of the corrected 0.025. In the original replay, the subsequent eastward
turn at step 23 is partially blocked, followed by stationary pushing. The
corrected replay makes two autonomous large northward moves at steps 22 and 23
before turning east at step 24, and completes the route without blockage.
The recorded positions and actions support this sequence; it is not inferred
from the Q margin alone.

## State preservation and accounting

The runner freshly restores the complete source snapshot for every episode.
Exact equality checks before and after each episode pass for online and target
weights, Adam state, the absolute update counter and both RNG states.
TD counters remain 2009 and 10000; Adam parameter steps remain 22009 and 30000.
The original binary checkpoints, immutable inputs and production source hashes
also pass a second check after the diagnostic.

The actual new work is **four episodes and 482 policy actions**: 400 unmodified
actions plus 82 corrected-episode actions. Exactly two actions are externally
replaced, one per corrected episode. There are **zero optimizer updates**,
new demonstration-collection actions or held-out scene evaluations. The two
models' shared historical training is not rerun or summed as a new budget.

The full traces retain every action, Q vector, state hash, reward and outcome
without sampling. The original failures remain unchanged in attempt 014 and
are reproduced in this attempt's control arms.

## Verification

- **50 targeted tests passed**: 20 new runner tests plus 12 existing
  action-diagnostic regression tests (32 tests in 10.07 seconds), and
  18 new read-only verification tests in 4.23 seconds. They cover fixed
  one-based indexing/action selection, extra-override rejection, unchanged
  full-state checks, paired-prefix checks and continued neural decisions.
  Ruff and whitespace checks pass. No production training code changed.
- Independent read-only verification passed all four traces and all
  **482 action records**, including exact original outcomes, the two identical
  prefixes, exactly one replacement per corrected episode, and the complete
  recorded greedy-selection rules.
- It independently recomputed **84 Q vectors**: prefix steps 1--21 for each
  of the four episodes. Later recorded Q argmax/action rules were audited,
  without claiming an independent off-route physics or full off-route Q replay.
  Verification added **zero episodes, resets, environment actions or updates**.
- An additional read-only review reconciled the raw trace logs, demonstration,
  source outcomes and summary. The black-and-white three-line outcome table
  was visually checked.

See the [outcome table](015_single_action_rescue/evidence/single_action_outcomes.png),
[paired results](015_single_action_rescue/evidence/replays.json),
[independent verification](015_single_action_rescue/evidence/verification.json),
and [archive manifest](015_single_action_rescue/evidence/archive_manifest.json).
All four complete JSON traces and JSONL action logs are archived in Git, with
hashes, resolved settings and source provenance. Models remain in their original
ignored result directories. No report or submission archive is modified.

The launch provenance marks the working tree dirty because of the existing
untracked report document. The experiment code was committed before launch;
its source and helper hash checks pass.

## Conclusion and next question

A single correction at step 21 is sufficient to restore success for **both**
tested models on scene 10001. Their downstream greedy behavior can complete
the demonstrated route once this critical decision is corrected. The tested
failure therefore has a concrete, locally correctable action-selection point.

This is **2/2 assisted diagnostic rescues**, not a new learned-policy success
rate. Neither model's parameters or original unassisted performance changed.
The result does not establish behavior on unseen scenes or exclude other failure
points at other checkpoints. A next separately recorded training experiment
can focus on maintaining the small-versus-large northward action ordering at
this state and nearby decisions, while checking all three scenes for new
regressions. No such training change is made in this attempt.
