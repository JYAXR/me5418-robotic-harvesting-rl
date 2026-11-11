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
Results are pending at the pre-execution commit.
