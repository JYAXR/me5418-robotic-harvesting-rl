# Attempt 028: frozen-policy generalization on unseen seeds

## Protocol declared before evaluation: 15 November 2026

Freeze models/step6_repair_027.pt at SHA-256
06733fccb5422d10ad7246e5368bbd5ed9cc3367f0b413847544ec26a96b0c88.
Its checkpoint metadata and attempt-011 dataset identify training seeds
10000, 10001 and 10002. The additional recovery label is from scene 10001.
Evaluate exactly 100 consecutive unseen seeds 280000--280099, each once,
with the checkpoint's original environment configuration, deterministic greedy
inference, zero exploration, no optimizer steps, no new replay data, no action
corrections and no model selection. This tests within-distribution generalization,
not a new layout family or increased randomization/noise.

Keep the environment's existing reference-route certification and internal
candidate retries unchanged. Record every certification attempt and its copied
simulation steps. Never replace a requested seed based on the policy outcome.
Retain reset errors separately; report success over all 100 requested seeds and
over successfully initialized scenes, with explicit denominators. Report a 95%
Wilson interval as descriptive scene-sampling uncertainty, not training-seed
uncertainty. Retain returns, lengths, damage, budget exhaustion, distance, contact,
blocking and stationary measures, initial geometry, observation hashes and every
chosen action/Q vector/transition outcome. Check distinct initial encoded
observations against all three training starts and other test scenes.

Classify outcomes using terminal flags first: success, damage, budget exhaustion,
or external truncation. For budget failures only, label persistent blocked stall
when the final stationary run is at least 20 steps and at least half of all steps
are collision-blocked; otherwise label stationary stall when the final stationary
run is at least 20 steps, or budget exhaustion while moving. Stationary means
executed displacement norm <= 1e-6. These are observed failure signatures, not
proofs of a unique causal mechanism. Record first blocked/stationary/contact
steps, trailing and longest stationary runs, action counts, terminal geometry,
and Q gaps for subsequent diagnosis; do not intervene on failed trajectories.

Run the three known scenes before and after the test, reporting these separately.
Verify frozen online/target weights, optimizer state, update count and agent/Torch
RNG around every evaluation. Reload the same file for independent normal-evaluator
checks on the first five valid requested seeds and the lowest seed in each observed
outcome category, deduplicated. Compare all episode metrics exactly. These extra
checks do not enter the main success rate and do not update weights.

Commit protocol, evaluation implementation and focused diagnostic tests before
running the 100-scene assessment. Record code/model/config hashes, package versions,
all outcomes and actual work (including reset certification and reloads). Archive
and push the completed attempt regardless of success rate. Do not change policy,
training code, rewards, environment, reports or previous experiment artifacts.
After observing these results, this seed set is no longer a blind final test set
for any future tuned model. Results pending at protocol commit.
