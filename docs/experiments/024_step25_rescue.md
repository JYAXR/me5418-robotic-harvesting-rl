# Attempt 024: correct only the off-reference step-25 decision

## Protocol declared before execution: 15 November 2026

Use the verified frozen attempt-023 full models at TD 2031, 2226, 2344, 2345
and 2346. Each failed on scene 10001 after the first demonstration divergence
at step 22, and selected east_large (action 6) at the off-reference step-25
state. Run a fresh greedy control and corrected episode from each identical
full snapshot and reset seed. Override only one-based step 25 with north_large
(action 4), then immediately return to the unchanged greedy policy.

Do not correct step 22, follow the demonstration afterwards, retrain, change
TD/supervision coefficients, learning rate, Adam, target network, scene,
reward, budget, or any other condition. This is an external recovery action
at an off-reference observation, not the original demonstration label at
that time index. Require exact reproduction of the original control, identical
first 24 transitions and identical observation/Q/proposed action at the fork.
Only one executed action may differ from the policy per corrected episode.

Run ten main episodes and ten independent verification episodes. The latter
use the original rollout helper with a single-action policy adapter, comparing
all actual decisions, full trajectories and final outcomes. Require unchanged
complete training state before and after every episode. Keep reset reference
certification enabled and count its copied-simulator actions separately.

Preserve every result whether or not rescue succeeds. All five models are
selected from known failures; no held-out or population success-rate claim is
permitted. Success would show sufficiency of this one intervention for those
frozen failures, not that step 25 is the only cause or that the learned policy
has improved. No new optimizer updates or model training. Record English code,
full evidence and actual work, and synchronize the attempt to Git. Reports and
production defaults remain unchanged. Results pending at protocol commit.
