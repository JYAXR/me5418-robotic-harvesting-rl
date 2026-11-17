# Post-run diagnosis: why two supervised recovery rounds remain weak

This analysis performs no optimizer updates and no new environment rollouts.
It uses saved datasets, checkpoints, update logs and complete evaluation traces.
Gradient computations use autograd only; parameter tensors and checkpoint hashes
are checked unchanged. The two-round training limit remains closed.

## Supported findings

### 1. Recovery aggregation still strongly favors expert-route states

There are 4103 accepted recovery labels, but only 140 first-action labels at
the actual queried learner states, one per successful query (3.41% of accepted
labels). The remaining labels come from subsequent planner-executed trajectory
states. Some states may coincide, so this is a statement about collection
provenance, not a count of unique or necessarily out-of-distribution states.

Queries were limited to steps 8, 24 and 40. Thirty-eight reached queries failed
to produce a route and supplied zero labels; later autonomous failure states
were not systematically queried. Successful full-route filtering guarantees
label execution quality, but selects away difficult states the planner cannot
handle. More labels in this scheme do not necessarily mean better coverage of
learner mistakes. In addition, sampling rows uniformly within each pool gives
long successful expert suffixes more weight than short ones or their recovery
entry decisions.

This is consistent with the sequential distribution issue addressed by
[Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html): learning
changes the states visited, so learner-state coverage matters. The present
limited-query/full-suffix implementation should not be treated as exhaustive
expert labeling of all learner-visited states.

### 2. The supervised objective improves average loss while losing action choices

On the same 1441 base labels:

| Endpoint | Exact action accuracy | Cross-entropy |
| --- | ---: | ---: |
| Source | 86.1901% | 0.737824 |
| Round 1 | 80.6384% | 0.577578 |
| Round 2 | 79.5975% | 0.547279 |

The first round's recorded accuracy after just its first update is 70.9924%.
The first update is not archived as a standalone model; this observation comes
from the existing fit log. Switching from the previous TD-plus-margin objective
to pure cross-entropy, resetting Adam and updating the whole network did not
include a constraint preserving previously successful behavior. Loss reduction
therefore did not imply preservation of greedy action choices or episode success.
The experiment changed several aspects together; it does not isolate which of
objective, initialization, step size or optimizer reset caused the early change.

On all 4103 final recovery labels, exact action accuracy is only 68.1209%, versus
65.2449% for the source evaluated on the same final pool. Final top-two label
coverage is 87.4482% and direction accuracy is 81.7938%. Of 1308 label mismatches,
561 choose the same direction with the other step size and 747 choose a different
direction. Splitting only small/large action heads cannot account for all errors.
These are expert-label mismatches, not independently established harmful actions;
some alternative actions may still be valid.

Roughly 47.4% of round-2 updates exceeded the clipping norm before clipping.
This is an optimization diagnostic, not proof that clipping or learning rate
causes failure. Extending the same fit without a retention check is not justified
by the available evidence.

### 3. Failures are dominated by prolonged blocking and limited recovery behavior

Round 1 has 16 training timeouts, all with at least 100 blocked steps. Round 2
has 12 training timeouts, 10 with at least 100 blocked steps. Nine of its ten
validation timeouts also have at least 100 blocked steps; all ten contain a run
of at least 50 consecutive blocked actions. Other failures are damage events.

The complete 5544-row final supervised dataset contains only 15 west_small labels
and zero west_large labels (base pool has neither). Round-2 evaluation executes
12 west_small actions on training scenes and nine on validation, and no
west_large actions. This documents highly asymmetric action coverage. It does
not prove that every blocked trajectory requires a westward action, but it
shows that retreat/escape behavior is poorly represented.

Pure successful-action classification does not explicitly penalize repeated
blocked actions or distinguish multiple locally safe alternatives by eventual
recovery value. The current policy contains no explicit blocked-action feedback
mechanism; remaining budget changes, but it has not learned to use that signal
to reliably leave the repeated-blocking states.

### 4. The current representation makes geometry reasoning implicit

The observation contains current/rest branch nodes, stiffness, contact flags,
safe limits, robot position, target displacement and remaining budget. It is
not justified to claim that obstacle geometry is absent. The encoder scales
absolute branch coordinates and flattens them into a 341-dimensional vector
for the two-layer MLP. It does not explicitly supply robot-relative obstacle
distances, per-direction clearance or an action-feasibility representation.

The planner explicitly computes swept clearance and searches a route; the MLP
must infer that structure from raw coordinates and sparse action labels. This
is a plausible representation/inductive-bias limitation, not a demonstrated
proof that the current network is too small or that increasing its width fixes it.
Future geometry features should retain global obstacle/goal information so a
local-clearance policy does not lose the ability to plan detours.

## Hypotheses not supported as primary explanations

- **Widespread exact contradictory labels:** only 4 of 3521 unique exact inputs
  have multiple action labels, involving 13 of 5544 rows. The maximum possible
  unweighted exact-label accuracy under only these conflicts is 99.9278%.
  Ignoring the budget feature gives 8 conflicting physical-input groups and
  a 99.8377% ceiling. Exact contradictions cannot explain accuracy near 68%.
  Nearby-state discontinuities and alternative good actions remain possible.
- **A general base/recovery gradient conflict:** on the common final dataset,
  full-batch gradient cosines are +0.7171 at the source, +0.0415 at round 1 and
  +0.4495 at round 2. The equally weighted negative gradient is a local descent
  direction for both pool losses at these inspected endpoints. This rules out
  claiming an observed global opposing-gradient explanation here, not transient
  minibatch conflicts or the effects of Adam's actual update direction.
- **TD alone:** no TD was used in these rounds, yet original behavior regressed.
  TD can still have harmed earlier experiments; it is not a sufficient explanation
  for the current supervised-only result.
- **A proven simulator or evaluator defect:** prior paired replay audits and the
  production-evaluator equivalence test passed. No new evidence from this analysis
  establishes such a defect; this is not an exhaustive correctness proof.

## Recommended direction, not executed

1. Redesign the collection unit around learner-visited decision states across
   complete trajectories, especially repeated blocking, near contact and recovery
   transitions. Balance scenes and state categories; prevent long safe expert
   suffixes from overwhelming the actual correction-entry labels. Preserve
   unresolved planner states explicitly rather than treating them as solved.
2. Add explicit robot-relative geometry and per-action clearance/contact-risk
   information, then check frozen action feasibility on the whole saved bank.
   If a hard action mask is considered, validate it against real substep physics
   and legitimate flexible-branch contacts; a rigid-obstacle filter may remove
   valid harvesting actions. No mask is applied in this analysis.
3. Before another full fit, specify a small offline fitting/retention experiment
   with clear stopping criteria. Use a dedicated policy objective/head instead
   of assuming value scores are suitable classification logits unchanged; retain
   behavior on previously successful trajectories with an explicit constraint
   and scene-balanced rehearsal. Do not preserve every bad source decision.
   First demonstrate fitting and preservation on a fixed data bank, then evaluate
   full autonomous episodes. These are proposals requiring a new experiment,
   not changes performed now.
4. Consider broader architectures or later RL only after those tests identify
   the bottleneck. Do not resume individual-step weight patches or treat another
   longer run with the same data/loss as the default next action.

Evidence: [dataset and frozen fit analysis](diagnostic_analysis.json),
[endpoint gradient analysis](gradient_analysis.json),
[saved-trajectory failure analysis](trajectory_analysis.json),
[main results](RESULTS.md), [original protocol](PROTOCOL.md).
