"""Supervised policy fitting and an explicit boundary before TD adaptation."""

import torch
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_learning import (
    DQNUpdater,
    TransitionBatch,
    demonstration_margin_loss,
    validate_demonstration_loss,
)


def supervised_update(
    learner: DQNUpdater, batch: TransitionBatch, *,
    objective: str = "margin_only", margin: float = 0.8,
) -> dict:
    """Fit demonstration actions without using rewards, next states or a target.

    The update matches the supervised controls in the fixed-scene diagnostic.
    Outputs are action scores during this phase, not calibrated return estimates.
    The caller must supply demonstrated actions rather than exploratory labels.
    """
    if objective not in ("margin_only", "cross_entropy"):
        raise ValueError(f"Unknown supervised objective: {objective}")
    validate_demonstration_loss(0.0, margin)
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = torch.nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    scores = learner.online(batch.observations)
    loss = (F.cross_entropy(scores, batch.actions) if objective == "cross_entropy"
            else demonstration_margin_loss(scores, batch.actions, margin))
    if not torch.isfinite(loss):
        raise FloatingPointError("The supervised fitting loss is not finite.")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True,
    )
    learner.optimizer.step()
    after = torch.nn.utils.parameters_to_vector(learner.online.parameters()).detach()
    if not torch.isfinite(after).all():
        raise FloatingPointError("The supervised update produced non-finite parameters.")
    learner.update_count += 1
    return {
        "loss_before": loss.item(), "gradient_norm": gradient_norm.item(),
        "parameter_delta_norm": torch.linalg.vector_norm(after - before).item(),
        "target_synchronized": False,
    }


def begin_td_phase(learner: DQNUpdater) -> dict:
    """Keep learned weights, synchronize the target and start a fresh Adam phase.

    A supervised target network remains at initialization unless explicitly
    synchronized. Copy the learned online model before the first TD target is
    computed. Reset Adam moments and the phase update clock deliberately; this
    is a new optimization phase, not an exact continuation of optimizer state.
    """
    pretraining_updates = learner.update_count
    learner.optimizer = torch.optim.Adam(
        learner.online.parameters(), lr=learner.config.learning_rate,
    )
    learner.online.zero_grad(set_to_none=True)
    learner.target.load_state_dict(learner.online.state_dict())
    learner.target.requires_grad_(False)
    learner.target.eval()
    learner.target.zero_grad(set_to_none=True)
    learner.update_count = 0
    return {
        "pretraining_updates": pretraining_updates, "phase_updates": 0,
        "optimizer_reset": True, "target_synchronized": True,
    }
