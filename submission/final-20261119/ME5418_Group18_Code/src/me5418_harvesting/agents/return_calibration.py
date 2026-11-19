"""Fixed demonstration returns and supervised calibration before TD adaptation."""

from math import isfinite
from numbers import Real

import torch
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_learning import (
    DQNUpdater,
    TransitionBatch,
    demonstration_margin_loss,
    validate_demonstration_loss,
)


def discounted_demonstration_returns(batch: TransitionBatch, gamma: float) -> torch.Tensor:
    """Compute recorded discounted returns for one complete ordered trajectory.

    The final reward is included exactly once. A true terminal endpoint is
    required; partial or externally truncated trajectories cannot supply these
    targets without an additional value estimate, which this function never uses.
    """
    if isinstance(gamma, bool) or not isinstance(gamma, Real) or not isfinite(gamma):
        raise ValueError("Gamma must be a finite real number between zero and one.")
    if not 0 <= gamma <= 1:
        raise ValueError("Gamma must be between zero and one.")
    rewards = batch.rewards
    if (rewards.ndim != 1 or rewards.numel() == 0 or not rewards.is_floating_point()
            or not torch.isfinite(rewards).all()):
        raise ValueError("Expected nonempty finite one-dimensional floating-point rewards.")
    count = len(rewards)
    for name in ("terminated", "truncated"):
        flags = getattr(batch, name)
        if flags.shape != rewards.shape or flags.dtype != torch.bool:
            raise ValueError("Termination flags must be one-dimensional boolean vectors.")
        if flags.device != rewards.device:
            raise ValueError("Trajectory rewards and termination flags must share a device.")
    if not bool(batch.terminated[-1]) or bool(batch.terminated[:-1].any()):
        raise ValueError("Expected exactly one task termination at the final transition.")
    if bool(batch.truncated.any()):
        raise ValueError("Externally truncated trajectories do not have complete returns.")
    observations, next_observations = batch.observations, batch.next_observations
    if (observations.ndim != 2 or next_observations.shape != observations.shape
            or observations.shape[0] != count or observations.shape[1] == 0
            or batch.actions.shape != rewards.shape):
        raise ValueError("Trajectory observations and actions must match the transition count.")
    if (observations.device != rewards.device or next_observations.device != rewards.device
            or observations.dtype != next_observations.dtype
            or not torch.isfinite(observations).all()
            or not torch.isfinite(next_observations).all()):
        raise ValueError("Trajectory observations must be finite and have matching layouts.")
    if not torch.equal(next_observations[:-1], observations[1:]):
        raise ValueError("Demonstration transitions must be contiguous and in trajectory order.")
    with torch.no_grad():
        targets = torch.empty_like(rewards)
        following_return = rewards.new_zeros(())
        for index in range(count - 1, -1, -1):
            following_return = rewards[index].detach() + gamma * following_return
            targets[index] = following_return
        if not torch.isfinite(targets).all():
            raise FloatingPointError("Discounted demonstration returns are not finite.")
    return targets.detach()


def calibration_update(
    learner: DQNUpdater, batch: TransitionBatch, returns: torch.Tensor, *,
    demonstration_weight: float = 1.0, demonstration_margin: float = 0.8,
) -> dict:
    """Fit demonstrated action values to fixed returns without bootstrapping.

    This update reads current observations, demonstrated actions and the supplied
    fixed targets only. Rewards, next observations and the target network do not
    enter the objective. The caller supplies complete-trajectory return targets.
    """
    validate_demonstration_loss(demonstration_weight, demonstration_margin)
    if (not isinstance(returns, torch.Tensor) or returns.ndim != 1 or returns.numel() == 0
            or returns.shape != batch.rewards.shape or not returns.is_floating_point()):
        raise ValueError("Return targets must be a nonempty floating-point vector matching rewards.")
    if returns.device != batch.observations.device or returns.device != batch.rewards.device:
        raise ValueError("Return targets and current observations must share the batch device.")
    if not torch.isfinite(returns).all():
        raise ValueError("Return targets must be finite.")
    if (batch.observations.ndim != 2 or len(batch.observations) != len(returns)
            or batch.actions.shape != returns.shape or batch.actions.dtype != torch.long
            or batch.actions.device != returns.device
            or not torch.isfinite(batch.observations).all()):
        raise ValueError("Current observations and demonstrated actions must match the targets.")

    targets = returns.detach()
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = torch.nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    scores = learner.online(batch.observations)
    chosen_q = scores.gather(1, batch.actions[:, None]).squeeze(1)
    return_loss = F.smooth_l1_loss(chosen_q, targets)
    margin_loss = return_loss.new_zeros(())
    accuracy = float((scores.detach().argmax(dim=1) == batch.actions).float().mean())
    if demonstration_weight > 0:
        margin_loss = demonstration_margin_loss(scores, batch.actions, demonstration_margin)
    loss = return_loss + demonstration_weight * margin_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("The return-calibration loss is not finite.")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True,
    )
    learner.optimizer.step()
    with torch.no_grad():
        scores_after = learner.online(batch.observations)
        chosen_after = scores_after.gather(1, batch.actions[:, None]).squeeze(1)
        return_after = F.smooth_l1_loss(chosen_after, targets)
        margin_after = return_after.new_zeros(())
        if demonstration_weight > 0:
            margin_after = demonstration_margin_loss(
                scores_after, batch.actions, demonstration_margin,
            )
        loss_after = return_after + demonstration_weight * margin_after
        after = torch.nn.utils.parameters_to_vector(learner.online.parameters())
        delta = torch.linalg.vector_norm(after - before)
        if not torch.isfinite(after).all() or not torch.isfinite(loss_after):
            raise FloatingPointError("The return-calibration update produced non-finite values.")
    learner.update_count += 1
    return {
        "loss_before": loss.item(), "loss_after": loss_after.item(),
        "return_loss_before": return_loss.item(), "return_loss_after": return_after.item(),
        "demonstration_loss_before": margin_loss.item(),
        "demonstration_loss_after": margin_after.item(),
        "gradient_norm": gradient_norm.item(), "parameter_delta_norm": delta.item(),
        "target_synchronized": False, "demonstration_action_accuracy": accuracy,
    }
