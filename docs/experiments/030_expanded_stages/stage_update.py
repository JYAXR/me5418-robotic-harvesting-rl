"""Fit demonstrated actions and actual returns before the unchanged TD objective."""

import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "stages030coverage", ROOT / "docs/experiments/029_training_coverage/coverage_update.py")
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)


def validate(batch, recovery, returns):
    coverage.validate(batch, recovery, 2.0)
    if returns is not None and (
        returns.shape != batch.rewards.shape or returns.dtype != torch.float32
        or returns.device != batch.observations.device or not torch.isfinite(returns).all()
    ):
        raise ValueError("Calibration requires one finite float32 return per demonstration row.")


def components(values, actions, recovery_values, returns):
    """Retain the existing two weighted rows and the separate recovery label."""
    margin = coverage.focused_margin_loss(values, actions, 2.0)
    recovery = coverage.recovery_term(recovery_values)
    actual = (F.smooth_l1_loss(values.gather(1, actions[:, None]).squeeze(1), returns.detach())
              if returns is not None else values.new_zeros(()))
    return margin + recovery + actual, margin, recovery, actual


def stage_update(learner, batch, recovery, returns=None):
    """Use no next state or target network; returns are fixed complete-episode MC targets."""
    validate(batch, recovery, returns)
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    values = learner.online(batch.observations)
    loss, margin, extra, actual = components(
        values, batch.actions, learner.online(recovery), returns)
    accuracy = float((values.detach().argmax(1) == batch.actions).float().mean())
    if not torch.isfinite(loss):
        raise FloatingPointError("The staged objective is not finite.")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True)
    learner.optimizer.step()
    with torch.no_grad():
        after_loss, after_margin, after_extra, after_actual = components(
            learner.online(batch.observations), batch.actions, learner.online(recovery), returns)
        after = nn.utils.parameters_to_vector(learner.online.parameters())
        delta = torch.linalg.vector_norm(after - before)
        if not torch.isfinite(after).all() or not torch.isfinite(after_loss):
            raise FloatingPointError("The staged update produced non-finite values.")
    learner.update_count += 1
    return {
        "loss_before": loss.item(), "loss_after": after_loss.item(),
        "demonstration_loss_before": margin.item(), "demonstration_loss_after": after_margin.item(),
        "recovery_loss_before": extra.item(), "recovery_loss_after": after_extra.item(),
        "return_loss_before": actual.item(), "return_loss_after": after_actual.item(),
        "gradient_norm": norm.item(), "parameter_delta_norm": delta.item(),
        "demonstration_action_accuracy": accuracy, "target_synchronized": False,
    }


def independent_update(learner, batch, recovery, returns=None):
    """Independently assemble the two pre-TD objectives for full-state replay checks."""
    def losses(q, rq):
        # This expression does not call the production weighted-margin helper.
        offsets = torch.full_like(q, 0.8)
        offsets.scatter_(1, batch.actions[:, None], 0.0)
        demonstrated = q.gather(1, batch.actions[:, None]).squeeze(1)
        hinges = (q + offsets).max(1).values - demonstrated
        coefficients = torch.ones_like(hinges)
        coefficients[40] = coefficients[55] = 2.0
        supervised = (hinges * coefficients).mean()
        recovery_offsets = torch.full_like(rq, 0.8)
        recovery_offsets[:, 4] = 0.0
        labels = torch.full((len(rq), 1), 4, dtype=torch.long, device=rq.device)
        recovery_chosen = rq.gather(1, labels).squeeze(1)
        extra = ((rq + recovery_offsets).max(1).values - recovery_chosen).sum() / 117
        actual = (F.smooth_l1_loss(q.gather(1, batch.actions[:, None]).squeeze(1), returns.detach())
                  if returns is not None else q.new_zeros(()))
        return supervised + extra + actual, supervised, extra, actual
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    q = learner.online(batch.observations)
    loss, margin, extra, actual = losses(q, learner.online(recovery))
    accuracy = float((q.detach().argmax(1) == batch.actions).float().mean())
    loss.backward()
    norm = nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True)
    learner.optimizer.step()
    with torch.no_grad():
        aloss, amargin, aextra, aactual = losses(
            learner.online(batch.observations), learner.online(recovery))
        after = nn.utils.parameters_to_vector(learner.online.parameters())
        delta = torch.linalg.vector_norm(after - before)
    learner.update_count += 1
    return {
        "loss_before": loss.item(), "loss_after": aloss.item(),
        "demonstration_loss_before": margin.item(), "demonstration_loss_after": amargin.item(),
        "recovery_loss_before": extra.item(), "recovery_loss_after": aextra.item(),
        "return_loss_before": actual.item(), "return_loss_after": aactual.item(),
        "gradient_norm": norm.item(), "parameter_delta_norm": delta.item(),
        "demonstration_action_accuracy": accuracy, "target_synchronized": False,
    }
