"""Add one off-reference supervised action without changing the original TD batch."""
import importlib.util
from math import isfinite
from numbers import Real
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_learning import td_loss

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("recovery025_baseline", ROOT / "docs/experiments/022_half_td_local_supervision/run_local_supervision.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)
local, v = baseline.local, baseline.v


def validate(batch, recovery, weight):
    if isinstance(weight, bool) or weight not in (0.0, 1.0):
        raise ValueError("Only the predeclared recovery weights zero and one are allowed.")
    if len(batch.actions) != 117 or recovery.shape != (1, batch.observations.shape[1]):
        raise ValueError("Keep 117 original TD rows and one separate recovery observation.")
    if recovery.dtype != torch.float32 or not torch.isfinite(recovery).all():
        raise ValueError("The recovery observation must contain finite float32 values.")


def recovery_term(values):
    """Give the new action the contribution of one ordinary sample, without dilution."""
    actions = torch.full((len(values),), 4, dtype=torch.long, device=values.device)
    return local.margin_terms(values, actions).sum() / 117


def recovery_td_update(learner, batch, recovery, recovery_weight=1.0):
    validate(batch, recovery, recovery_weight)
    selected_weight = 2.0
    td_weight = 0.5
    if isinstance(selected_weight, bool) or selected_weight not in (1.0, 2.0):
        raise ValueError("The selected coefficient must be 1 or 2.")
    if (isinstance(td_weight, bool) or not isinstance(td_weight, Real)
            or not isfinite(td_weight) or td_weight not in (0.5, 1.0)):
        raise ValueError("TD weight must be the declared treatment 0.5 or control 1.0.")
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    td_before, _, targets = td_loss(learner.online, learner.target, batch, learner.config.gamma)
    values = learner.online(batch.observations)
    margin_before = local.weighted_margin_loss(values, batch.actions, 55, selected_weight)
    accuracy = float((values.detach().argmax(dim=1) == batch.actions).float().mean())
    # Keep the original graph exactly at coefficient one; no MC term is constructed.
    scaled_td = td_before if td_weight == 1.0 else td_weight * td_before
    extra_before = recovery_term(learner.online(recovery)) if recovery_weight else td_before.new_zeros(())
    loss = scaled_td + 1.0 * margin_before
    if recovery_weight:
        loss = loss + extra_before
    if not torch.isfinite(loss):
        raise FloatingPointError("The weighted TD objective is not finite.")
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True)
    learner.optimizer.step()
    with torch.no_grad():
        after_values = learner.online(batch.observations)
        after_chosen = after_values.gather(1, batch.actions[:, None]).squeeze(1)
        td_after = F.smooth_l1_loss(after_chosen, targets)
        margin_after = local.weighted_margin_loss(after_values, batch.actions, 55, selected_weight)
        scaled_after = td_after if td_weight == 1.0 else td_weight * td_after
        extra_after = recovery_term(learner.online(recovery)) if recovery_weight else td_after.new_zeros(())
        loss_after = scaled_after + 1.0 * margin_after
        if recovery_weight:
            loss_after = loss_after + extra_after
        after = nn.utils.parameters_to_vector(learner.online.parameters())
        delta = torch.linalg.vector_norm(after - before)
        if not torch.isfinite(after).all() or not torch.isfinite(loss_after):
            raise FloatingPointError("The weighted TD update produced non-finite values.")
    learner.update_count += 1
    synchronized = learner.update_count % learner.config.target_update_interval == 0
    if synchronized:
        learner.target.load_state_dict(learner.online.state_dict())
    return {
        "loss_before": loss.item(), "loss_after": loss_after.item(),
        "gradient_norm": gradient_norm.item(), "parameter_delta_norm": delta.item(),
        "target_synchronized": synchronized, "td_loss_before": td_before.item(),
        "td_loss_after": td_after.item(), "demonstration_loss_before": margin_before.item(),
        "demonstration_loss_after": margin_after.item(), "demonstration_action_accuracy": accuracy,
        "td_weight": float(td_weight), "selected_weight": float(selected_weight),
        "recovery_weight": float(recovery_weight), "recovery_loss_before": extra_before.item(), "recovery_loss_after": extra_after.item(),
    }


def independent_update(learner, batch, recovery, recovery_weight=1.0):
    """Compute the same objective without either production margin helper."""
    validate(batch, recovery, recovery_weight)
    weight = 2.0
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    chosen = learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    with torch.no_grad():
        following = learner.target(batch.next_observations).max(dim=1).values
        targets = batch.rewards + learner.config.gamma * torch.where(batch.terminated, torch.zeros_like(following), following)
    td = F.smooth_l1_loss(chosen, targets)
    values = learner.online(batch.observations)
    def margin(q):
        hinges = v.anchor.independent_hinges(q, batch.actions)
        coefficients = torch.ones_like(hinges)
        coefficients[55] = weight
        return (hinges * coefficients).mean()
    supervised = margin(values)
    accuracy = float((values.detach().argmax(1) == batch.actions).float().mean())
    def extra(q):
        offsets = torch.full_like(q, 0.8)
        offsets[:, 4] = 0.0
        return ((q + offsets).max(dim=1).values - q[:, 4]).sum() / 117
    extra_before = extra(learner.online(recovery)) if recovery_weight else td.new_zeros(())
    loss = 0.5 * td + supervised
    if recovery_weight:
        loss = loss + extra_before
    loss.backward()
    norm = nn.utils.clip_grad_norm_(learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True)
    learner.optimizer.step()
    with torch.no_grad():
        after_values = learner.online(batch.observations)
        after_td = F.smooth_l1_loss(after_values.gather(1, batch.actions[:, None]).squeeze(1), targets)
        after_margin = margin(after_values)
        extra_after = extra(learner.online(recovery)) if recovery_weight else after_td.new_zeros(())
        after_loss = 0.5 * after_td + after_margin
        if recovery_weight:
            after_loss = after_loss + extra_after
        after = nn.utils.parameters_to_vector(learner.online.parameters())
        delta = torch.linalg.vector_norm(after - before)
    learner.update_count += 1
    sync = learner.update_count % learner.config.target_update_interval == 0
    if sync:
        learner.target.load_state_dict(learner.online.state_dict())
    return {"loss_before": loss.item(), "loss_after": after_loss.item(),
            "gradient_norm": norm.item(), "parameter_delta_norm": delta.item(), "target_synchronized": sync,
            "td_loss_before": td.item(), "td_loss_after": after_td.item(),
            "demonstration_loss_before": supervised.item(), "demonstration_loss_after": after_margin.item(),
            "demonstration_action_accuracy": accuracy, "td_weight": 0.5, "selected_weight": float(weight),
            "recovery_weight": float(recovery_weight), "recovery_loss_before": extra_before.item(), "recovery_loss_after": extra_after.item()}

