"""CPU data collection and DQN updates shared by demonstration and training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from me5418_harvesting.agents.dqn import DQNConfig, build_q_network
from me5418_harvesting.envs.harvesting_env import HarvestingEnv


@dataclass(frozen=True)
class TransitionBatch:
    """Keep task termination and external truncation separate for bootstrapping."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor

    def sample(self, size: int, rng: np.random.Generator) -> TransitionBatch:
        """Sample distinct transitions using an explicit reproducible RNG."""
        if not 0 < size <= len(self.actions):
            raise ValueError("Batch size must be positive and not exceed the collected transitions.")
        indices = torch.from_numpy(rng.choice(len(self.actions), size, replace=False))
        return TransitionBatch(
            self.observations[indices], self.actions[indices], self.rewards[indices],
            self.next_observations[indices], self.terminated[indices], self.truncated[indices],
        )


def collect_transitions(env: HarvestingEnv, count: int, seed: int) -> TransitionBatch:
    """Collect random-policy experience, retaining final observations before reset."""
    if count <= 0:
        raise ValueError("Transition count must be positive.")
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed)
    records = []
    for _ in range(count):
        action = int(rng.integers(env.action_space.n))
        next_observation, reward, terminated, truncated, _ = env.step(action)
        records.append((
            observation.copy(), action, reward, next_observation.copy(), terminated, truncated,
        ))
        # The stored next state belongs to the old episode, never to a new reset.
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()
    observations, actions, rewards, next_observations, terminated, truncated = zip(*records)
    return TransitionBatch(
        torch.from_numpy(np.stack(observations)),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(rewards, dtype=torch.float32),
        torch.from_numpy(np.stack(next_observations)),
        torch.tensor(terminated, dtype=torch.bool),
        torch.tensor(truncated, dtype=torch.bool),
    )


def td_loss(
    online: nn.Module, target: nn.Module, batch: TransitionBatch, gamma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Huber loss, chosen-action Q values, and detached Bellman targets."""
    chosen_q = online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_q = target(batch.next_observations).max(dim=1).values
        # Success, damage, or task-budget exhaustion ends future reward.
        # Only an external truncation of an unfinished task still bootstraps.
        next_q = torch.where(batch.terminated, torch.zeros_like(next_q), next_q)
        targets = batch.rewards + gamma * next_q
    return F.smooth_l1_loss(chosen_q, targets), chosen_q, targets


def validate_demonstration_loss(weight: float, margin: float) -> None:
    """Reject invalid supervision settings before updating model or optimizer state."""
    if isinstance(weight, bool) or not isinstance(weight, Real) or not isfinite(weight) or weight < 0:
        raise ValueError("Demonstration weight must be finite and nonnegative.")
    if isinstance(margin, bool) or not isinstance(margin, Real) or not isfinite(margin) or margin <= 0:
        raise ValueError("Demonstration margin must be finite and positive.")


def demonstration_margin_loss(
    q_values: torch.Tensor, actions: torch.Tensor, margin: float,
) -> torch.Tensor:
    """Prefer each demonstrated action by a fixed Q-value margin.

    Uses the large-margin objective described by Hester et al., AAAI 2018:
    https://arxiv.org/abs/1704.03732. The surrounding learner remains ordinary
    one-step DQN; this objective alone does not implement the full DQfD method.
    """
    validate_demonstration_loss(0.0, margin)
    if (q_values.ndim != 2 or q_values.shape[0] == 0 or actions.ndim != 1
            or actions.shape[0] != q_values.shape[0]):
        raise ValueError("Expected nonempty batched Q values and one demonstration action per row.")
    offsets = torch.full_like(q_values, margin)
    offsets.scatter_(1, actions[:, None], 0.0)
    demonstrated_q = q_values.gather(1, actions[:, None]).squeeze(1)
    return ((q_values + offsets).max(dim=1).values - demonstrated_q).mean()


@dataclass(frozen=True)
class UpdateStats:
    """Diagnostics for one update; both losses use the same Bellman targets."""

    loss_before: float
    loss_after: float
    gradient_norm: float
    parameter_delta_norm: float
    target_synchronized: bool
    td_loss_before: float = 0.0
    td_loss_after: float = 0.0
    demonstration_loss_before: float = 0.0
    demonstration_loss_after: float = 0.0
    demonstration_action_accuracy: float | None = None


class DQNUpdater:
    """Own online/target networks and execute independently verifiable DQN updates."""

    def __init__(self, input_dim: int, action_count: int, config: DQNConfig):
        self.config = config
        self.online = build_q_network(input_dim, action_count, config)
        self.target = deepcopy(self.online).requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.update_count = 0

    def update(
        self, batch: TransitionBatch, *,
        demonstration_weight: float = 0.0, demonstration_margin: float = 0.8,
    ) -> UpdateStats:
        """Update DQN; positive supervision weight requires a demonstration-only batch."""
        validate_demonstration_loss(demonstration_weight, demonstration_margin)
        self.online.train()
        self.optimizer.zero_grad(set_to_none=True)
        parameters_before = nn.utils.parameters_to_vector(self.online.parameters()).detach().clone()
        loss, _, targets = td_loss(self.online, self.target, batch, self.config.gamma)
        td_before = loss
        demonstration_before = loss.new_zeros(())
        accuracy = None
        if demonstration_weight > 0:
            q_values = self.online(batch.observations)
            demonstration_before = demonstration_margin_loss(
                q_values, batch.actions, demonstration_margin,
            )
            accuracy = float((q_values.detach().argmax(dim=1) == batch.actions).float().mean())
            loss = loss + demonstration_weight * demonstration_before
        if not torch.isfinite(loss):
            raise FloatingPointError("The DQN loss is not finite.")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        with torch.no_grad():
            q_values_after = self.online(batch.observations)
            chosen_q = q_values_after.gather(1, batch.actions[:, None]).squeeze(1)
            td_after = F.smooth_l1_loss(chosen_q, targets)
            loss_after = td_after
            demonstration_after = loss_after.new_zeros(())
            if demonstration_weight > 0:
                demonstration_after = demonstration_margin_loss(
                    q_values_after, batch.actions, demonstration_margin,
                )
                loss_after = loss_after + demonstration_weight * demonstration_after
            parameters_after = nn.utils.parameters_to_vector(self.online.parameters())
            delta = torch.linalg.vector_norm(parameters_after - parameters_before)
            if not torch.isfinite(parameters_after).all() or not torch.isfinite(loss_after):
                raise FloatingPointError("The DQN update produced non-finite parameters or loss.")
        self.update_count += 1
        synchronize = self.update_count % self.config.target_update_interval == 0
        if synchronize:
            self.target.load_state_dict(self.online.state_dict())
        return UpdateStats(
            loss.item(), loss_after.item(), gradient_norm.item(), delta.item(), synchronize,
            td_before.item(), td_after.item(),
            demonstration_before.item(), demonstration_after.item(), accuracy,
        )
