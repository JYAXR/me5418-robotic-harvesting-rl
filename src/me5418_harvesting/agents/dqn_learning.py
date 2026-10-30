"""CPU data collection and DQN updates for the neural-network demonstration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

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


@dataclass(frozen=True)
class UpdateStats:
    """Diagnostics for one update; both losses use the same Bellman targets."""

    loss_before: float
    loss_after: float
    gradient_norm: float
    parameter_delta_norm: float
    target_synchronized: bool


class DQNUpdater:
    """Own online/target networks and execute independently verifiable DQN updates."""

    def __init__(self, input_dim: int, action_count: int, config: DQNConfig):
        self.config = config
        self.online = build_q_network(input_dim, action_count, config)
        self.target = deepcopy(self.online).requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.update_count = 0

    def update(self, batch: TransitionBatch) -> UpdateStats:
        """Backpropagate, clip gradients, update parameters, and periodically sync."""
        self.online.train()
        self.optimizer.zero_grad(set_to_none=True)
        parameters_before = nn.utils.parameters_to_vector(self.online.parameters()).detach().clone()
        loss, _, targets = td_loss(self.online, self.target, batch, self.config.gamma)
        if not torch.isfinite(loss):
            raise FloatingPointError("The DQN loss is not finite.")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        with torch.no_grad():
            chosen_q = self.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
            loss_after = F.smooth_l1_loss(chosen_q, targets)
            parameters_after = nn.utils.parameters_to_vector(self.online.parameters())
            delta = torch.linalg.vector_norm(parameters_after - parameters_before)
            if not torch.isfinite(parameters_after).all() or not torch.isfinite(loss_after):
                raise FloatingPointError("The DQN update produced non-finite parameters or loss.")
        self.update_count += 1
        synchronize = self.update_count % self.config.target_update_interval == 0
        if synchronize:
            self.target.load_state_dict(self.online.state_dict())
        return UpdateStats(
            loss.item(), loss_after.item(), gradient_norm.item(), delta.item(), synchronize
        )
