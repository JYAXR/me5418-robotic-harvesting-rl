"""Combine complete demonstrations without carrying return targets across resets."""

from collections.abc import Sequence
from numbers import Integral

import torch

from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.return_calibration import discounted_demonstration_returns

FIELDS = tuple(TransitionBatch.__dataclass_fields__)


def slice_transition_batch(batch: TransitionBatch, start: int, stop: int) -> TransitionBatch:
    """Copy a nonempty contiguous range without sharing storage or autograd history."""
    if not isinstance(batch, TransitionBatch):
        raise TypeError("Expected a TransitionBatch to slice.")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (start, stop)):
        raise ValueError("Batch slice bounds must be integers.")
    if not isinstance(batch.actions, torch.Tensor) or batch.actions.ndim != 1:
        raise ValueError("Batch actions must be a one-dimensional tensor.")
    count = len(batch.actions)
    if not 0 <= start < stop <= count:
        raise ValueError("Batch slice bounds must select a nonempty in-range interval.")
    if any(not isinstance(getattr(batch, name), torch.Tensor)
           or getattr(batch, name).ndim == 0 or len(getattr(batch, name)) != count
           for name in FIELDS):
        raise ValueError("Every batch field must have the same transition count.")
    return TransitionBatch(*(
        getattr(batch, name)[int(start):int(stop)].detach().clone() for name in FIELDS
    ))


def build_multiscene_batch(
    trajectories: Sequence[TransitionBatch], seeds: Sequence[int], gamma: float,
) -> tuple[TransitionBatch, torch.Tensor, list[dict]]:
    """Concatenate complete ordered episodes and compute their returns separately.

    Every trajectory must end at its only task termination and contain no
    external truncation. Per-episode offsets preserve that boundary after
    concatenation. Each transition appears once, so a mean loss over the result
    weights scenes by trajectory length rather than balancing scenes equally.
    Returned tensors are detached copies; no input tensor is modified or aliased.
    Metadata returns sum the stored rewards, which may already be float32-rounded.
    """
    if not isinstance(trajectories, Sequence) or not isinstance(seeds, Sequence):
        raise TypeError("Trajectories and seeds must be finite sequences.")
    if not trajectories or len(trajectories) != len(seeds):
        raise ValueError("Provide one seed for each nonempty demonstration collection.")
    if any(isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 for seed in seeds):
        raise ValueError("Scene seeds must be nonnegative integers.")
    if len({int(seed) for seed in seeds}) != len(seeds):
        raise ValueError("Each demonstration must have a distinct scene seed.")

    episode_returns, episodes = [], []
    first = None
    offset = 0
    for seed, trajectory in zip(seeds, trajectories):
        if not isinstance(trajectory, TransitionBatch):
            raise TypeError("Every demonstration must be a TransitionBatch.")
        if any(not isinstance(getattr(trajectory, name), torch.Tensor) for name in FIELDS):
            raise ValueError("Every demonstration field must be a tensor.")
        if (not trajectory.observations.is_floating_point()
                or trajectory.actions.dtype != torch.long
                or trajectory.actions.device != trajectory.rewards.device
                or bool((trajectory.actions < 0).any())):
            raise ValueError("Expected floating-point observations and nonnegative long actions.")
        targets = discounted_demonstration_returns(trajectory, gamma)
        if first is None:
            first = trajectory
        else:
            for name in FIELDS:
                current, reference = getattr(trajectory, name), getattr(first, name)
                if (current.shape[1:] != reference.shape[1:] or current.dtype != reference.dtype
                        or current.device != reference.device):
                    raise ValueError("All episodes must have matching field layouts, dtypes and devices.")
        count = len(trajectory.actions)
        episodes.append({
            "seed": int(seed), "start": offset, "stop": offset + count, "transitions": count,
            "undiscounted_return": sum(float(reward) for reward in trajectory.rewards.tolist()),
            "initial_discounted_return": float(targets[0]),
            "minimum_return": float(targets.min()), "maximum_return": float(targets.max()),
        })
        episode_returns.append(targets)
        offset += count

    combined = TransitionBatch(*(
        torch.cat([getattr(trajectory, name).detach() for trajectory in trajectories], dim=0)
        for name in FIELDS
    ))
    return combined, torch.cat(episode_returns, dim=0).detach(), episodes
