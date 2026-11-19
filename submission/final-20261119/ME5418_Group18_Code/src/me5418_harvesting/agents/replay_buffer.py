"""Bounded replay storage shared across training episodes."""

import numpy as np
import torch

from me5418_harvesting.agents.dqn_learning import TransitionBatch


class ReplayBuffer:
    """Copy transitions on insertion and sample without replacement."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = capacity
        self.records = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.records)

    def add(self, observation, action, reward, next_observation, terminated, truncated):
        record = (
            np.asarray(observation, dtype=np.float32).copy(), int(action), float(reward),
            np.asarray(next_observation, dtype=np.float32).copy(),
            bool(terminated), bool(truncated),
        )
        if len(self.records) < self.capacity:
            self.records.append(record)
        else:
            self.records[self.position] = record
        self.position = (self.position + 1) % self.capacity

    def sample(self, size: int, rng: np.random.Generator) -> TransitionBatch:
        if not 0 < size <= len(self):
            raise ValueError("Batch size must be positive and not exceed replay size.")
        indices = rng.choice(len(self), size, replace=False)
        columns = list(zip(*(self.records[int(index)] for index in indices)))
        return TransitionBatch(
            torch.from_numpy(np.stack(columns[0])),
            torch.tensor(columns[1], dtype=torch.long),
            torch.tensor(columns[2], dtype=torch.float32),
            torch.from_numpy(np.stack(columns[3])),
            torch.tensor(columns[4], dtype=torch.bool),
            torch.tensor(columns[5], dtype=torch.bool),
        )
