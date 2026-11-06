"""Observation preprocessing, exploration, and portable DQN policy files."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_learning import DQNUpdater
from me5418_harvesting.envs.harvesting_env import HarvestingEnv


class ObservationEncoder:
    """Use fixed physical scales, with log stiffness and unchanged binary masks."""

    VERSION = 1

    def __init__(self, env: HarvestingEnv):
        self.low = env.workspace_low.copy()
        self.span = env.workspace_high - env.workspace_low
        if not np.isfinite(self.span).all() or np.any(self.span <= 0):
            raise ValueError("Workspace spans must be finite and positive.")
        self.branches = env.max_branches
        self.nodes = env.nodes_per_branch
        self.size = env.observation_size

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32).copy()
        if values.shape != (self.size,) or not np.isfinite(values).all():
            raise ValueError("Observation has an invalid shape or non-finite values.")
        values[:2] = (values[:2] - self.low) / self.span
        values[2:4] /= self.span
        width = 4 * self.nodes + 4
        for index in range(self.branches):
            block = values[4 + index * width:4 + (index + 1) * width]
            if block[-1] == 0:
                block[:] = 0
                continue
            coordinates = block[:4 * self.nodes].reshape(-1, 2)
            coordinates[:] = (coordinates - self.low) / self.span
            # A stiffness of 1e6 maps to approximately one, without clipping.
            block[-4] = np.log1p(max(float(block[-4]), 0.0)) / np.log1p(1e6)
            block[-2] /= float(np.max(self.span))
        return values


class DQNAgent:
    """Connect raw environment observations to a learnable discrete-action policy."""

    def __init__(self, env: HarvestingEnv, config: DQNConfig, seed: int):
        self.encoder = ObservationEncoder(env)
        self.action_count = int(env.action_space.n)
        self.learner = DQNUpdater(env.observation_size, self.action_count, config)
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between zero and one.")
        encoded = self.encoder(observation)
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        self.learner.online.eval()
        with torch.no_grad():
            values = self.learner.online(torch.from_numpy(encoded).unsqueeze(0))
        if not torch.isfinite(values).all():
            raise FloatingPointError("The policy produced non-finite action values.")
        return int(values.argmax(dim=1).item())

    def save(self, path: Path, environment_config: dict, metadata: dict):
        """Atomically save an inference checkpoint, not a training-resume snapshot."""
        temporary = path.with_suffix(".tmp")
        torch.save({
            "format_version": 1,
            "encoder_version": ObservationEncoder.VERSION,
            "environment_config": environment_config,
            "dqn_config": asdict(self.learner.config),
            "online": self.learner.online.state_dict(),
            "metadata": metadata,
        }, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path):
        """Recreate the matching environment and preprocessing for evaluation."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1 or payload["encoder_version"] != ObservationEncoder.VERSION:
            raise ValueError("Unsupported policy or observation encoder version.")
        env = HarvestingEnv(payload["environment_config"])
        agent = cls(env, DQNConfig(**payload["dqn_config"]), seed=0)
        agent.learner.online.load_state_dict(payload["online"])
        agent.learner.target.load_state_dict(payload["online"])
        return agent, env, payload["metadata"]
