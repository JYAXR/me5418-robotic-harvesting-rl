"""Geometry-aware residual policy with an immutable inherited action prior."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent, ObservationEncoder
from me5418_harvesting.envs.harvesting_env import HarvestingEnv


class GeometryEncoder(ObservationEncoder):
    """Append commanded-path geometry, not a prediction of deformable physics."""

    VERSION = 2
    EXTRA = 40

    def __init__(self, env):
        super().__init__(env)
        self.small = env.small_step
        self.large = env.large_step
        self.radius = env.contact_radius
        self.directions = env._DIRECTIONS.copy()

    def extend(self, encoded):
        base = np.asarray(encoded, dtype=np.float32)
        if base.shape != (self.size,) or not np.isfinite(base).all():
            raise ValueError("Expected one finite encoded base observation.")
        # Use double precision for relative distances after workspace translation.
        robot = self.low.astype(np.float64) + base[:2].astype(np.float64) * self.span
        target = robot + base[2:4].astype(np.float64) * self.span
        moves = np.concatenate((self.directions * self.small, self.directions * self.large))
        points = robot[None, None, :] + moves[:, None, :] * np.linspace(0, 1, 13)[None, :, None]
        current = np.full(8, np.inf)
        rest = np.full(8, np.inf)
        nearest_stiffness = np.zeros(8)
        width = 4 * self.nodes + 4
        for i in range(self.branches):
            block = base[4 + i * width:4 + (i + 1) * width]
            if block[-1] == 0:
                continue
            shapes = block[:4 * self.nodes].reshape(2, self.nodes, 2) * self.span + self.low
            for kind, nodes in enumerate(shapes):
                start, vector = nodes[:-1], np.diff(nodes, axis=0)
                offset = points[:, :, None, :] - start
                fraction = np.clip((offset * vector).sum(-1) / np.maximum((vector * vector).sum(-1), 1e-12), 0, 1)
                distance = np.linalg.norm(offset - fraction[..., None] * vector, axis=-1).min(axis=(1, 2))
                if kind == 0:
                    nearer = distance < current
                    nearest_stiffness[nearer] = block[-4]
                    current = np.minimum(current, distance)
                else:
                    rest = np.minimum(rest, distance)
        end = robot + moves
        boundary = np.minimum(points - self.low - self.radius,
                              self.low + self.span - self.radius - points).min(axis=(1, 2))
        progress = (np.linalg.norm(target - robot) - np.linalg.norm(target - end, axis=1)) / self.large
        features = np.stack((np.clip((current - self.radius) / self.large, -2, 2),
                             np.clip((rest - self.radius) / self.large, -2, 2),
                             np.clip(boundary / self.large, -2, 2), progress,
                             np.clip(nearest_stiffness, 0, 2)), axis=1)
        return np.concatenate((base, features.reshape(-1).astype(np.float32)))

    def __call__(self, observation):
        return self.extend(super().__call__(observation))


class ResidualPolicy(nn.Module):
    """Learn policy corrections while keeping the original network fixed."""

    def __init__(self, inherited, base_size):
        super().__init__()
        self.base_size = base_size
        self.inherited = inherited.requires_grad_(False)
        self.correction = nn.Sequential(nn.Linear(base_size + GeometryEncoder.EXTRA, 128),
                                        nn.ReLU(), nn.Linear(128, 8))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def prior(self, observations):
        q = self.inherited(observations[:, :self.base_size])
        # Preserve initial greedy choices without treating absolute Q units as logits.
        return (q - q.max(dim=1, keepdim=True).values).clamp(min=-4.0)

    def forward(self, observations):
        return self.prior(observations) + self.correction(observations)


class RecoveryAgent(DQNAgent):
    """Inference-compatible experimental policy; no automatic TD updates."""

    def __init__(self, env, config, seed):
        super().__init__(env, config, seed)
        self.encoder = GeometryEncoder(env)
        self.learner.online = ResidualPolicy(self.learner.online, env.observation_size)
        self.learner.target = deepcopy(self.learner.online).requires_grad_(False)
        self.learner.optimizer = torch.optim.Adam(self.learner.online.correction.parameters(), lr=1e-4)

    @classmethod
    def from_source(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1 or payload["encoder_version"] != 1:
            raise ValueError("Expected an original DQN source checkpoint.")
        env = HarvestingEnv(payload["environment_config"])
        agent = cls(env, DQNConfig(**payload["dqn_config"]), seed=0)
        agent.learner.online.inherited.load_state_dict(payload["online"])
        agent.learner.target.load_state_dict(agent.learner.online.state_dict())
        return agent, env, payload

    def save(self, path: Path, environment_config, metadata):
        temporary = path.with_suffix(".tmp")
        torch.save({"format_version": 2, "encoder_version": 2, "kind": "geometry_residual_policy",
                    "environment_config": environment_config, "dqn_config": asdict(self.learner.config),
                    "online": self.learner.online.state_dict(), "metadata": metadata}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload["format_version"] != 2 or payload["encoder_version"] != 2
                or payload.get("kind") != "geometry_residual_policy"):
            raise ValueError("Unsupported residual policy checkpoint.")
        env = HarvestingEnv(payload["environment_config"])
        agent = cls(env, DQNConfig(**payload["dqn_config"]), seed=0)
        agent.learner.online.load_state_dict(payload["online"])
        agent.learner.target.load_state_dict(payload["online"])
        return agent, env, payload["metadata"]
