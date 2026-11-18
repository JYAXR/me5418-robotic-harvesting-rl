"""Augment coordinate DQN observations with multi-route passage geometry."""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_learning import DQNUpdater
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ARCHIVE = Path(__file__).resolve().parents[1] / "research_results/20261118_generalization"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))
from coordinate_policy import CoordinateAgent, CoordinateEncoder, transform_encoded

BRANCHES = 14
NODES = 5
REGIONS = 3
LANES = 2
BRANCH_WIDTH = 4 * NODES + 4
BASE_SIZE = 5 + BRANCHES * BRANCH_WIDTH
# Six values per lane, one more copy for the next region, three distances and a validity flag.
GEOMETRY_SIZE = 6 * (REGIONS + 1) * LANES + REGIONS + 1


def _with_passage_geometry(values: np.ndarray, relative: bool) -> np.ndarray:
    """Append observed tips' centers and gaps, plus the nearest forward pair of lanes."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0 or values.shape[-1] != BASE_SIZE or not np.isfinite(values).all():
        raise ValueError("Expected finite normalized absolute multi-route observations.")
    flat = values.reshape(-1, BASE_SIZE)
    pairs = flat[:, 4:4 + REGIONS * LANES * 2 * BRANCH_WIDTH].reshape(
        -1, REGIONS, LANES, 2, BRANCH_WIDTH,
    )
    if not (pairs[..., -1] == 1.0).all():
        raise ValueError("Multi-route observations must contain all paired passage branches.")
    current_tips = pairs[..., 2 * NODES - 2:2 * NODES]
    rest_tips = pairs[..., 4 * NODES - 2:4 * NODES]
    robot = flat[:, None, None, :2]
    rest_center = rest_tips.mean(axis=-2) - robot
    current_center = current_tips.mean(axis=-2) - robot
    rest_width = np.linalg.norm(rest_tips[..., 1, :] - rest_tips[..., 0, :], axis=-1)
    current_width = np.linalg.norm(current_tips[..., 1, :] - current_tips[..., 0, :], axis=-1)
    lanes = np.concatenate((
        rest_center, rest_width[..., None], current_center, current_width[..., None],
    ), axis=-1)

    forward_distance = rest_center[..., 0].mean(axis=-1)
    ahead = forward_distance >= 0.0
    has_next = ahead.any(axis=-1, keepdims=True)
    next_region = np.where(ahead, forward_distance, np.inf).argmin(axis=-1)
    next_lanes = lanes[np.arange(len(flat)), next_region] * has_next[:, :, None]
    geometry = np.concatenate((
        lanes.reshape(-1, 6 * REGIONS * LANES), next_lanes.reshape(-1, 6 * LANES),
        np.where(ahead, forward_distance, 0.0), has_next.astype(np.float32),
    ), axis=-1)
    original = transform_encoded(values, relative, BRANCHES, NODES)
    return np.concatenate((original, geometry.reshape(*values.shape[:-1], GEOMETRY_SIZE)), axis=-1)


class GeometricCoordinateEncoder(CoordinateEncoder):
    def __init__(self, env: HarvestingEnv, relative: bool):
        if (env.scene_parameters.get("layout") != "multi_route"
                or (env.max_branches, env.nodes_per_branch) != (BRANCHES, NODES)):
            raise ValueError("Passage features require the 14-branch, 5-node multi-route layout.")
        super().__init__(env, relative)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return _with_passage_geometry(self.base(observation), self.relative)


class GeometricCoordinateAgent(CoordinateAgent):
    FORMAT = "geometric-coordinate-dqn-v1"

    def __init__(self, env: HarvestingEnv, config: DQNConfig, seed: int, relative: bool = True):
        self.relative = bool(relative)
        self.encoder = GeometricCoordinateEncoder(env, self.relative)
        self.action_count = int(env.action_space.n)
        self.learner = DQNUpdater(BASE_SIZE + GEOMETRY_SIZE, self.action_count, config)
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_baseline(cls, agent: CoordinateAgent, env: HarvestingEnv):
        """Preserve the baseline Q function and initialize a fresh optimizer."""
        with torch.random.fork_rng(devices=[]):
            expanded = cls(env, agent.learner.config, 0, agent.relative)
        for source, destination in (
            (agent.learner.online, expanded.learner.online),
            (agent.learner.target, expanded.learner.target),
        ):
            weights = source.state_dict()
            first_layer = weights["0.weight"]
            if first_layer.shape[1] != BASE_SIZE:
                raise ValueError("The source model must use the original coordinate encoding.")
            widened = torch.zeros_like(destination.state_dict()["0.weight"])
            widened[:, :BASE_SIZE] = first_layer
            weights["0.weight"] = widened
            destination.load_state_dict(weights)
        expanded.rng.bit_generator.state = deepcopy(agent.rng.bit_generator.state)
        return expanded


def transform_for_agent(values: np.ndarray, agent: CoordinateAgent) -> np.ndarray:
    """Convert normalized absolute training rows to the chosen agent's encoding."""
    if isinstance(agent, GeometricCoordinateAgent):
        return _with_passage_geometry(values, agent.relative)
    return transform_encoded(values, agent.relative, agent.encoder.base.branches,
                             agent.encoder.base.nodes)
