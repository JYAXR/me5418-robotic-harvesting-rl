"""Add local waypoint and swept-clearance observations to the learned policy."""

import numpy as np
from geometric_coordinate_policy import (
    BASE_SIZE,
    BRANCH_WIDTH,
    BRANCHES,
    GEOMETRY_SIZE,
    NODES,
    GeometricCoordinateAgent,
    GeometricCoordinateEncoder,
    _with_passage_geometry,
)

from me5418_harvesting.agents.dqn_learning import DQNUpdater

GUIDE_SIZE = 8 + 3 * 8


class GuidedCoordinateEncoder(GeometricCoordinateEncoder):
    """Read lane alignment and action clearance from the observed polylines."""

    def __init__(self, env, relative):
        super().__init__(env, relative)
        self.span = self.base.span
        self.radius = env.contact_radius
        self.large_step = env.large_step
        self.offsets = np.concatenate((
            env._DIRECTIONS * env.small_step, env._DIRECTIONS * env.large_step,
        ))
        points = np.linspace(0.25, 1.0, 4, dtype=np.float32)
        self.sweeps = self.offsets[:, None, :] * points[None, :, None]

    def __call__(self, observation):
        return self.from_absolute(self.base(observation))

    def from_absolute(self, values):
        original = _with_passage_geometry(values, self.relative)
        flat = np.asarray(values).reshape(-1, BASE_SIZE)
        guide = np.empty((len(flat), GUIDE_SIZE), dtype=np.float32)
        for start in range(0, len(flat), 128):
            guide[start:start + 128] = self._features(flat[start:start + 128])
        return np.concatenate((original, guide.reshape(*original.shape[:-1], GUIDE_SIZE)), axis=-1)

    def _features(self, flat):
        robot = flat[:, :2] * self.span
        target = robot + flat[:, 2:4] * self.span
        nodes = flat[:, 4:-1].reshape(-1, BRANCHES, BRANCH_WIDTH)
        current = nodes[..., :2 * NODES].reshape(-1, BRANCHES, NODES, 2) * self.span
        rest = nodes[..., 2 * NODES:4 * NODES].reshape(-1, BRANCHES, NODES, 2) * self.span
        gate_current = current[:, :12].reshape(-1, 3, 2, 2, NODES, 2)
        gate_rest = rest[:, :12].reshape(-1, 3, 2, 2, NODES, 2)
        extent = np.concatenate((gate_current[..., 0], gate_rest[..., 0]), axis=-1)
        margin = 1.1 * self.radius
        entry = extent.min(axis=(2, 3, 4)) - margin
        exit_x = extent.max(axis=(2, 3, 4)) + margin
        pending = robot[:, 0, None] < exit_x
        has_gate = pending.any(axis=1)
        region = pending.argmax(axis=1)
        row = np.arange(len(flat))
        tips = gate_rest[row, region, :, :, -1, 1]
        lane = np.abs(tips[..., 1] - tips[..., 0]).argmax(axis=1)
        current_tips = gate_current[row, region, lane, :, -1, 1]
        low = current_tips.min(axis=1) + margin
        high = current_tips.max(axis=1) - margin
        center = current_tips.mean(axis=1)

        waypoint = target.copy()
        waypoint[has_gate, 0] = exit_x[row, region][has_gate]
        waypoint[has_gate, 1] = center[has_gate]
        offset = waypoint - robot
        alignment = np.maximum(low - robot[:, 1], 0) - np.maximum(robot[:, 1] - high, 0)
        # Note the inter-gate twig bounding box while approaching a gate after the first one.
        twig = current[row, 12 + np.maximum(region - 1, 0)]
        before = twig[..., 0].min(axis=1) - margin - robot[:, 0]
        after = twig[..., 0].max(axis=1) + margin - robot[:, 0]
        has_twig = has_gate & (region > 0)
        goal = np.column_stack((
            offset, np.where(has_gate, alignment, 0),
            np.where(has_gate, entry[row, region] - robot[:, 0], 0),
            np.where(has_gate, high - low, 0),
            np.where(has_twig, before, 0), np.where(has_twig, after, 0),
            self.large_step * has_gate,
        )) / self.large_step

        remaining = offset[:, None, :] - self.offsets
        progress = (np.abs(offset).sum(axis=1, keepdims=True) - np.abs(remaining).sum(axis=2))
        clearance, endpoint_clearance = self._swept_clearances(robot, current)
        return np.clip(np.concatenate((
            goal, progress / self.large_step,
            clearance / self.radius - 1, endpoint_clearance / self.radius - 1,
        ), axis=1), -3, 3)

    def _swept_clearances(self, robot, nodes):
        points = robot[:, None, None, :] + self.sweeps
        start = nodes[:, :, :-1].reshape(len(robot), -1, 2)
        vector = nodes[:, :, 1:].reshape(len(robot), -1, 2) - start
        relative = points[..., None, :] - start[:, None, None, ...]
        fraction = np.sum(relative * vector[:, None, None, ...], axis=-1)
        squared_length = np.maximum(np.sum(vector**2, axis=-1), 1e-10)
        fraction = np.clip(fraction / squared_length[:, None, None, :], 0, 1)
        nearest = relative - fraction[..., None] * vector[:, None, None, ...]
        branch_distance = np.sqrt(np.sum(nearest**2, axis=-1).min(axis=-1))
        wall_distance = np.minimum(points, self.span - points).min(axis=-1)
        clearance = np.minimum(branch_distance, wall_distance)
        return clearance.min(axis=2), clearance[:, :, -1]


class GuidedCoordinateAgent(GeometricCoordinateAgent):
    """Keep the DQN action rule and widen the input with local navigation geometry."""

    FORMAT = "guided-coordinate-dqn-v1"

    def __init__(self, env, config, seed, relative=True):
        self.relative = bool(relative)
        self.encoder = GuidedCoordinateEncoder(env, self.relative)
        self.action_count = int(env.action_space.n)
        self.learner = DQNUpdater(BASE_SIZE + GEOMETRY_SIZE + GUIDE_SIZE, self.action_count, config)
        self.rng = np.random.default_rng(seed)


class CompactGuidedEncoder(GuidedCoordinateEncoder):
    """Retain robot, target, budget, and geometry in place of raw branch slots."""

    def from_absolute(self, values):
        features = super().from_absolute(values)
        return np.concatenate((features[..., :4], features[..., BASE_SIZE - 1:]), axis=-1)


class CompactGuidedAgent(GuidedCoordinateAgent):
    """Train a DQN using only the compact geometric representation."""

    FORMAT = "compact-guided-dqn-v1"

    def __init__(self, env, config, seed, relative=True):
        self.relative = bool(relative)
        self.encoder = CompactGuidedEncoder(env, self.relative)
        self.action_count = int(env.action_space.n)
        self.learner = DQNUpdater(5 + GEOMETRY_SIZE + GUIDE_SIZE, self.action_count, config)
        self.rng = np.random.default_rng(seed)
