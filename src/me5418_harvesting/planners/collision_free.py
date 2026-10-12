"""A* over the same eight cardinal actions used by the RL environment."""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from me5418_harvesting.envs.harvesting_env import HarvestingEnv


class NoCollisionFreeRoute(RuntimeError):
    """The action-grid planner found no route at its required clearance."""


def collision_free_actions(
    env: HarvestingEnv,
    *,
    destination: np.ndarray | None = None,
    goal_radius: float | None = None,
    include_current_shape: bool = False,
) -> list[int]:
    """Find a route around resting branches without changing their geometry.

    A fine 0.005 grid exactly represents the default 0.025/0.060 actions. Every
    intermediate grid point is checked, with extra clearance that covers the
    distance between samples. The route cannot cut corners between waypoints.
    """
    if env.scene is None:
        raise RuntimeError("Reset the environment before planning")
    goal = env.scene.target if destination is None else np.asarray(destination)
    tolerance = env.target_radius if goal_radius is None else goal_radius
    grid = math.gcd(round(env.small_step * 10000), round(env.large_step * 10000)) / 10000
    origin = env.scene.end_effector.astype(np.float64)
    indices_x = np.arange(
        math.ceil((env.workspace_low[0] + env.contact_radius - origin[0]) / grid),
        math.floor((env.workspace_high[0] - env.contact_radius - origin[0]) / grid) + 1,
    )
    indices_y = np.arange(
        math.ceil((env.workspace_low[1] + env.contact_radius - origin[1]) / grid),
        math.floor((env.workspace_high[1] - env.contact_radius - origin[1]) / grid) + 1,
    )
    xx, yy = np.meshgrid(indices_x, indices_y, indexing="ij")
    points = np.stack((xx * grid + origin[0], yy * grid + origin[1]), axis=-1)
    minimum_distances = np.full(xx.shape, np.inf)
    for branch in env.scene.branches:
        shapes = [branch.rest_nodes]
        if include_current_shape:
            shapes.append(branch.current_nodes)
        for nodes in shapes:
            for start, finish in itertools.pairwise(nodes):
                vector = finish - start
                fraction = np.clip(
                    np.sum((points - start) * vector, axis=-1) / float(vector @ vector),
                    0.0,
                    1.0,
                )
                closest = start + fraction[..., None] * vector
                minimum_distances = np.minimum(
                    minimum_distances, np.linalg.norm(points - closest, axis=-1)
                )
    free = minimum_distances >= env.contact_radius + grid * 0.6
    x_offset, y_offset = int(indices_x[0]), int(indices_y[0])
    action_offsets = [
        (int(dx * round(size / grid)), int(dy * round(size / grid)))
        for size in (env.small_step, env.large_step)
        for dx, dy in env._DIRECTIONS
    ]

    def clear_edge(state: tuple[int, int], offset: tuple[int, int]) -> bool:
        dx, dy = offset
        count = max(abs(dx), abs(dy))
        for index in range(1, count + 1):
            x = state[0] + dx * index // count - x_offset
            y = state[1] + dy * index // count - y_offset
            if x < 0 or y < 0 or x >= free.shape[0] or y >= free.shape[1] or not free[x, y]:
                return False
        return True

    def heuristic(state: tuple[int, int]) -> float:
        position = origin + np.array(state) * grid
        return max(0.0, float(np.linalg.norm(position - goal)) - tolerance)

    start_state = (0, 0)
    queue = [(heuristic(start_state), 0, start_state)]
    sequence = itertools.count(1)
    costs = {start_state: 0.0}
    previous: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    settled: set[tuple[int, int]] = set()
    while queue:
        _, _, state = heapq.heappop(queue)
        if state in settled:
            continue
        settled.add(state)
        if heuristic(state) <= 0.0:
            actions = []
            while state != start_state:
                state, action = previous[state]
                actions.append(action)
            return actions[::-1]
        for action, offset in enumerate(action_offsets):
            next_state = (state[0] + offset[0], state[1] + offset[1])
            if not clear_edge(state, offset):
                continue
            new_cost = costs[state] + (env.small_step if action < 4 else env.large_step)
            if new_cost >= costs.get(next_state, float("inf")) - 1e-9:
                continue
            costs[next_state] = new_cost
            previous[next_state] = (state, action)
            heapq.heappush(queue, (new_cost + heuristic(next_state), next(sequence), next_state))
    raise NoCollisionFreeRoute("No collision-free action-grid route found at required clearance")
