"""Initial-scene geometry checks; these do not prove task reachability."""

from itertools import combinations

import numpy as np

TOLERANCE = 1e-6


def _cross(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def _intersects(a, b, c, d):
    """Include endpoint contact and collinear overlap in segment intersection."""
    if np.any(np.maximum(np.minimum(a, b), np.minimum(c, d)) >
              np.minimum(np.maximum(a, b), np.maximum(c, d)) + TOLERANCE):
        return False
    first = (_cross(b - a, c - a), _cross(b - a, d - a))
    second = (_cross(d - c, a - c), _cross(d - c, b - c))
    return all(not (min(pair) > TOLERANCE or max(pair) < -TOLERANCE)
               for pair in (first, second))


def _only_shared_endpoint(a, endpoint, c):
    """Permit an endpoint join, but reject branches that overlap along a line."""
    first, second = a - endpoint, c - endpoint
    return abs(_cross(first, second)) > TOLERANCE or float(first @ second) < 0


def scene_issues(scene, workspace_low, workspace_high, contact_radius, target_radius):
    """Return reasons to reject a freshly generated scene, without mutating it.

    Branches have zero thickness in this model. Only the opposing final tips
    of an explicitly closed passage may touch across two different branches.
    """
    low = np.asarray(workspace_low, dtype=float)
    high = np.asarray(workspace_high, dtype=float)
    issues = []
    for name, point, radius in (
        ("Start", scene.end_effector, contact_radius),
        ("Target", scene.target, max(contact_radius, target_radius)),
    ):
        point = np.asarray(point)
        if point.shape != (2,) or not np.isfinite(point).all():
            issues.append(f"{name} position must contain two finite coordinates.")
        elif np.any(point - radius < low - TOLERANCE) or np.any(
            point + radius > high + TOLERANCE
        ):
            issues.append(f"{name} clearance region extends outside the workspace.")
    if issues:
        return issues
    if np.linalg.norm(scene.end_effector - scene.target) <= target_radius + TOLERANCE:
        issues.append("Start is already inside the success region.")
    if not scene.branches:
        issues.append("The scene must contain at least one branch.")

    for index, branch in enumerate(scene.branches):
        nodes = branch.current_nodes
        if (nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) < 2
                or not np.isfinite(nodes).all() or not np.isfinite(branch.rest_nodes).all()):
            issues.append(f"Branch {index} has invalid node coordinates.")
            continue
        if not np.array_equal(nodes, branch.rest_nodes):
            issues.append(f"Branch {index} is initially deformed.")
        if np.any(nodes < low - TOLERANCE) or np.any(nodes > high + TOLERANCE):
            issues.append(f"Branch {index} extends outside the workspace.")
        if np.any(np.linalg.norm(np.diff(nodes, axis=0), axis=1) <= TOLERANCE):
            issues.append(f"Branch {index} contains a zero-length segment.")
        if (not np.isfinite([branch.stiffness, branch.safe_limit]).all()
                or branch.stiffness <= 0 or branch.safe_limit <= 0):
            issues.append(f"Branch {index} has invalid material parameters.")
        for name, point, radius in (
            ("Start", scene.end_effector, contact_radius),
            ("Target", scene.target, max(contact_radius, target_radius)),
        ):
            if branch.minimum_distance_to(point) < radius - TOLERANCE:
                issues.append(f"{name} clearance region overlaps branch {index}.")
    if issues:
        return issues

    closed_pairs = {
        tuple(sorted(pair)) for pair, kind in zip(scene.passage_pairs, scene.passage_kinds)
        if kind == "closed"
    }
    segments = [
        (index, segment, nodes[segment], nodes[segment + 1])
        for index, branch in enumerate(scene.branches)
        for nodes in (branch.current_nodes.astype(float),)
        for segment in range(len(nodes) - 1)
    ]
    for (i, si, a, b), (j, sj, c, d) in combinations(segments, 2):
        if not _intersects(a, b, c, d):
            continue
        # Neighbouring segments share a node, but must not double back on it.
        if i == j and sj == si + 1 and _only_shared_endpoint(a, b, d):
            continue
        if (
            (i, j) in closed_pairs
            and si == len(scene.branches[i].current_nodes) - 2
            and sj == len(scene.branches[j].current_nodes) - 2
            and np.linalg.norm(b - d) <= TOLERANCE
            and _only_shared_endpoint(a, b, c)
        ):
            continue
        issues.append(f"Branch segments ({i}, {si}) and ({j}, {sj}) intersect or overlap.")
    return issues
