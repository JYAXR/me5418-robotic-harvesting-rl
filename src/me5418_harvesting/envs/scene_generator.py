"""Generate seeded scenes for training, validation, and testing."""

from dataclasses import dataclass

import numpy as np

from me5418_harvesting.envs.branch_physics import Branch


@dataclass
class Scene:
    end_effector: np.ndarray
    target: np.ndarray
    branches: list[Branch]
    passage_pairs: tuple[tuple[int, int], ...] = ()
    passage_kinds: tuple[str, ...] = ()
    layout: str = "tip_gates"


def _generate_mixed_canopy(
    rng: np.random.Generator,
    max_branches: int,
    nodes_per_branch: int,
    tip_gap_range: tuple[float, float],
    stiffness_range: tuple[float, float],
    safe_limit_range: tuple[float, float],
) -> Scene:
    """Mix closed, naturally open, and deformable passages with local clutter.

    Closed passages have touching tips and effectively rigid branches. They
    cannot be cleared by the small elastic displacements of the soft passages.
    All obstacles use the same segment-contact solver during execution.
    """
    if max_branches < 14:
        raise ValueError("mixed_canopy requires max_branches >= 14")
    position = np.array([0.08, rng.uniform(0.47, 0.53)], dtype=np.float32)
    target = np.array([0.92, rng.uniform(0.46, 0.54)], dtype=np.float32)
    branches: list[Branch] = []
    parameter = np.linspace(0.0, 1.0, nodes_per_branch)

    def add_branch(anchor, tip, stiffness, safe_limit, bow_scale=0.010):
        anchor = np.asarray(anchor, dtype=np.float64)
        tip = np.asarray(tip, dtype=np.float64)
        tangent = tip - anchor
        normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
        bow = float(rng.uniform(-bow_scale, bow_scale))
        nodes = anchor + parameter[:, None] * tangent
        nodes += bow * np.sin(np.pi * parameter)[:, None] * normal
        nodes = nodes.astype(np.float32)
        branches.append(Branch(nodes.copy(), nodes.copy(), stiffness, safe_limit))

    # The closed and open middle passages swap locations across seeds; there
    # is no universally valid "push all gaps" or "follow one side" route.
    middle = ("closed", "open") if rng.integers(2) == 0 else ("open", "closed")
    kinds = ("pushable", *middle, "pushable")
    for index, (x_coord, kind) in enumerate(zip((0.24, 0.42, 0.61, 0.79), kinds)):
        center_y = float(rng.uniform(0.39, 0.61))
        gap = 0.0 if kind == "closed" else (
            float(rng.uniform(0.115, 0.150)) if kind == "open"
            else float(rng.uniform(*tip_gap_range))
        )
        for side in (0, 1):
            below = side == 0
            interior = (index % 2 == 0 and not below) or (index % 2 == 1 and below)
            anchor_y = (
                float(rng.uniform(0.14, 0.18)) if interior else 0.0
            ) if below else (
                float(rng.uniform(0.83, 0.87)) if interior else 1.0
            )
            anchor_x = x_coord + float(rng.uniform(-0.023, 0.023))
            tip_y = center_y + (-0.5 if below else 0.5) * gap
            stiffness = 1e6 if kind == "closed" else float(rng.uniform(*stiffness_range))
            safe_limit = float(rng.uniform(*safe_limit_range))
            add_branch((anchor_x, anchor_y), (x_coord, tip_y), stiffness, safe_limit)

    # Short oblique, curved, and near-horizontal twigs occupy approach areas,
    # inter-passage corridors, and the fruit neighbourhood, not only the edge.
    templates = (
        ((0.035, 0.78), (0.13, 0.67)),
        ((0.09, 0.12), (0.16, 0.29)),
        ((0.28, 0.30), (0.315, 0.30)),
        ((0.52, 0.28), (0.55, 0.40)),
        ((0.97, 0.28), (0.885, 0.35)),
        ((0.97, 0.74), (0.865, 0.63)),
    )
    for anchor, tip in templates:
        jitter = rng.uniform(-0.004, 0.004, size=2)
        stiffness = float(rng.uniform(0.6, 2.5))
        safe_limit = float(rng.uniform(0.035, 0.065))
        add_branch(np.asarray(anchor) + jitter, np.asarray(tip) + jitter, stiffness, safe_limit)

    return Scene(
        position, target, branches,
        passage_pairs=((0, 1), (2, 3), (4, 5), (6, 7)),
        passage_kinds=kinds,
        layout="mixed_canopy",
    )


def generate_scene(
    rng: np.random.Generator,
    max_branches: int = 5,
    nodes_per_branch: int = 5,
    gap_height_range: tuple[float, float] = (0.40, 0.60),
    tip_gap_range: tuple[float, float] = (0.028, 0.043),
    stiffness_range: tuple[float, float] = (0.35, 1.10),
    safe_limit_range: tuple[float, float] = (0.075, 0.110),
    branch_tilt_degrees_range: tuple[float, float] = (0.0, 0.0),
    layout: str = "tip_gates",
) -> Scene:
    """Create opposing tips that form narrow, deformable inner passages.

    The first gate has a lower wall anchor and an interior upper anchor; the
    second gate reverses those roles. A rigid-geometry route snakes around the
    interior anchors, while safe contact can open the shorter central tip gaps.
    """
    if max_branches < 2 or nodes_per_branch < 2:
        raise ValueError("Use at least two branches and two nodes per branch")
    tilt_low, tilt_high = branch_tilt_degrees_range
    if not 0.0 <= tilt_low <= tilt_high <= 25.0:
        raise ValueError("Branch tilt magnitudes must satisfy 0 <= low <= high <= 25 degrees")
    if layout == "mixed_canopy":
        return _generate_mixed_canopy(
            rng, max_branches, nodes_per_branch, tip_gap_range, stiffness_range, safe_limit_range
        )
    if layout != "tip_gates":
        raise ValueError(f"Unknown scene layout: {layout}")
    end_effector = np.array([0.10, 0.50], dtype=np.float32)
    target = np.array([0.90, rng.uniform(0.47, 0.53)], dtype=np.float32)
    branches: list[Branch] = []

    gate_count = min(2, max_branches // 2)
    gate_positions = [0.50] if gate_count == 1 else [0.33, 0.67]
    for gate_index, x_coord in enumerate(gate_positions):
        center_y = float(rng.uniform(*gap_height_range))
        tip_gap = float(rng.uniform(*tip_gap_range))
        for side in (0, 1):
            anchored_below = side == 0
            interior_anchor = (gate_index == 0 and not anchored_below) or (
                gate_index == 1 and anchored_below
            )
            if anchored_below:
                anchor_y = float(rng.uniform(0.16, 0.21)) if interior_anchor else 0.0
                tip_y = center_y - tip_gap / 2.0
            else:
                anchor_y = float(rng.uniform(0.79, 0.84)) if interior_anchor else 1.0
                tip_y = center_y + tip_gap / 2.0
            parameter = np.linspace(0.0, 1.0, nodes_per_branch)
            # A shallow bow permits bending and removes exact collinearity.
            bow = float(rng.uniform(-0.018, 0.018))
            x_values = x_coord + bow * np.sin(np.pi * parameter)
            y_values = anchor_y + parameter * (tip_y - anchor_y)
            nodes = np.column_stack((x_values, y_values)).astype(np.float32)
            stiffness = float(rng.uniform(*stiffness_range))
            safe_limit = float(rng.uniform(*safe_limit_range))
            branches.append(Branch(nodes.copy(), nodes.copy(), stiffness, safe_limit))

    # Draw tilt only after the base scene is complete. With a matched seed,
    # the inclined and upright variants retain identical gaps, stiffness,
    # safety limits, and target; only the branch x geometry changes.
    if tilt_high > 0.0:
        parameter = np.linspace(0.0, 1.0, nodes_per_branch)
        for branch in branches:
            tilt = np.deg2rad(rng.uniform(tilt_low, tilt_high))
            lean_direction = float(rng.choice((-1.0, 1.0)))
            height = abs(float(branch.rest_nodes[-1, 1] - branch.rest_nodes[0, 1]))
            offset = lean_direction * height * np.tan(tilt)
            branch.rest_nodes[:, 0] += offset * (1.0 - parameter)
            branch.current_nodes = branch.rest_nodes.copy()

    return Scene(
        end_effector=end_effector, target=target, branches=branches,
        passage_pairs=tuple((2 * index, 2 * index + 1) for index in range(gate_count)),
        passage_kinds=("pushable",) * gate_count,
    )
