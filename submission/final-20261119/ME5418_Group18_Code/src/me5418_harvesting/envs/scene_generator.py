"""Generate seeded scenes for training, validation, and testing."""

from dataclasses import dataclass

import numpy as np

from me5418_harvesting.envs.branch_physics import Branch
from me5418_harvesting.envs.scene_validation import scene_issues


@dataclass
class Scene:
    end_effector: np.ndarray
    target: np.ndarray
    branches: list[Branch]
    passage_pairs: tuple[tuple[int, int], ...] = ()
    passage_kinds: tuple[str, ...] = ()
    layout: str = "tip_gates"
    generation_attempts: int = 1


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


def _generate_multi_route(
    rng: np.random.Generator, max_branches: int, nodes_per_branch: int,
    tip_gap_range, stiffness_range, safe_limit_range,
) -> Scene:
    """Create three two-lane decisions with open detours and material-dependent gaps.

    Each region contains four branches. The fixed central anchors leave less
    than one robot diameter between them, preventing a straight central bypass.
    An open lane always exists; its side changes at least once across regions.
    Passage labels are diagnostic metadata, not additional agent observations.
    """
    if max_branches < 14:
        raise ValueError("multi_route requires max_branches >= 14")
    position = np.array([0.08, rng.uniform(0.42, 0.58)], dtype=np.float32)
    target = np.array([0.92, rng.uniform(0.42, 0.58)], dtype=np.float32)
    branches = []
    kinds = []
    open_sides = rng.permutation([0, 1, int(rng.integers(2))])
    alternatives = rng.permutation(["pushable", "closed", "pushable"])
    parameter = np.linspace(0.0, 1.0, nodes_per_branch)

    def add_branch(anchor, tip, stiffness, safe_limit):
        anchor, tip = np.asarray(anchor), np.asarray(tip)
        nodes = anchor + parameter[:, None] * (tip - anchor)
        nodes[:, 0] += rng.uniform(-0.006, 0.006) * np.sin(np.pi * parameter)
        nodes = nodes.astype(np.float32)
        branches.append(Branch(nodes.copy(), nodes.copy(), stiffness, safe_limit))

    for region, base_x in enumerate((0.25, 0.50, 0.75)):
        x = base_x + rng.uniform(-0.015, 0.015)
        for lane in (0, 1):
            kind = "open" if lane == open_sides[region] else str(alternatives[region])
            kinds.append(kind)
            center = rng.uniform(0.255, 0.285) if lane == 0 else rng.uniform(0.715, 0.745)
            gap = (0.0 if kind == "closed" else rng.uniform(0.13, 0.17)
                   if kind == "open" else rng.uniform(*tip_gap_range))
            anchors = (0.0, 0.48) if lane == 0 else (0.52, 1.0)
            for side, anchor_y in enumerate(anchors):
                stiffness = 1e6 if kind == "closed" else float(rng.uniform(*stiffness_range))
                tip_y = center + (-0.5 if side == 0 else 0.5) * gap
                add_branch((x, anchor_y), (x, tip_y), stiffness,
                           float(rng.uniform(*safe_limit_range)))

    # Short corridor twigs penalize careless transfers between upper/lower lanes.
    # They remain separated from the region walls and do not span a whole corridor.
    for base_x in (0.375, 0.625):
        x = base_x + rng.uniform(-0.008, 0.008)
        y = rng.uniform(0.44, 0.56)
        direction = float(rng.choice((-1, 1)))
        add_branch((x - 0.02, y - 0.025 * direction),
                   (x + 0.02, y + 0.025 * direction),
                   float(rng.uniform(0.6, 2.5)), float(rng.uniform(0.035, 0.065)))
    return Scene(
        position, target, branches,
        passage_pairs=tuple((index, index + 1) for index in range(0, 12, 2)),
        passage_kinds=tuple(kinds), layout="multi_route",
    )


def _sample_scene(
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
    if layout == "multi_route":
        return _generate_multi_route(
            rng, max_branches, nodes_per_branch, tip_gap_range, stiffness_range, safe_limit_range,
        )
    if layout not in {"tip_gates", "easy_gate"}:
        raise ValueError(f"Unknown scene layout: {layout}")
    end_effector = np.array([0.10, 0.50], dtype=np.float32)
    target = np.array([0.90, rng.uniform(0.47, 0.53)], dtype=np.float32)
    branches: list[Branch] = []

    gate_count = 1 if layout == "easy_gate" else min(2, max_branches // 2)
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

    return Scene(
        end_effector=end_effector, target=target, branches=branches,
        passage_pairs=tuple((2 * index, 2 * index + 1) for index in range(gate_count)),
        passage_kinds=("open" if layout == "easy_gate" else "pushable",) * gate_count,
        layout=layout,
    )


def _tilt_branches(scene: Scene, rng: np.random.Generator, tilt_range) -> None:
    """Draw tilt after position jitter to retain matched upright/inclined scenes."""
    if scene.layout in {"mixed_canopy", "multi_route"} or tilt_range[1] == 0:
        return
    for branch in scene.branches:
        parameter = np.linspace(0.0, 1.0, len(branch.rest_nodes))
        tilt = np.deg2rad(rng.uniform(*tilt_range))
        lean_direction = float(rng.choice((-1.0, 1.0)))
        height = abs(float(branch.rest_nodes[-1, 1] - branch.rest_nodes[0, 1]))
        offset = lean_direction * height * np.tan(tilt)
        branch.rest_nodes[:, 0] += offset * (1.0 - parameter)
        branch.current_nodes = branch.rest_nodes.copy()


def _jitter_positions(
    scene: Scene, rng: np.random.Generator, start_jitter, target_jitter,
    gate_x_jitter: float, twig_jitter,
) -> None:
    """Translate complete branch pairs together, preserving their gap and shape.

    Zero amplitudes consume no random draws, preserving legacy seeded scenes.
    All changes precede validity checks; rejection sampling handles collisions.
    """
    for position, amplitude in (
        (scene.end_effector, start_jitter), (scene.target, target_jitter),
    ):
        if np.any(amplitude):
            position += rng.uniform(-amplitude, amplitude).astype(np.float32)
    paired = set()
    for pair_index, pair in enumerate(scene.passage_pairs):
        paired.update(pair)
        if gate_x_jitter > 0:
            # Keep both lanes and their central anchors aligned within each region.
            if scene.layout != "multi_route" or pair_index % 2 == 0:
                offset = np.float32(rng.uniform(-gate_x_jitter, gate_x_jitter))
            for index in pair:
                branch = scene.branches[index]
                branch.rest_nodes[:, 0] += offset
                branch.current_nodes = branch.rest_nodes.copy()
    if np.any(twig_jitter):
        for index, branch in enumerate(scene.branches):
            if index not in paired:
                offset = rng.uniform(-twig_jitter, twig_jitter).astype(np.float32)
                branch.rest_nodes += offset
                branch.current_nodes = branch.rest_nodes.copy()


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
    *,
    workspace_low=(0.0, 0.0),
    workspace_high=(1.0, 1.0),
    contact_radius: float = 0.035,
    target_radius: float = 0.050,
    max_attempts: int = 100,
    start_jitter: tuple[float, float] = (0.0, 0.0),
    target_jitter: tuple[float, float] = (0.0, 0.0),
    gate_x_jitter: float = 0.0,
    twig_jitter: tuple[float, float] = (0.0, 0.0),
) -> Scene:
    """Sample valid initial geometry, with deterministic bounded rejection sampling.

    Validation does not establish that a collision-free or deformable route exists.
    Geometry rejected on one attempt consumes RNG draws before the next attempt.
    """
    low, high = np.asarray(workspace_low), np.asarray(workspace_high)
    if (low.shape != (2,) or high.shape != (2,)
            or not np.isfinite([low, high]).all() or np.any(high <= low)):
        raise ValueError("Workspace bounds must be finite increasing coordinate pairs.")
    if (not np.isfinite([contact_radius, target_radius]).all()
            or contact_radius <= 0 or target_radius <= 0):
        raise ValueError("Contact and target radii must be finite and positive.")
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer.")
    amplitudes = []
    for name, values in (
        ("start_jitter", start_jitter), ("target_jitter", target_jitter),
        ("twig_jitter", twig_jitter),
    ):
        amplitude = np.asarray(values, dtype=float)
        if (amplitude.shape != (2,) or not np.isfinite(amplitude).all()
                or np.any(amplitude < 0)):
            raise ValueError(f"{name} must contain two finite nonnegative amplitudes.")
        amplitudes.append(amplitude)
    if not np.isfinite(gate_x_jitter) or gate_x_jitter < 0:
        raise ValueError("gate_x_jitter must be finite and nonnegative.")
    for name, bounds, positive in (
        ("gap_height_range", gap_height_range, False),
        ("tip_gap_range", tip_gap_range, True),
        ("stiffness_range", stiffness_range, True),
        ("safe_limit_range", safe_limit_range, True),
        ("branch_tilt_degrees_range", branch_tilt_degrees_range, False),
    ):
        if (len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[1] < bounds[0]
                or (positive and bounds[0] <= 0)):
            raise ValueError(f"Invalid finite ordered range: {name}")
    if layout == "easy_gate" and tip_gap_range[0] <= 2 * contact_radius:
        raise ValueError("easy_gate requires a tip gap wider than the robot diameter.")
    if layout == "multi_route" and not (
        0.04 < 2 * contact_radius < 0.13 and tip_gap_range[1] < 2 * contact_radius
    ):
        raise ValueError("multi_route requires blocked central anchors and open/narrow lane gaps.")
    for attempt in range(1, max_attempts + 1):
        scene = _sample_scene(
            rng, max_branches, nodes_per_branch, gap_height_range, tip_gap_range,
            stiffness_range, safe_limit_range, branch_tilt_degrees_range, layout,
        )
        _jitter_positions(scene, rng, amplitudes[0], amplitudes[1], gate_x_jitter, amplitudes[2])
        _tilt_branches(scene, rng, branch_tilt_degrees_range)
        issues = scene_issues(scene, low, high, contact_radius, target_radius)
        if not issues:
            scene.generation_attempts = attempt
            return scene
    raise ValueError(
        f"Could not generate valid scene geometry after {max_attempts} attempts. "
        f"Last rejection: {'; '.join(issues[:4])}"
    )
