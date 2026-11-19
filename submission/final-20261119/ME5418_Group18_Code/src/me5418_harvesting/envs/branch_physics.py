"""Fixed-anchor elastic polylines with segment and contact constraints.

This is a geometric quasi-static surrogate, not a calibrated plant model.
Node springs restore the rest shape; positional projections enforce contact
clearance and limit segment strain. Stiffness limits deformation per substep.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Branch:
    rest_nodes: np.ndarray
    current_nodes: np.ndarray
    stiffness: float
    safe_limit: float
    in_contact: bool = False

    def __post_init__(self) -> None:
        self.rest_nodes = np.asarray(self.rest_nodes, dtype=np.float32)
        self.current_nodes = np.asarray(self.current_nodes, dtype=np.float32)
        if self.rest_nodes.shape != self.current_nodes.shape:
            raise ValueError("Rest and current nodes must have the same shape.")
        if self.rest_nodes.ndim != 2 or self.rest_nodes.shape[1] != 2:
            raise ValueError("Branch nodes must have shape (n_nodes, 2).")
        if len(self.rest_nodes) < 2:
            raise ValueError("A branch must have at least two nodes.")
        if self.stiffness <= 0 or self.safe_limit <= 0:
            raise ValueError("Stiffness and the safe deformation limit must be positive.")

    @property
    def max_displacement(self) -> float:
        displacement = np.linalg.norm(self.current_nodes - self.rest_nodes, axis=1)
        return float(displacement.max(initial=0.0))

    @property
    def normalized_deformation(self) -> float:
        return self.max_displacement / self.safe_limit

    def closest_points(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return each segment's closest point and its interpolation weight."""
        segment_starts = self.current_nodes[:-1]
        segment_vectors = self.current_nodes[1:] - segment_starts
        segment_lengths_sq = np.sum(segment_vectors * segment_vectors, axis=1)
        safe_lengths_sq = np.maximum(segment_lengths_sq, 1e-12)
        projection = np.sum((point - segment_starts) * segment_vectors, axis=1)
        projection = np.clip(projection / safe_lengths_sq, 0.0, 1.0)
        closest_points = segment_starts + projection[:, None] * segment_vectors
        return closest_points, projection

    def minimum_distance_to(self, point: np.ndarray) -> float:
        """Return the minimum distance from a point to the branch polyline."""
        closest, _ = self.closest_points(np.asarray(point, dtype=np.float32))
        return float(np.linalg.norm(closest - point, axis=1).min())

    @property
    def max_segment_strain(self) -> float:
        rest_lengths = np.linalg.norm(np.diff(self.rest_nodes, axis=0), axis=1)
        lengths = np.linalg.norm(np.diff(self.current_nodes, axis=0), axis=1)
        return float(np.max(np.abs(lengths / rest_lengths - 1.0)))

    def _project_lengths(self) -> None:
        """Keep each connected segment within 3 percent of its resting length."""
        rest_lengths = np.linalg.norm(np.diff(self.rest_nodes, axis=0), axis=1)
        for index, rest_length in enumerate(rest_lengths):
            vector = self.current_nodes[index + 1] - self.current_nodes[index]
            length = float(np.linalg.norm(vector))
            if length < 1e-8:
                continue
            allowed = float(np.clip(length, 0.97 * rest_length, 1.03 * rest_length))
            correction = vector * ((length - allowed) / length)
            if index == 0:
                self.current_nodes[index + 1] -= correction
            else:
                self.current_nodes[index] += 0.5 * correction
                self.current_nodes[index + 1] -= 0.5 * correction
        self.current_nodes[0] = self.rest_nodes[0]

    def update_contact(
        self,
        end_effector: np.ndarray,
        motion: np.ndarray,
        contact_radius: float,
        relaxation_rate: float,
    ) -> None:
        """Resolve normal contact, rather than shifting nodes along the action.

        At a narrow tip gap the normals point outward, so opposing branch tips
        separate to make room for the robot. A finite per-substep displacement
        allowance makes a stiff obstacle block the commanded movement.
        """
        before = self.current_nodes.copy()
        self.current_nodes[1:] += min(0.5, relaxation_rate * self.stiffness) * (
            self.rest_nodes[1:] - self.current_nodes[1:]
        )
        touched = False
        max_shift = 2.5 * max(float(np.linalg.norm(motion)), 0.001) / (1.0 + self.stiffness)
        for _ in range(24):
            iteration_start = self.current_nodes.copy()
            self._project_lengths()
            closest, fractions = self.closest_points(end_effector)
            for index in range(len(closest)):
                offset = closest[index] - end_effector
                distance = float(np.linalg.norm(offset))
                if distance >= contact_radius:
                    continue
                touched = True
                if distance < 1e-8:
                    # The substeps normally prevent this degeneracy. Use the
                    # previous robot position to retain its side of the branch.
                    offset = closest[index] - (end_effector - motion)
                normal = offset / max(float(np.linalg.norm(offset)), 1e-8)
                fraction = float(fractions[index])
                left_weight = 0.0 if index == 0 else 1.0 - fraction
                right_weight = fraction
                weight_sum = left_weight**2 + right_weight**2
                if weight_sum < 1e-8:
                    continue  # A fixed anchor cannot be displaced.
                correction = (contact_radius - distance) * normal / weight_sum
                self.current_nodes[index] += left_weight * correction
                self.current_nodes[index + 1] += right_weight * correction

            # Bound elastic motion, then restore the fixed anchor exactly.
            shifts = self.current_nodes - before
            lengths = np.linalg.norm(shifts, axis=1)
            scale = np.minimum(1.0, max_shift / np.maximum(lengths, 1e-10))
            self.current_nodes = before + shifts * scale[:, None]
            self.current_nodes[0] = self.rest_nodes[0]
            if (
                np.max(np.linalg.norm(self.current_nodes - iteration_start, axis=1)) < 1e-7
                and self.max_segment_strain < 0.03001
            ):
                break
        self.in_contact = touched
