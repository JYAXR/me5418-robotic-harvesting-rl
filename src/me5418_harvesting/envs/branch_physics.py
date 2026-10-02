"""Lightweight branch model used by the initial environment scaffold."""

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
            raise ValueError("rest_nodes and current_nodes must have the same shape")
        if self.rest_nodes.ndim != 2 or self.rest_nodes.shape[1] != 2:
            raise ValueError("branch nodes must have shape (n_nodes, 2)")
        if self.stiffness <= 0 or self.safe_limit <= 0:
            raise ValueError("stiffness and safe_limit must be positive")

    @property
    def max_displacement(self) -> float:
        displacement = np.linalg.norm(self.current_nodes - self.rest_nodes, axis=1)
        return float(displacement.max(initial=0.0))

    @property
    def normalized_deformation(self) -> float:
        return self.max_displacement / self.safe_limit

    def update_contact(
        self,
        end_effector: np.ndarray,
        motion: np.ndarray,
        contact_radius: float,
        relaxation_rate: float,
    ) -> None:
        """Apply a simple contact displacement followed by elastic relaxation.

        This is an intentionally transparent MVP approximation. It should be
        replaced by the validated quasi-static solver used for final results.
        """
        distances = np.linalg.norm(self.current_nodes - end_effector, axis=1)
        nearest = int(np.argmin(distances))
        self.in_contact = bool(distances[nearest] <= contact_radius)

        if self.in_contact and nearest > 0:
            compliance = 1.0 / (1.0 + self.stiffness)
            self.current_nodes[nearest] += compliance * motion

        self.current_nodes[1:] += relaxation_rate * (
            self.rest_nodes[1:] - self.current_nodes[1:]
        )
        self.current_nodes[0] = self.rest_nodes[0]

