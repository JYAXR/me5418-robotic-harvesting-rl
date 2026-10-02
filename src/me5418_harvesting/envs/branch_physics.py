"""初始环境骨架使用的轻量枝条模型。"""

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
            raise ValueError("静止节点与当前节点必须具有相同形状")
        if self.rest_nodes.ndim != 2 or self.rest_nodes.shape[1] != 2:
            raise ValueError("枝条节点形状必须为 (n_nodes, 2)")
        if self.stiffness <= 0 or self.safe_limit <= 0:
            raise ValueError("刚度和安全形变上限必须为正数")

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
        """施加简化的接触位移，然后执行弹性恢复。

        这是一个刻意保持透明的 MVP 近似。最终实验应将其替换为经过验证的
        准静态求解器。
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
