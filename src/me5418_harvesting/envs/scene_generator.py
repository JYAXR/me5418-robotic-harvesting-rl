"""为训练集、验证集和测试集生成带随机种子的场景。"""

from dataclasses import dataclass

import numpy as np

from me5418_harvesting.envs.branch_physics import Branch


@dataclass
class Scene:
    end_effector: np.ndarray
    target: np.ndarray
    branches: list[Branch]


def generate_scene(
    rng: np.random.Generator,
    max_branches: int = 5,
    nodes_per_branch: int = 5,
) -> Scene:
    """生成可复现的受阻场景，并保留可行的外围绕行路径。"""
    branch_count = int(rng.integers(2, max_branches + 1))
    end_effector = np.array([0.10, 0.50], dtype=np.float32)
    target = np.array([0.90, rng.uniform(0.40, 0.60)], dtype=np.float32)
    branches: list[Branch] = []

    for x_coord in np.linspace(0.35, 0.72, branch_count):
        y_top = rng.uniform(0.72, 0.86)
        y_values = np.linspace(0.05, y_top, nodes_per_branch)
        x_values = np.full(nodes_per_branch, x_coord + rng.uniform(-0.02, 0.02))
        x_values += np.linspace(0.0, rng.uniform(-0.04, 0.04), nodes_per_branch)
        nodes = np.column_stack((x_values, y_values)).astype(np.float32)
        stiffness = float(rng.uniform(0.5, 2.0))
        safe_limit = float(rng.uniform(0.06, 0.12))
        branches.append(Branch(nodes.copy(), nodes.copy(), stiffness, safe_limit))

    return Scene(end_effector=end_effector, target=target, branches=branches)
