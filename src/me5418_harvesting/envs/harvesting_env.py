"""面向柔性障碍物采摘任务的 Gymnasium 环境接口。"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from me5418_harvesting.envs.scene_generator import Scene, generate_scene


class HarvestingEnv(gym.Env[np.ndarray, int]):
    """完全可观测的确定性或随机马尔可夫环境。"""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["ansi"]}

    _DIRECTIONS = np.array(
        [[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.float32
    )

    def __init__(self, config: dict[str, Any] | None = None, render_mode: str | None = None):
        super().__init__()
        cfg = config or {}
        self.max_branches = int(cfg.get("max_branches", 5))
        self.nodes_per_branch = int(cfg.get("nodes_per_branch", 5))
        self.max_steps = int(cfg.get("max_steps", 100))
        self.small_step = float(cfg.get("small_step", 0.025))
        self.large_step = float(cfg.get("large_step", 0.060))
        self.target_radius = float(cfg.get("target_radius", 0.050))
        self.contact_radius = float(cfg.get("contact_radius", 0.035))
        self.relaxation_rate = float(cfg.get("relaxation_rate", 0.10))
        self.noise_fraction = float(cfg.get("execution_noise_fraction", 0.0))
        self.workspace_low = np.asarray(cfg.get("workspace_low", [0.0, 0.0]), dtype=np.float32)
        self.workspace_high = np.asarray(cfg.get("workspace_high", [1.0, 1.0]), dtype=np.float32)
        self.reward_weights = {
            "progress": 10.0,
            "deformation": 1.0,
            "travel": 0.2,
            "damage": 50.0,
            "success": 100.0,
            **cfg.get("reward", {}),
        }
        self.render_mode = render_mode

        branch_features = 4 * self.nodes_per_branch + 4
        self.observation_size = 5 + self.max_branches * branch_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(8)
        self.scene: Scene | None = None
        self.step_count = 0
        self.total_travel = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.scene = generate_scene(self.np_random, self.max_branches, self.nodes_per_branch)
        self.step_count = 0
        self.total_travel = 0.0
        return self._observation(), {"seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.scene is None:
            raise RuntimeError("调用 step() 前必须先调用 reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"无效动作：{action}")

        direction = self._DIRECTIONS[action % 4]
        step_size = self.small_step if action < 4 else self.large_step
        commanded_motion = direction * step_size
        noise_bound = self.noise_fraction * step_size
        noise = self.np_random.uniform(-noise_bound, noise_bound, size=2).astype(np.float32)
        executed_motion = commanded_motion + noise

        old_distance = float(np.linalg.norm(self.scene.target - self.scene.end_effector))
        new_position = np.clip(
            self.scene.end_effector + executed_motion, self.workspace_low, self.workspace_high
        )
        actual_motion = new_position - self.scene.end_effector
        self.scene.end_effector = new_position

        for branch in self.scene.branches:
            branch.update_contact(
                self.scene.end_effector,
                actual_motion,
                self.contact_radius,
                self.relaxation_rate,
            )

        self.step_count += 1
        travel = float(np.linalg.norm(actual_motion))
        self.total_travel += travel
        new_distance = float(np.linalg.norm(self.scene.target - self.scene.end_effector))
        deformations = [branch.normalized_deformation for branch in self.scene.branches]
        total_deformation = float(sum(deformations))
        max_deformation = float(max(deformations, default=0.0))
        success = new_distance <= self.target_radius
        damage = max_deformation > 1.0
        terminated = success or damage
        truncated = self.step_count >= self.max_steps and not terminated

        reward = (
            self.reward_weights["progress"] * (old_distance - new_distance)
            - self.reward_weights["deformation"] * total_deformation
            - self.reward_weights["travel"] * travel
            - self.reward_weights["damage"] * float(damage)
            + self.reward_weights["success"] * float(success and not damage)
        )
        info = {
            "success": success,
            "damage": damage,
            "distance_to_target": new_distance,
            "travel_distance": self.total_travel,
            "max_normalized_deformation": max_deformation,
            "executed_motion": actual_motion.copy(),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        if self.scene is None:
            raise RuntimeError("场景尚未初始化")
        features: list[float] = [
            *self.scene.end_effector,
            *(self.scene.target - self.scene.end_effector),
        ]
        for branch_index in range(self.max_branches):
            if branch_index < len(self.scene.branches):
                branch = self.scene.branches[branch_index]
                features.extend(branch.current_nodes.reshape(-1))
                features.extend(branch.rest_nodes.reshape(-1))
                features.extend(
                    [branch.stiffness, float(branch.in_contact), branch.safe_limit, 1.0]
                )
            else:
                features.extend([0.0] * (4 * self.nodes_per_branch + 4))
        features.append(1.0 - self.step_count / self.max_steps)
        return np.asarray(features, dtype=np.float32)

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self.scene is None:
            return None
        distance = np.linalg.norm(self.scene.target - self.scene.end_effector)
        return f"step={self.step_count} distance={distance:.3f} branches={len(self.scene.branches)}"
