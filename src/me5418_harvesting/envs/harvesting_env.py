"""Gymnasium environment for harvesting among flexible obstacles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from me5418_harvesting.envs.scene_generator import Scene, generate_scene


@dataclass(frozen=True)
class _MotionResult:
    """Executed displacement and diagnostics for one action."""

    executed_motion: np.ndarray
    collision_blocked: bool
    contact_occurred: bool
    minimum_clearance: float


class HarvestingEnv(gym.Env[np.ndarray, int]):
    """Fully observable Markov environment with deterministic or stochastic transitions."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 8,
    }

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
        self.scene_parameters = cfg.get("scene", {})
        self.show_gate_detail = bool(cfg.get("show_gate_detail", True))
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
        self.peak_deformation = 0.0
        self.contact_steps = 0
        self.trajectory: list[np.ndarray] = []
        self._figure: Any | None = None
        self._axes: Any | None = None
        self.demo_policy = ""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.scene = generate_scene(
            self.np_random, self.max_branches, self.nodes_per_branch, **self.scene_parameters
        )
        self.step_count = 0
        self.total_travel = 0.0
        self.peak_deformation = 0.0
        self.contact_steps = 0
        self.trajectory = [self.scene.end_effector.copy()]
        return self._observation(), {"seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an action, resolve contact, and evaluate the transition."""
        if self.scene is None:
            raise RuntimeError("Call reset() before calling step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        commanded_motion, executed_motion = self._action_motion(action)
        old_distance = float(np.linalg.norm(self.scene.target - self.scene.end_effector))
        motion = self._simulate_motion(executed_motion)
        return self._finish_step(old_distance, commanded_motion, motion)

    def _action_motion(self, action: int) -> tuple[np.ndarray, np.ndarray]:
        """Decode the action and sample bounded execution noise."""
        direction = self._DIRECTIONS[action % 4]
        step_size = self.small_step if action < 4 else self.large_step
        commanded_motion = direction * step_size
        noise_bound = self.noise_fraction * step_size
        noise = self.np_random.uniform(-noise_bound, noise_bound, size=2).astype(np.float32)
        executed_motion = commanded_motion + noise

        return commanded_motion, executed_motion

    def _simulate_motion(self, executed_motion: np.ndarray) -> _MotionResult:
        """Resolve short motions to prevent tunnelling through thin branches."""
        old_position = self.scene.end_effector.copy()
        substep_limit = max(1e-4, self.contact_radius * 0.15)
        substep_count = max(
            1,
            int(np.ceil(float(np.linalg.norm(executed_motion)) / substep_limit)),
        )
        substep_motion = executed_motion / substep_count
        collision_blocked = False
        contact_occurred = False
        current_position = old_position.copy()
        step_minimum_clearance = min(
            branch.minimum_distance_to(current_position) for branch in self.scene.branches
        )
        for _ in range(substep_count):
            previous_shapes = [branch.current_nodes.copy() for branch in self.scene.branches]
            candidate_position, substep_contact = self._try_contact_substep(
                current_position, substep_motion, substep_count, previous_shapes
            )
            contact_occurred |= substep_contact
            if candidate_position is None:
                collision_blocked = True
                break

            candidate_motion = candidate_position - current_position
            overlapping = self._overlaps_branch(candidate_position)
            if overlapping:
                collision_blocked = True
                # If finite compliance cannot make room for the full substep,
                # accept only its collision-free fraction. Keep the same side
                # of the obstacle; a contact update must not displace material
                # through the robot's previously valid position either.
                if self._overlaps_branch(current_position):
                    self._restore_branch_shapes(previous_shapes)
                    break
                lower_fraction = self._collision_free_fraction(current_position, candidate_motion)
                current_position += lower_fraction * candidate_motion
            else:
                current_position = candidate_position
            self.peak_deformation = max(
                self.peak_deformation,
                max(branch.normalized_deformation for branch in self.scene.branches),
            )
            step_minimum_clearance = min(
                step_minimum_clearance,
                min(branch.minimum_distance_to(current_position) for branch in self.scene.branches),
            )
            if self.peak_deformation > 1.0:
                break

        self.scene.end_effector = current_position
        actual_motion = current_position - old_position
        self.trajectory.append(current_position.copy())

        return _MotionResult(
            actual_motion, collision_blocked, contact_occurred, step_minimum_clearance
        )

    def _try_contact_substep(
        self,
        current_position: np.ndarray,
        substep_motion: np.ndarray,
        substep_count: int,
        previous_shapes: list[np.ndarray],
    ) -> tuple[np.ndarray | None, bool]:
        """Halve difficult substeps until contact and segment strain agree."""
        contact_occurred = False
        for attempt in range(8):
            fraction = 0.5**attempt
            candidate_position = np.clip(
                current_position + fraction * substep_motion,
                self.workspace_low + self.contact_radius,
                self.workspace_high - self.contact_radius,
            )
            candidate_motion = candidate_position - current_position
            for branch, shape in zip(self.scene.branches, previous_shapes):
                branch.current_nodes = shape.copy()
                branch.update_contact(
                    candidate_position,
                    candidate_motion,
                    self.contact_radius,
                    fraction * self.relaxation_rate / substep_count,
                )
                contact_occurred |= branch.in_contact
            if all(branch.max_segment_strain <= 0.0301 for branch in self.scene.branches):
                return candidate_position, contact_occurred
        self._restore_branch_shapes(previous_shapes)
        return None, contact_occurred

    def _restore_branch_shapes(self, previous_shapes: list[np.ndarray]) -> None:
        """Restore node positions after an infeasible contact update."""
        for branch, shape in zip(self.scene.branches, previous_shapes):
            branch.current_nodes = shape

    def _overlaps_branch(self, position: np.ndarray) -> bool:
        """Check penetration using the same tolerance as the contact solver."""
        return any(
            branch.minimum_distance_to(position) < self.contact_radius - 1e-5
            for branch in self.scene.branches
        )

    def _collision_free_fraction(
        self, current_position: np.ndarray, candidate_motion: np.ndarray
    ) -> float:
        """Bisect a blocked substep to retain its collision-free fraction."""
        lower_fraction, upper_fraction = 0.0, 1.0
        for _ in range(12):
            middle_fraction = 0.5 * (lower_fraction + upper_fraction)
            middle_position = current_position + middle_fraction * candidate_motion
            if all(
                branch.minimum_distance_to(middle_position) >= self.contact_radius - 1e-5
                for branch in self.scene.branches
            ):
                lower_fraction = middle_fraction
            else:
                upper_fraction = middle_fraction
        return lower_fraction

    def _finish_step(
        self, old_distance: float, commanded_motion: np.ndarray, motion: _MotionResult
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Update episode counters, reward, termination, and diagnostics."""
        self.step_count += 1
        self.contact_steps += int(motion.contact_occurred)
        travel = float(np.linalg.norm(motion.executed_motion))
        self.total_travel += travel
        new_distance = float(np.linalg.norm(self.scene.target - self.scene.end_effector))
        deformations = [branch.normalized_deformation for branch in self.scene.branches]
        total_deformation = float(sum(deformations))
        max_deformation = float(max(deformations, default=0.0))
        damage = self.peak_deformation > 1.0
        success = new_distance <= self.target_radius and not damage
        budget_exhausted = self.step_count >= self.max_steps
        # The observed action budget defines the task's finite horizon.
        terminated = success or damage or budget_exhausted
        # External wrappers may truncate an unfinished task before this horizon.
        truncated = False

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
            "budget_exhausted": budget_exhausted,
            "distance_to_target": new_distance,
            "travel_distance": self.total_travel,
            "max_normalized_deformation": max_deformation,
            "peak_normalized_deformation": self.peak_deformation,
            "contact_steps": self.contact_steps,
            "minimum_substep_clearance": motion.minimum_clearance,
            "minimum_clearance": min(
                branch.minimum_distance_to(self.scene.end_effector)
                for branch in self.scene.branches
            ),
            "collision_blocked": motion.collision_blocked,
            "commanded_motion": commanded_motion.copy(),
            "executed_motion": motion.executed_motion.copy(),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        if self.scene is None:
            raise RuntimeError("The scene has not been initialized.")
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

    def render(self) -> str | np.ndarray | None:
        """Render a textual status, an RGB frame, or an interactive window."""
        if self.scene is None:
            return None
        if self.render_mode == "ansi":
            distance = np.linalg.norm(self.scene.target - self.scene.end_effector)
            return (
                f"step={self.step_count} distance={distance:.3f} "
                f"branches={len(self.scene.branches)}"
            )
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        if self.render_mode == "human":
            self._render_human()
            return None
        return None

    def _draw_scene(self, axes: Any) -> None:
        if self.scene is None:
            return

        from matplotlib.patches import Circle

        axes.set_xlim(float(self.workspace_low[0]), float(self.workspace_high[0]))
        axes.set_ylim(float(self.workspace_low[1]), float(self.workspace_high[1]))
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel("x position")
        axes.set_ylabel("y position")
        label = f" | {self.demo_policy}" if self.demo_policy else ""
        title = "Mixed-canopy harvesting" if self.scene.layout == "mixed_canopy" else "Deformable tip-gap harvesting"
        axes.set_title(f"{title} | step {self.step_count}{label}", fontsize=12)
        axes.grid(alpha=0.15)

        for index, branch in enumerate(self.scene.branches):
            label = "rest shape" if index == 0 else None
            axes.plot(
                branch.rest_nodes[:, 0],
                branch.rest_nodes[:, 1],
                linestyle="--",
                color="#9ca3af",
                linewidth=1.2,
                label=label,
            )
            axes.scatter(
                branch.current_nodes[0, 0],
                branch.current_nodes[0, 1],
                marker="s",
                s=28,
                color="#111827",
                zorder=4,
                label="fixed anchor" if index == 0 else None,
            )
            label = "flexible branch" if index == 0 else None
            color = "#dc2626" if branch.in_contact else (
                "#92400e" if branch.stiffness >= 1e4 else "#2f855a"
            )
            axes.plot(
                branch.current_nodes[:, 0],
                branch.current_nodes[:, 1],
                marker="o",
                markersize=4,
                color=color,
                linewidth=2.2,
                label=label,
            )

        if self.trajectory:
            trajectory = np.asarray(self.trajectory)
            axes.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="#2563eb",
                linewidth=1.4,
                alpha=0.75,
                label="trajectory",
            )

        axes.add_patch(
            Circle(
                self.scene.target,
                self.target_radius,
                facecolor="#f59e0b",
                edgecolor="#92400e",
                alpha=0.75,
                label="target region",
            )
        )
        axes.add_patch(
            Circle(
                self.scene.end_effector,
                self.contact_radius,
                facecolor="#2563eb",
                edgecolor="#1e3a8a",
                label="end effector",
            )
        )
        axes.legend(loc="lower right", fontsize=7)
        axes.text(
            0.02,
            0.03,
            f"travel = {self.total_travel:.2f}\npeak deformation / limit = {self.peak_deformation:.2f}",
            transform=axes.transAxes,
            fontsize=8,
            color="#374151",
        )
        if self.scene.layout == "mixed_canopy":
            for number, ((first, second), kind) in enumerate(
                zip(self.scene.passage_pairs, self.scene.passage_kinds), start=1
            ):
                center = 0.5 * (
                    self.scene.branches[first].rest_nodes[-1]
                    + self.scene.branches[second].rest_nodes[-1]
                )
                color = {"pushable": "#1d4ed8", "closed": "#92400e", "open": "#047857"}[kind]
                axes.text(
                    center[0] + 0.012, center[1] + 0.025,
                    f"P{number}: {kind}", fontsize=6.5, color=color,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1},
                )
        if self.show_gate_detail and self.scene.passage_pairs:
            self._draw_gate_detail(axes)

    def _draw_gate_detail(self, axes: Any) -> None:
        """Magnify the closest opposing tips so the opening is observable."""
        from matplotlib.patches import Circle

        pairs = [
            (self.scene.branches[first], self.scene.branches[second])
            for first, second in self.scene.passage_pairs
        ]
        gate_index = min(
            range(len(pairs)),
            key=lambda index: abs(
                pairs[index][0].rest_nodes[-1, 0] - self.scene.end_effector[0]
            ),
        )
        lower, upper = pairs[gate_index]
        center = 0.5 * (lower.rest_nodes[-1] + upper.rest_nodes[-1])
        detail = axes.inset_axes([0.04, 0.69, 0.29, 0.27])
        detail.set_xlim(center[0] - 0.09, center[0] + 0.09)
        detail.set_ylim(center[1] - 0.085, center[1] + 0.085)
        detail.set_aspect("equal")
        for branch in (lower, upper):
            detail.plot(
                *branch.rest_nodes.T, "--", color="#9ca3af", linewidth=1.3
            )
            detail.plot(
                *branch.current_nodes.T,
                "-o",
                color="#dc2626" if branch.in_contact else "#2f855a",
                linewidth=2.0,
                markersize=3,
            )
        detail.add_patch(Circle(self.scene.end_effector, self.contact_radius, color="#2563eb"))
        gap = float(np.linalg.norm(upper.current_nodes[-1] - lower.current_nodes[-1]))
        detail.set_title(f"Gate {gate_index + 1} detail | gap={gap:.3f}", fontsize=7.3)
        detail.tick_params(labelsize=6)
        detail.grid(alpha=0.15)

    def _render_rgb_array(self) -> np.ndarray:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(6, 6), dpi=100, layout="tight")
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        self._draw_scene(axes)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()

    def _render_human(self) -> None:
        import matplotlib.pyplot as plt

        if self._figure is None:
            self._figure, self._axes = plt.subplots(figsize=(6, 6), layout="tight")
            plt.show(block=False)
        self._axes.clear()
        self._draw_scene(self._axes)
        self._figure.canvas.draw_idle()
        plt.pause(1.0 / float(self.metadata["render_fps"]))

    def close(self) -> None:
        if self._figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self._figure)
            self._figure = None
            self._axes = None
