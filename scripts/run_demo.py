"""Run and visualize one or more harvesting-environment episodes."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from me5418_harvesting import HarvestingEnv
from me5418_harvesting.planners.collision_free import (
    NoCollisionFreeRoute,
    collision_free_actions,
)


def action_toward(env: HarvestingEnv, destination: np.ndarray) -> int:
    """Select a cardinal action toward a point, using small steps nearby."""
    if env.scene is None:
        raise RuntimeError("The environment must be reset before selecting an action.")
    delta = destination - env.scene.end_effector
    if abs(float(delta[0])) >= abs(float(delta[1])):
        direction = 2 if delta[0] >= 0 else 3
        remaining = abs(float(delta[0]))
    else:
        direction = 0 if delta[1] >= 0 else 1
        remaining = abs(float(delta[1]))
    return direction + 4 if remaining > env.large_step else direction


def greedy_action(env: HarvestingEnv) -> int:
    """Approach the target directly without adapting the step size to contact."""
    if env.scene is None:
        raise RuntimeError("The environment must be reset before selecting an action.")
    return action_toward(env, env.scene.target)


def push_action(env: HarvestingEnv) -> int:
    """Find each observed tip gap, align with it, and gently open it.

    This geometric demonstration controller uses only the current/resting
    nodes exposed in the observation. It is a handwritten policy, not RL.
    """
    if env.scene is None:
        raise RuntimeError("The environment must be reset before selecting an action.")
    position = env.scene.end_effector
    pairs = [
        (env.scene.branches[first], env.scene.branches[second])
        for first, second in env.scene.passage_pairs
    ]
    for lower, upper in pairs:
        center = 0.5 * (lower.rest_nodes[-1] + upper.rest_nodes[-1])
        if position[0] > center[0] + env.contact_radius + 0.025:
            continue
        # Align before reaching contact; crossing an off-center gap would
        # require unnecessary deformation of one side of the passage.
        if (
            position[0] < center[0] - 2.0 * env.contact_radius
            and abs(float(position[1] - center[1])) > env.small_step / 2.0
        ):
            return 0 if position[1] < center[1] else 1
        near_branch = any(
            branch.minimum_distance_to(position)
            <= env.contact_radius + 1.5 * env.small_step
            for branch in (lower, upper)
        )
        return 2 if near_branch else 6
    near_branch = any(
        branch.minimum_distance_to(env.scene.end_effector)
        <= env.contact_radius + 1.5 * env.small_step
        for branch in env.scene.branches
    )
    direct_action = action_toward(env, env.scene.target)
    return direct_action % 4 if near_branch else direct_action


class HybridDemoController:
    """Scripted gap opening interleaved with real collision-free A* detours.

    Select narrow soft passages using observed stiffness and resting tip gap.
    A* reaches each approach point while treating every branch as an obstacle;
    small rightward actions then execute contact through the actual solver.
    Closed/stiff passages are never removed from the planning geometry.
    This is an environment demonstration, not a learned or optimal policy.
    """

    def __init__(self, env: HarvestingEnv):
        self.pairs = [
            (first, second) for first, second in env.scene.passage_pairs
            if max(env.scene.branches[first].stiffness, env.scene.branches[second].stiffness) < 10
            and np.linalg.norm(
                env.scene.branches[first].rest_nodes[-1] - env.scene.branches[second].rest_nodes[-1]
            ) < 2 * env.contact_radius
        ]
        self.stage = 0
        self.phase = "approach"
        self.pending: deque[int] = deque()

    def next_action(self, env: HarvestingEnv) -> int:
        if self.stage >= len(self.pairs):
            if not self.pending:
                self.pending.extend(collision_free_actions(env, include_current_shape=True))
            if not self.pending:
                raise RuntimeError("Hybrid demo reached its goal without episode termination")
            return self.pending.popleft()

        first, second = self.pairs[self.stage]
        pair = [env.scene.branches[first], env.scene.branches[second]]
        center = 0.5 * (pair[0].rest_nodes[-1] + pair[1].rest_nodes[-1])
        approach = center - np.array([2 * env.contact_radius + env.small_step, 0])
        if self.phase == "approach":
            if np.linalg.norm(env.scene.end_effector - approach) <= env.small_step / 3:
                self.phase = "cross"
                self.pending.clear()
            else:
                if not self.pending:
                    self.pending.extend(collision_free_actions(
                        env, destination=approach, goal_radius=env.small_step / 3,
                        include_current_shape=True,
                    ))
                if not self.pending:
                    raise RuntimeError("Hybrid approach planner returned no actions")
                return self.pending.popleft()

        tip_x = max(float(branch.current_nodes[-1, 0]) for branch in pair)
        clear_of_pair = all(
            branch.minimum_distance_to(env.scene.end_effector) >= env.contact_radius + 0.01
            for branch in pair
        )
        if env.scene.end_effector[0] > tip_x + env.contact_radius + env.small_step and clear_of_pair:
            self.stage += 1
            self.phase = "approach"
            self.pending.clear()
            return self.next_action(env)
        return 2  # Small contact action; no obstacle or safety rule is disabled.


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
    )


def snapshot(env: HarvestingEnv) -> dict[str, object]:
    return {
        "step": env.step_count,
        "position": env.scene.end_effector.tolist(),
        "target": env.scene.target.tolist(),
        "peak_deformation": env.peak_deformation,
        "passage_pairs": env.scene.passage_pairs,
        "passage_kinds": env.scene.passage_kinds,
        "layout": env.scene.layout,
        "branches": [
            {
                "rest": branch.rest_nodes.tolist(),
                "current": branch.current_nodes.tolist(),
                "contact": branch.in_contact,
                "stiffness": branch.stiffness,
                "safe_limit": branch.safe_limit,
            }
            for branch in env.scene.branches
        ],
    }


def run_episode(
    policy: str,
    seed: int,
    output: Path | None,
    config: dict | None = None,
) -> dict[str, object]:
    env = HarvestingEnv(config, render_mode="rgb_array" if output else None)
    env.demo_policy = policy
    _, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    frames = [env.render()] if output else []
    trace = [snapshot(env)] if output else []
    episode_return = 0.0
    initial_clearance = min(
        branch.minimum_distance_to(env.scene.end_effector) for branch in env.scene.branches
    )
    final_info: dict[str, object] = {
        "success": False, "damage": False,
        "distance_to_target": float(np.linalg.norm(env.scene.target - env.scene.end_effector)),
        "travel_distance": 0.0, "max_normalized_deformation": 0.0,
        "peak_normalized_deformation": 0.0, "contact_steps": 0,
        "minimum_substep_clearance": initial_clearance, "minimum_clearance": initial_clearance,
        "collision_blocked": False, "commanded_motion": np.zeros(2), "executed_motion": np.zeros(2),
    }
    planning_failure = None
    detour_actions = []
    if policy == "detour":
        try:
            detour_actions = collision_free_actions(env)
        except NoCollisionFreeRoute as exc:
            planning_failure = str(exc)
    hybrid = HybridDemoController(env) if policy == "hybrid" else None
    minimum_clearance = initial_clearance
    maximum_strain = 0.0
    contact_snapshot_saved = False

    for _ in range(0 if planning_failure else env.max_steps):
        if policy == "random":
            action = env.action_space.sample()
        elif policy == "detour":
            if env.step_count >= len(detour_actions):
                raise RuntimeError("The planned route ended before reaching the target")
            action = detour_actions[env.step_count]
        elif policy == "push":
            action = push_action(env)
        elif policy == "hybrid":
            try:
                action = hybrid.next_action(env)
            except NoCollisionFreeRoute as exc:
                planning_failure = str(exc)
                break
        else:
            action = greedy_action(env)
        _, reward, terminated, truncated, final_info = env.step(action)
        episode_return += reward
        minimum_clearance = min(minimum_clearance, final_info["minimum_substep_clearance"])
        maximum_strain = max(
            maximum_strain, max(branch.max_segment_strain for branch in env.scene.branches)
        )
        if output:
            frames.append(env.render())
            trace.append(snapshot(env))
            if (
                policy in {"push", "hybrid"}
                and not contact_snapshot_saved
                and abs(float(env.scene.end_effector[0] - env.scene.branches[0].rest_nodes[-1, 0]))
                <= 0.05
                and np.linalg.norm(
                    env.scene.branches[1].current_nodes[-1]
                    - env.scene.branches[0].current_nodes[-1]
                ) >= 2.0 * env.contact_radius - 0.002
                and any(branch.in_contact for branch in env.scene.branches)
            ):
                output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frames[-1]).save(output.with_name(output.stem + "_contact.png"))
                contact_snapshot_saved = True
        if terminated or truncated:
            break

    if output is not None:
        save_gif(frames, output, int(env.metadata["render_fps"]))
        Image.fromarray(frames[0]).save(output.with_name(output.stem + "_start.png"))
        Image.fromarray(frames[-1]).save(output.with_name(output.stem + "_final.png"))
    env.close()
    result = {
        "seed": seed,
        "policy": policy,
        "steps": env.step_count,
        "return": episode_return,
        "minimum_clearance_over_episode": minimum_clearance,
        "maximum_segment_strain": maximum_strain,
        "planning_failure": planning_failure,
        **final_info,
    }
    result = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }
    if output:
        output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        output.with_name(output.stem + "_trace.json").write_text(
            json.dumps({"robot_radius": env.contact_radius, "frames": trace}, indent=2),
            encoding="utf-8",
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=("random", "greedy", "push", "detour", "hybrid"),
        default="push",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument("--no-gif", action="store_true", help="Run diagnostics without rendering")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/demo"),
        help="Directory for animated GIF files, snapshots, and episode metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for episode in range(args.episodes):
        seed = args.seed + episode
        output = None if args.no_gif else args.output_dir / f"{args.policy}_seed_{seed}.gif"
        result = run_episode(args.policy, seed, output, config)
        print(result)
        if output:
            print(f"Saved visualization to {output.resolve()}")


if __name__ == "__main__":
    main()
