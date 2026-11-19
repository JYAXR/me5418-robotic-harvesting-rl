"""Check a reference route through actual dynamics without changing the input state."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from me5418_harvesting.envs.harvesting_env import HarvestingEnv


def _copy_environment(env: HarvestingEnv) -> HarvestingEnv:
    """Copy only simulation state; do not reset scenes or copy rendering handles."""
    from me5418_harvesting.envs.harvesting_env import HarvestingEnv

    config = {
        name: getattr(env, name)
        for name in (
            "max_branches", "nodes_per_branch", "max_steps", "small_step", "large_step",
            "target_radius", "contact_radius", "relaxation_rate",
        )
    }
    config.update({
        "workspace_low": env.workspace_low.copy(),
        "workspace_high": env.workspace_high.copy(),
        "execution_noise_fraction": env.noise_fraction,
        "reward": deepcopy(env.reward_weights),
    })
    probe = HarvestingEnv(config)
    probe.scene = deepcopy(env.scene)
    probe.np_random = deepcopy(env.np_random)
    probe.step_count = env.step_count
    probe.total_travel = env.total_travel
    probe.peak_deformation = env.peak_deformation
    probe.contact_steps = env.contact_steps
    probe.trajectory = [position.copy() for position in env.trajectory]
    return probe


def verify_reference_route(env: HarvestingEnv) -> dict:
    """Plan around branches and execute the plan in an isolated environment copy.

    A successful rollout witnesses a feasible route for this state and RNG.
    Failure is not a proof of infeasibility: the planner uses a discretized
    collision-free grid, does not deliberately bend branches, and may be affected
    by execution noise. No planning or rollout state is written back to env.
    """
    from me5418_harvesting.planners.collision_free import (
        NoCollisionFreeRoute,
        collision_free_actions,
    )

    if env.scene is None:
        raise RuntimeError("Reset the environment before verifying a reference route.")
    probe = _copy_environment(env)
    initial_steps = probe.step_count
    initial_travel = probe.total_travel
    initial_contacts = probe.contact_steps
    damage = probe.peak_deformation > 1.0
    at_goal = bool(np.linalg.norm(probe.scene.target - probe.scene.end_effector)
                   <= probe.target_radius)
    result = {
        "reference": "collision_free_astar",
        "success": at_goal and not damage,
        "damage": damage,
        "budget_exhausted": probe.step_count >= probe.max_steps,
        "planning_failure": False,
        "reason": "not_checked",
        "steps": 0,
        "planned_actions": 0,
        "initial_step_count": initial_steps,
        "remaining_budget": max(0, probe.max_steps - initial_steps),
        "travel_distance": 0.0,
        "return": 0.0,
        "contact_steps": 0,
        "peak_normalized_deformation": float(probe.peak_deformation),
    }
    try:
        if result["success"] or result["damage"] or result["budget_exhausted"]:
            result["reason"] = "input_state_already_terminal"
            return result
        try:
            actions = collision_free_actions(probe, include_current_shape=True)
        except NoCollisionFreeRoute as exc:
            result.update(planning_failure=True, reason="planner_found_no_route", detail=str(exc))
            return result
        result["planned_actions"] = len(actions)
        result["reason"] = "route_exhausted_before_success"
        for action in actions:
            _, reward, terminated, truncated, info = probe.step(action)
            result.update({
                "success": bool(info["success"]), "damage": bool(info["damage"]),
                "budget_exhausted": bool(info["budget_exhausted"]),
                "steps": probe.step_count - initial_steps,
                "travel_distance": float(probe.total_travel - initial_travel),
                "contact_steps": probe.contact_steps - initial_contacts,
                "peak_normalized_deformation": float(probe.peak_deformation),
            })
            result["return"] += float(reward)
            if terminated or truncated:
                result["reason"] = (
                    "success" if info["success"] else "damage" if info["damage"]
                    else "budget_exhausted" if info["budget_exhausted"] else "external_truncation"
                )
                break
        return result
    finally:
        probe.close()
