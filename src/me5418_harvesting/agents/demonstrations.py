"""Optional replay initialization from successful, physically executed reference routes."""

from collections.abc import Callable
from copy import deepcopy

from me5418_harvesting.agents.dqn_agent import ObservationEncoder
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.envs.harvesting_env import HarvestingEnv
from me5418_harvesting.planners.collision_free import NoCollisionFreeRoute, collision_free_actions


def seed_replay_from_reference(
    environment_config: dict, encoder: ObservationEncoder, replay: ReplayBuffer,
    episodes: int, seed: int,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Try each training-only seed once; keep failures in the collection record.

    This is plain off-policy replay initialization, not DQfD or behavior cloning.
    It neither updates weights nor consumes the online agent's exploration RNG.
    Only successful complete trajectories enter replay. Rewards and terminal
    flags come from env.step, including the final transition before any reset.
    """
    if type(episodes) is not int or episodes < 0 or type(seed) is not int or seed < 0:
        raise ValueError("Demonstration episodes and seed must be nonnegative integers.")
    env = HarvestingEnv(deepcopy(environment_config))
    records = []
    initial_size = len(replay)

    def snapshot():
        return {
            "method": "collision_free_astar_replay_initialization",
            "requested_episodes": episodes, "seed": seed,
            "successful_episodes": sum(
                row["success"] and not row["damage"] for row in records
            ),
            "collection_environment_steps": sum(row["steps"] for row in records),
            "accepted_transitions": len(replay) - initial_size,
            "optimizer_updates": 0, "records": records,
            "accounting_note": (
                "Collection steps exclude planner search and reset feasibility probes. "
                "Demonstrations use privileged geometry and are additional training data."
            ),
        }

    try:
        # Do not silently overwrite either existing data or earlier demonstrations.
        if episodes * env.max_steps > replay.capacity - initial_size:
            raise ValueError("Replay capacity must hold the maximum demonstration step budget.")
        for index in range(episodes):
            record = {
                "seed": seed + index, "success": False, "damage": False,
                "budget_exhausted": False, "steps": 0, "return": 0.0,
                "accepted_transitions": 0, "reason": "collecting",
            }
            records.append(record)
            if progress is not None:
                progress(snapshot())
            pending = []
            try:
                observation, reset_info = env.reset(seed=seed + index)
            except ValueError as exc:
                record.update(reason="scene_generation_failed", detail=str(exc))
                continue
            record["scene_generation_attempts"] = reset_info.get("scene_generation_attempts", 1)
            record["reference_route_attempts"] = reset_info.get("reference_route_attempts", 0)
            try:
                actions = collision_free_actions(env, include_current_shape=True)
            except NoCollisionFreeRoute as exc:
                record.update(reason="planner_found_no_route", detail=str(exc))
                continue
            record["planned_actions"] = len(actions)
            record["reason"] = "executing_route"
            for action in actions:
                next_observation, reward, terminated, truncated, info = env.step(action)
                pending.append((
                    encoder(observation), action, reward, encoder(next_observation),
                    terminated, truncated,
                ))
                record["steps"] += 1
                record["return"] += float(reward)
                record.update({
                    "success": bool(info["success"]), "damage": bool(info["damage"]),
                    "budget_exhausted": bool(info["budget_exhausted"]),
                    "travel_distance": float(info["travel_distance"]),
                    "peak_normalized_deformation": float(info["peak_normalized_deformation"]),
                })
                observation = next_observation
                if terminated or truncated:
                    record["reason"] = (
                        "success" if info["success"] else "damage" if info["damage"]
                        else "budget_exhausted" if info["budget_exhausted"] else "external_truncation"
                    )
                    break
            else:
                record["reason"] = "route_exhausted_before_success"
            if record["success"] and not record["damage"]:
                for transition in pending:
                    replay.add(*transition)
                record["accepted_transitions"] = len(pending)
            if progress is not None:
                progress(snapshot())
    except BaseException as exc:
        if records and records[-1]["reason"] in ("collecting", "executing_route"):
            records[-1].update(reason="collection_interrupted", error_type=type(exc).__name__)
        raise
    finally:
        env.close()
        if progress is not None:
            progress(snapshot())
    return snapshot()
