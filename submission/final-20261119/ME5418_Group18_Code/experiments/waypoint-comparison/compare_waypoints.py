"""Compare matched archived planner rollouts with local waypoint scoring."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "research_results/paired_planner_evaluation"
sys.path[:0] = [
    str(REPO / "src"),
    str(REPO / "scripts"),
    str(REPO / "research_results/20261118_generalization"),
]

import torch
from geometric_coordinate_policy import BRANCHES, BRANCH_WIDTH, NODES
from guided_coordinate_policy import CompactGuidedAgent
from me5418_harvesting.envs.branch_physics import Branch
from me5418_harvesting.planners.receding_horizon import RecedingHorizonPlanner


MODEL = REPO / "models/compact_guided_generalization.pt"
MODEL_SHA256 = "75d7d7ca728b1c85c8ba6fa226076f85f4f72bdce3cfe4c6c0177946e5fe51e6"
SETTINGS = {
    "1": (1, None),
    "3b32": (3, 32),
    "5": (5, 32),
    "7": (7, 32),
    "3": (3, None),
}


def waypoint_offset(flat, encoder):
    """Return the pre-clipping offset and gate flag from the DQN waypoint rule."""
    robot = flat[:2] * encoder.span
    nodes = flat[4:-1].reshape(BRANCHES, BRANCH_WIDTH)
    current = nodes[:12, :2 * NODES].reshape(3, 2, 2, NODES, 2) * encoder.span
    rest = nodes[:12, 2 * NODES:4 * NODES].reshape(3, 2, 2, NODES, 2) * encoder.span
    extent = np.concatenate((current[..., 0], rest[..., 0]), axis=-1)
    exit_x = extent.max(axis=(1, 2, 3)) + 1.1 * encoder.radius
    pending = robot[0] < exit_x
    if not pending.any():
        return robot, (robot + flat[2:4] * encoder.span) - robot, False
    region = int(pending.argmax())
    tips = rest[region, :, :, -1, 1]
    lane = int(np.abs(tips[:, 1] - tips[:, 0]).argmax())
    center = current[region, lane, :, -1, 1].mean()
    waypoint = np.asarray((exit_x[region], center), dtype=robot.dtype)
    return robot, waypoint - robot, True


def state_key(env):
    return (
        env.scene.end_effector.tobytes(),
        tuple(branch.current_nodes.tobytes() for branch in env.scene.branches),
        tuple(branch.in_contact for branch in env.scene.branches),
    )


class MemoizedBranch(Branch):
    """Reuse identical branch operations in candidate simulation only."""

    def __init__(self, branch):
        super().__init__(branch.rest_nodes, branch.current_nodes, branch.stiffness,
                         branch.safe_limit, branch.in_contact)
        self.source = deepcopy(branch)
        self.shape = self.current_nodes.shape
        self.rest_key = self.rest_nodes.tobytes()
        self.rest_bounds = (
            *map(float, self.rest_nodes.min(axis=0)), *map(float, self.rest_nodes.max(axis=0)),
        )
        self.rest_fixed = bool((np.linalg.norm(np.diff(self.rest_nodes, axis=0), axis=1) > 1e-8).all())
        self.contact = lru_cache(maxsize=2048)(self.compute_contact)
        self.distance = lru_cache(maxsize=2048)(self.compute_distance)
        self.strain = lru_cache(maxsize=512)(self.compute_strain)
        self.deformation = lru_cache(maxsize=512)(self.compute_deformation)

    def compute_contact(self, nodes, position, motion, radius, relaxation):
        self.source.current_nodes = np.frombuffer(nodes, np.float32).reshape(self.shape).copy()
        self.source.update_contact(np.frombuffer(position, np.float32),
                                   np.frombuffer(motion, np.float32), radius, relaxation)
        return self.source.current_nodes.tobytes(), self.source.in_contact

    def compute_distance(self, nodes, position):
        self.source.current_nodes = np.frombuffer(nodes, np.float32).reshape(self.shape)
        return self.source.minimum_distance_to(np.frombuffer(position, np.float32))

    def compute_strain(self, nodes):
        self.source.current_nodes = np.frombuffer(nodes, np.float32).reshape(self.shape)
        return self.source.max_segment_strain

    def compute_deformation(self, nodes):
        self.source.current_nodes = np.frombuffer(nodes, np.float32).reshape(self.shape)
        return self.source.normalized_deformation

    def update_contact(self, end_effector, motion, contact_radius, relaxation_rate):
        shape = self.current_nodes.tobytes()
        if self.rest_fixed and shape == self.rest_key:
            x, y = map(float, end_effector)
            left, bottom, right, top = self.rest_bounds
            margin = contact_radius + 1e-6
            if x < left - margin or x > right + margin or y < bottom - margin or y > top + margin:
                # A resting branch outside contact has neither spring motion nor length correction.
                self.in_contact = False
                return
        nodes, self.in_contact = self.contact(shape, end_effector.tobytes(), motion.tobytes(),
                                              contact_radius, relaxation_rate)
        self.current_nodes = np.frombuffer(nodes, np.float32).reshape(self.shape).copy()

    def minimum_distance_to(self, point):
        return self.distance(self.current_nodes.tobytes(), point.tobytes())

    @property
    def max_segment_strain(self):
        return self.strain(self.current_nodes.tobytes())

    @property
    def normalized_deformation(self):
        return self.deformation(self.current_nodes.tobytes())


def make_physics(env, memoize_branches=False):
    if env.noise_fraction != 0:
        raise ValueError("The cached prediction model requires zero execution noise.")
    simulator = deepcopy(env)
    if memoize_branches:
        assert env.scene.end_effector.dtype == np.float32
        assert all(branch.current_nodes.dtype == np.float32 for branch in env.scene.branches)
        simulator.scene.branches = [MemoizedBranch(branch) for branch in simulator.scene.branches]

    @lru_cache(maxsize=24000)
    def physical(state, action):
        xy, shapes, contacts = state
        simulator.scene.end_effector = np.frombuffer(xy, np.float32).copy()
        for branch, shape, contact in zip(simulator.scene.branches, shapes, contacts, strict=True):
            branch.current_nodes = np.frombuffer(shape, np.float32).reshape(branch.rest_nodes.shape).copy()
            branch.in_contact = contact
        simulator._initialize_episode()
        _, reward, ended, truncated, _ = simulator.step(action)
        return state_key(simulator), reward, ended or truncated

    return physical


def checked_waypoint(observation, encoder):
    flat = encoder.base(observation)
    robot, offset, has_gate = waypoint_offset(flat, encoder)
    features = encoder._features(flat[None, :])[0]
    np.testing.assert_allclose(
        np.clip(offset / encoder.large_step, -3, 3), features[:2], atol=1e-6, rtol=0,
    )
    candidate_offset = offset[None, :] - encoder.offsets
    progress = (np.abs(offset).sum() - np.abs(candidate_offset).sum(axis=1)) / encoder.large_step
    np.testing.assert_allclose(np.clip(progress, -3, 3), features[8:16], atol=1e-6, rtol=0)
    assert has_gate == bool(features[7])
    return tuple(map(float, robot + offset)) if has_gate else None


def guided_rollout(env, encoder, obs, depth, beam, weight, physical, progress=None):
    planner = RecedingHorizonPlanner(depth=depth, beam_width=beam)

    @lru_cache(maxsize=256)
    def decision(root, remaining, waypoint):
        @lru_cache(maxsize=None)
        def distance(xy):
            x, y = np.frombuffer(xy, np.float32)
            return abs(float(x) - waypoint[0]) + abs(float(y) - waypoint[1])

        def transition(state, action):
            key, budget = state
            result, reward, terminal = physical(key, action)
            if waypoint is not None:
                reward += weight * (distance(key[0]) - distance(result[0]))
            return (result, budget - 1), reward, terminal or budget == 1

        return planner.plan((root, remaining), transition)

    trace = []
    total_reward = 0.0
    for step in range(1, env.max_steps + 1):
        key = state_key(env)
        waypoint = checked_waypoint(obs, encoder) if weight else None
        remaining = min(depth, env.max_steps - step + 1)
        plan = decision(key, remaining, waypoint)
        obs, reward, done, trunc, info = env.step(plan.action)
        if env.noise_fraction == 0:
            predicted, raw_reward, terminal = physical(key, plan.action)
            assert predicted == state_key(env) and raw_reward == reward
            assert (terminal or step == env.max_steps) == (done or trunc)
        total_reward += reward
        trace.append({
            "step": step, "action": plan.action, "xy": env.scene.end_effector.tolist(),
            "waypoint": waypoint, "reward": reward, "travel": env.total_travel,
            "peak": env.peak_deformation, "contact_steps": env.contact_steps,
            "terminal": done or trunc, "plan_score": plan.predicted_return,
        })
        if done or trunc:
            break
        if progress is not None and step % 25 == 0:
            progress(step)

    return {
        "steps": step, "success": bool(info["success"]), "damage": bool(info["damage"]),
        "timeout": bool(info["budget_exhausted"] and not info["success"] and not info["damage"]),
        "travel": env.total_travel, "peak": env.peak_deformation, "reward": total_reward,
        "contacts": env.contact_steps, "trace": trace,
    }


def run_seed(seed, setting_names, weight, output, memoize_branches):
    torch.set_num_threads(1)
    agent, env, _ = CompactGuidedAgent.load(MODEL)
    initial_obs, initial_info = env.reset(seed=seed)
    assert env.noise_fraction == 0 and env.max_steps == 200
    initial_hash = hashlib.sha256(initial_obs.tobytes()).hexdigest()
    physical = make_physics(env, memoize_branches)
    for suffix in setting_names:
        target = output / f"{seed}-{suffix}.json"
        if target.exists():
            raise FileExistsError(target)
        gold = json.loads((ARCHIVE / "data/episodes" / target.name).read_text())
        assert initial_hash == gold["initial_observation_sha256"]
        assert initial_info["reference_route"] == gold["reference"]
        depth, beam = SETTINGS[suffix]
        row = guided_rollout(deepcopy(env), agent.encoder, initial_obs, depth, beam, weight, physical)
        row.update(seed=seed, method=gold["method"], depth=depth, beam=beam, waypoint_weight=weight,
                   initial_observation_sha256=initial_hash, model_sha256=MODEL_SHA256)
        if weight == 0:
            keys = ("action", "xy", "reward", "travel", "peak", "contact_steps", "terminal")
            for actual, expected in zip(row["trace"], gold["trace"], strict=True):
                assert all(actual[k] == expected[k] for k in keys)
            assert all(row[k] == gold[k] for k in ("success", "damage", "timeout"))
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, indent=2), encoding="utf-8")
        temporary.replace(target)
        print(f'{seed} {row["method"]}: success={row["success"]}, '
              f'damage={row["damage"]}, timeout={row["timeout"]}, steps={row["steps"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5420000, 5420010)),
                        choices=range(5420000, 5420010))
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--waypoint-weight", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--memoize-branches", action="store_true")
    parser.add_argument("--skip-completed-in", type=Path,
                        help="Skip episode files already produced by a separate run.")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.settings)) != len(args.settings):
        parser.error("Seeds and settings must be unique.")
    if not np.isfinite(args.waypoint_weight) or args.waypoint_weight < 0 or args.workers < 1:
        parser.error("Use a finite nonnegative waypoint weight and positive worker count.")
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == MODEL_SHA256
    args.out.mkdir(parents=True, exist_ok=True)
    pending = {}
    for seed in args.seeds:
        settings = []
        for suffix in args.settings:
            filename = f"{seed}-{suffix}.json"
            existing = args.skip_completed_in / filename if args.skip_completed_in else None
            if existing is None or not existing.exists():
                settings.append(suffix)
                continue
            row = json.loads(existing.read_text(encoding="utf-8"))
            assert row["seed"] == seed and row["waypoint_weight"] == args.waypoint_weight
            assert (row["depth"], row["beam"]) == SETTINGS[suffix]
            assert row["model_sha256"] == MODEL_SHA256
        if settings:
            pending[seed] = settings
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
        futures = [pool.submit(run_seed, s, settings, args.waypoint_weight, args.out,
                               args.memoize_branches)
                   for s, settings in pending.items()]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
