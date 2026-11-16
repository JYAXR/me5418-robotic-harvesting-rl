"""Bounded process-local cache of successfully certified seeded reset states."""
import json
from collections import OrderedDict
from copy import deepcopy


class CertifiedSceneCache:
    """Reuse accepted scenes without changing reset information or random state.

    Entries never cross processes or software versions. Call clear() after any
    runtime monkey-patching of simulation/planner code. Each hit returns fresh
    copies; mutable episode state is never shared between environments. Use one
    cache per worker; this class is not intended for concurrent thread access.
    """

    def __init__(self, capacity=128):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Cache capacity must be a positive integer.")
        self.capacity = capacity
        self._entries = OrderedDict()

    def clear(self):
        self._entries.clear()

    def __len__(self):
        return len(self._entries)

    def key(self, env, seed, options):
        from me5418_harvesting.envs.harvesting_env import HarvestingEnv

        if type(env) is not HarvestingEnv:
            return None
        if type(seed) is not int or seed < 0 or options is not None or not env.verify_reference_route:
            return None
        names = ("max_branches", "nodes_per_branch", "max_steps", "small_step", "large_step",
                 "target_radius", "contact_radius", "relaxation_rate", "noise_fraction",
                 "reference_max_attempts", "verify_reference_route", "observation_size")
        config = {name: getattr(env, name) for name in names}
        config.update(workspace_low=env.workspace_low.tolist(), workspace_high=env.workspace_high.tolist(),
                      reward=env.reward_weights, scene=env.scene_parameters,
                      directions=env._DIRECTIONS.tolist(), action_count=env.action_space.n)
        return (type(env).__module__, type(env).__qualname__, seed,
                json.dumps(config, sort_keys=True, allow_nan=False, default=lambda x: x.tolist()))

    def restore(self, key, env):
        if key is None or key not in self._entries:
            return None
        self._entries.move_to_end(key)
        scene, rng, rng_seed, info = deepcopy(self._entries[key])
        env.scene, env.np_random = scene, rng
        env._np_random_seed = rng_seed
        env._initialize_episode()
        return info

    def store(self, key, env, info):
        if key is None or not info.get("reference_route", {}).get("success", False):
            return
        self._entries[key] = deepcopy((env.scene, env.np_random, env.np_random_seed, info))
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
