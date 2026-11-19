"""Fresh DQN with explicitly recorded, invertible coordinate preprocessing."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent, ObservationEncoder
from me5418_harvesting.envs.harvesting_env import HarvestingEnv


def transform_encoded(values, relative, branches=14, nodes=5):
    """Transform original normalized observations; leave other features intact.

    Keep global robot position, target-relative vector, stiffness, masks and
    budget. Relative node coordinates plus global robot position retain the
    information needed to recover absolute coordinates on valid branches.
    """
    result = np.asarray(values, dtype=np.float32).copy()
    if not relative:
        return result
    batched = result.reshape(-1, result.shape[-1])
    width = 4 * nodes + 4
    blocks = batched[:,4:-1].reshape(-1,branches,width)
    coords = blocks[:,:,:4*nodes].reshape(-1,branches,2*nodes,2)
    coords -= batched[:,None,None,:2]
    blocks *= blocks[:,:,-1:] != 0
    return result


class CoordinateEncoder:
    def __init__(self, env, relative):
        self.base = ObservationEncoder(env)
        self.relative = relative

    def __call__(self, observation):
        return transform_encoded(self.base(observation), self.relative,
                                 self.base.branches, self.base.nodes)


class CoordinateAgent(DQNAgent):
    FORMAT = 'fresh-coordinate-dqn-v1'

    def __init__(self, env, config, seed, relative=False):
        super().__init__(env, config, seed)
        self.relative = bool(relative)
        self.encoder = CoordinateEncoder(env, self.relative)

    def save(self, path, environment_config, metadata):
        path = Path(path)
        temporary = path.with_suffix('.tmp')
        torch.save(dict(format_version=self.FORMAT, base_encoder_version=1,
            relative=self.relative, environment_config=environment_config,
            dqn_config=asdict(self.learner.config), online=self.learner.online.state_dict(),
            metadata=metadata), temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['format_version'] != cls.FORMAT or payload['base_encoder_version'] != 1:
            raise ValueError('Use the matching coordinate-DQN checkpoint format.')
        env = HarvestingEnv(payload['environment_config'])
        # Loading a checkpoint for evaluation must not advance training Torch RNG.
        with torch.random.fork_rng(devices=[]):
            agent = cls(env, DQNConfig(**payload['dqn_config']), 0, payload['relative'])
        agent.learner.online.load_state_dict(payload['online'])
        agent.learner.target.load_state_dict(payload['online'])
        return agent, env, payload['metadata']
