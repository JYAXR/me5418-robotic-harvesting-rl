"""DQN configuration and network factory."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DQNConfig:
    hidden_sizes: tuple[int, int] = (256, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 200_000
    target_update_interval: int = 1_000


def build_q_network(input_dim: int, action_count: int, config: DQNConfig):
    """Create the two-hidden-layer MLP described in the proposal."""
    try:
        from torch import nn
    except ImportError as exc:
        raise RuntimeError('Install training dependencies with: pip install -e ".[train]"') from exc

    return nn.Sequential(
        nn.Linear(input_dim, config.hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(config.hidden_sizes[0], config.hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(config.hidden_sizes[1], action_count),
    )
