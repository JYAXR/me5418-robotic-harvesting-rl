"""DQN configuration and network construction."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class DQNConfig:
    hidden_sizes: tuple[int, int] = (256, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 200_000
    target_update_interval: int = 1_000
    gradient_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        if len(self.hidden_sizes) != 2 or any(
            not isinstance(size, int) or size <= 0 for size in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain two positive integers.")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between zero and one.")
        if any(
            not isinstance(value, int) or value <= 0
            for value in (self.batch_size, self.replay_capacity, self.target_update_interval)
        ):
            raise ValueError("Batch size, replay capacity, and target interval must be positive integers.")
        if not isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be finite and positive.")


def build_q_network(input_dim: int, action_count: int, config: DQNConfig):
    """Create the two-hidden-layer MLP defined in the proposal."""
    if input_dim <= 0 or action_count <= 0:
        raise ValueError("Input dimension and action count must be positive.")
    try:
        from torch import nn
    except ImportError as exc:
        raise RuntimeError('Install the network dependencies: pip install -e ".[network]"') from exc

    return nn.Sequential(
        nn.Linear(input_dim, config.hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(config.hidden_sizes[0], config.hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(config.hidden_sizes[1], action_count),
    )
