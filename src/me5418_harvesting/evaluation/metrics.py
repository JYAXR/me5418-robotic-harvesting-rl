"""Episode-level metrics shared by every method."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    success: bool
    damage: bool
    action_count: int
    travel_distance: float
    max_normalized_deformation: float
    episode_return: float
    model_evaluations: int = 0
    decision_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, bool | int | float]:
        return asdict(self)


def summarize(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    if not episodes:
        raise ValueError("at least one episode is required")
    numeric = {
        key: np.asarray([float(getattr(item, key)) for item in episodes])
        for key in episodes[0].to_dict()
    }
    summary: dict[str, float] = {}
    for key, values in numeric.items():
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary

