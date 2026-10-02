"""支持可选束搜索的通用滚动时域规划器。"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

TransitionModel = Callable[[Any, int], tuple[Any, float, bool]]


@dataclass(frozen=True)
class PlanResult:
    action: int
    predicted_return: float
    model_evaluations: int


class RecedingHorizonPlanner:
    def __init__(self, action_count: int = 8, depth: int = 3, beam_width: int | None = None):
        if depth < 1:
            raise ValueError("规划深度必须为正整数")
        self.action_count = action_count
        self.depth = depth
        self.beam_width = beam_width

    def plan(self, state: Any, transition: TransitionModel) -> PlanResult:
        beam: list[tuple[float, Any, int | None, bool]] = [(0.0, state, None, False)]
        evaluations = 0
        for _ in range(self.depth):
            candidates: list[tuple[float, Any, int | None, bool]] = []
            for score, current_state, first_action, done in beam:
                if done:
                    candidates.append((score, current_state, first_action, True))
                    continue
                for action in range(self.action_count):
                    next_state, reward, terminated = transition(current_state, action)
                    evaluations += 1
                    root_action = action if first_action is None else first_action
                    candidates.append((score + reward, next_state, root_action, terminated))
            candidates.sort(key=lambda item: item[0], reverse=True)
            beam = candidates[: self.beam_width] if self.beam_width else candidates
        best_score, _, best_action, _ = beam[0]
        if best_action is None:
            raise RuntimeError("规划器没有评估任何动作")
        return PlanResult(best_action, best_score, evaluations)
