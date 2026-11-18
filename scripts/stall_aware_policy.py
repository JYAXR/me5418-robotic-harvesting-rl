"""A greedy Q-policy ablation that temporarily avoids repeated no-motion actions."""

from collections import deque

import numpy as np
import torch


class StallAwarePolicy:
    """Use only recent observations, attempted actions, and the checkpoint Q scores."""

    DISPLACEMENT_TOLERANCE = 1e-6
    REPEAT_LIMIT = 2
    POSITION_CELL = 0.0025
    MEMORY_STEPS = 24

    def __init__(self, agent):
        self.agent = agent
        self.reset()

    def reset(self) -> None:
        """Clear all action history before starting an independent episode."""
        self.last_position = None
        self.last_action = None
        self.stalled_action = None
        self.stall_count = 0
        self.step = 0
        self.action_overrides = 0
        self.suppressed = deque(maxlen=self.MEMORY_STEPS)

    def act(self, observation: np.ndarray) -> int:
        position = observation[:2]
        pose = tuple(np.rint(position / self.POSITION_CELL).astype(int))
        self.step += 1
        while self.suppressed and self.step - self.suppressed[0][0] >= self.MEMORY_STEPS:
            self.suppressed.popleft()
        if (self.last_position is not None
                and np.linalg.norm(position - self.last_position) < self.DISPLACEMENT_TOLERANCE):
            self.stall_count = self.stall_count + 1 if self.last_action == self.stalled_action else 1
            self.stalled_action = self.last_action
            if self.stall_count == self.REPEAT_LIMIT:
                self.suppressed.append((self.step, pose, self.stalled_action))
        else:
            self.stalled_action = None
            self.stall_count = 0

        suppressed = {action for _, saved_pose, action in self.suppressed if saved_pose == pose}
        if not suppressed or len(suppressed) == self.agent.action_count:
            action = self.agent.act(observation)
        else:
            encoded = self.agent.encoder(observation)
            self.agent.learner.online.eval()
            with torch.no_grad():
                values = self.agent.learner.online(torch.from_numpy(encoded).unsqueeze(0))[0]
            if not torch.isfinite(values).all():
                raise FloatingPointError("The policy produced non-finite action values.")
            greedy = int(values.argmax().item())
            ranked = values.clone()
            ranked[list(suppressed)] = -torch.inf
            action = int(ranked.argmax().item())
            self.action_overrides += int(action != greedy)
        self.last_position = position.copy()
        self.last_action = action
        return action
