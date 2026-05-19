"""Agent abstract base class.

Agents always pick from [0, NUM_ACTIONS_TOTAL). They do NOT know about
feasibility — the Env clamps infeasible components (F4a).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Agent(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """Return an action index in [0, NUM_ACTIONS_TOTAL)."""

    def update(self, batch: dict) -> dict[str, float]:
        """Train on a batch. Default: no-op. Returns metrics dict for logging."""
        return {}

    def save(self, path: Path) -> None:
        """Default no-op. Trainable agents override."""

    def load(self, path: Path) -> None:
        """Default no-op. Trainable agents override."""
