"""Agent abstract base class.

Agents always pick from [0, NUM_ACTIONS_TOTAL). They do NOT know about
feasibility — the Env clamps infeasible components (F4a).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class OnPolicyAgent(Protocol):
    """Duck-typed interface OnPolicyTrainer requires of its agent.

    Concrete agents like PPOAgent and (future) on-policy SAC variants satisfy this
    by providing these methods plus a `cfg` attribute exposing `gamma` and `gae_lambda`.
    """

    cfg: object  # must expose .gamma and .gae_lambda

    def act(self, obs, eval_mode: bool = False) -> int: ...
    def act_with_logp_value(self, obs) -> tuple[int, float, float]: ...
    def value_only(self, obs) -> float: ...
    def update(self, batch: dict) -> dict[str, float]: ...
    def save(self, path) -> None: ...
    def load(self, path) -> None: ...
