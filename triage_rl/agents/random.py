"""Uniform-random reference policy."""
from __future__ import annotations

import numpy as np

from sepsisSimDiabetes.Action import Action
from triage_rl.agents.base import Agent


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        return int(self._rng.integers(0, Action.NUM_ACTIONS_TOTAL))
