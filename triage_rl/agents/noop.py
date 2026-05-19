"""No-op reference policy: keep current SOC, no treatment."""
from __future__ import annotations

import numpy as np

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.agents.base import Agent


class NoOpAgent(Agent):
    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        soc = int(obs[State.SOC_IDX])
        # (antib=0, vent=0, vaso=0, soc=current). Index packing: 16*a + 8*v + 4*p + soc.
        return soc
