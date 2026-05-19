"""No-op reference policy: keep current SOC, no treatment."""
from __future__ import annotations

import numpy as np

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.agents.base import Agent


class NoOpAgent(Agent):
    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        soc = int(obs[State.SOC_IDX])
        # Stay at current SOC with no treatment; let Action compute the packed idx.
        return Action(selected_actions={Action.SOC_STRING: soc}).get_action_idx()
