"""Rule-based SOC triage heuristic: escalate care level by number of abnormal vitals.

Obs layout: [hr, sysbp, percoxyg, glucose, antibiotic, vaso, vent, soc]
  Normal values: hr=1, sysbp=1, percoxyg=1, glucose=2; -1 means masked (unknown).

Rule:
  0 abnormal vitals → ASYNC        (action 0: no treatment, soc=0)
  1 abnormal vital  → AMBULATORY   (action 1: no treatment, soc=1)
  2+ abnormal vitals → ICU         (action 3: no treatment, soc=3)

No active treatment is prescribed — this is a pure site-of-care heuristic.
Actions are always feasible (no ventilation or vasopressors at ASYNC/AMBULATORY,
no treatment at ICU).
"""
from __future__ import annotations

import numpy as np

from triage_rl.agents.base import Agent
from sepsisSimDiabetes.State import State


# Mirror of State.get_num_abnormal() thresholds — derived from State to stay in sync.
_VITAL_NORMALS = [
    State.NUM_HR // 2,       # hr normal bin = 1
    State.NUM_SYSBP // 2,    # sysbp normal bin = 1
    State.NUM_OXYG - 1,      # percoxyg normal bin = 1
    State.NUM_GLUC // 2,     # glucose normal bin = 2
]


class SocHeuristicAgent(Agent):
    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        num_abnormal = sum(
            int(obs[i] != _VITAL_NORMALS[i] and obs[i] != -1)
            for i in range(4)
        )
        if num_abnormal >= 2:
            soc = State.ICU
        elif num_abnormal == 1:
            soc = State.AMBULATORY
        else:
            soc = State.ASYNC
        # action_idx = treat_idx * 4 + soc; treat_idx=0 means no treatment
        return int(soc)
