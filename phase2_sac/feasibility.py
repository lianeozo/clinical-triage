"""Static feasibility mask for the 32-action triage POMDP.

action_idx ∈ [0, 32) decodes to (antibiotic, ventilation, vasopressors, action_soc)
via the packing 16·a + 8·v + 4·p + s. Feasibility is determined by
SOC_TREATMENT_FEASIBILITY[action_soc], which depends only on the action's chosen
SOC (not on the patient's current state).

FEASIBILITY_MASK[i] = True iff action_idx i is feasible at its chosen SOC.
"""
from __future__ import annotations

import numpy as np

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.MDP import SOC_TREATMENT_FEASIBILITY


def _build_mask() -> np.ndarray:
    mask = np.zeros(Action.NUM_ACTIONS_TOTAL, dtype=np.bool_)
    for idx in range(Action.NUM_ACTIONS_TOTAL):
        a = Action(action_idx=idx)
        rules = SOC_TREATMENT_FEASIBILITY[a.soc]
        mask[idx] = (a.antibiotic in rules["antibiotic"]
                     and a.ventilation in rules["ventilation"]
                     and a.vasopressors in rules["vasopressors"])
    return mask


FEASIBILITY_MASK: np.ndarray = _build_mask()
