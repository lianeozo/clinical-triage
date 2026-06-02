"""Single source of truth for the 5 reward variants (reward0..reward4).

Pure functions over an 8-dim obs/state vector
[hr, sysbp, percoxyg, glucose, antibiotic, vaso, vent, soc] so the same logic
runs in the env (true state) and on offline-dataset rows (observations).
Reconstructed from git commits a8dadfb/ae7b9e2/63239a8/f791379/5f52ddb and
verified against the milestone descriptions.
"""
from __future__ import annotations

import numpy as np

_ICU = 3
_NUM_SOC = 4

# Per-variant params: terminal scale T, treatment costs (antib, vent, vaso),
# whether the severity-aware ICU-overuse term is on, and SOC resource costs by level.
REWARD_VARIANTS = {
    "reward0": dict(T=10000, treat=(10, 60, 40),  icu_overuse=False, soc_cost=None),
    "reward1": dict(T=1000,  treat=(10, 60, 40),  icu_overuse=False, soc_cost=None),
    "reward2": dict(T=1000,  treat=(60, 120, 80), icu_overuse=False, soc_cost=None),
    "reward3": dict(T=1000,  treat=(10, 60, 40),  icu_overuse=True,  soc_cost=None),
    "reward4": dict(T=1000,  treat=(10, 60, 40),  icu_overuse=False,
                    soc_cost=(0, 5, 20, 50)),
}


def num_abnormal(o) -> int:
    o = np.asarray(o)
    n = 0
    if o[0] != 1 and o[0] != -1: n += 1   # hr
    if o[1] != 1 and o[1] != -1: n += 1   # sysbp
    if o[2] != 1 and o[2] != -1: n += 1   # percoxyg
    if o[3] != 2 and o[3] != -1: n += 1   # glucose (normal == 2)
    return n


def on_treatment(o) -> bool:
    o = np.asarray(o)
    return not (o[4] == 0 and o[5] == 0 and o[6] == 0)


def _decode_action(action_idx: int):
    # antib*16 + vent*8 + vaso*4 + soc  (matches Action.get_action_idx)
    antib = action_idx // 16
    rest = action_idx % 16
    vent = rest // 8
    rest = rest % 8
    vaso = rest // 4
    soc = rest % 4
    return int(antib), int(vent), int(vaso), int(soc)


def compute_reward(prev_obs, action_idx: int, next_obs, version: str = "reward0") -> float:
    p = REWARD_VARIANTS[version]
    prev_obs = np.asarray(prev_obs)
    next_obs = np.asarray(next_obs)
    n_next = num_abnormal(next_obs)
    n_prev = num_abnormal(prev_obs)

    # Absorbing states (terminal returns early, like the original).
    if n_next >= 3:
        return float(-p["T"])
    if n_next == 0 and not on_treatment(next_obs):
        return float(p["T"])

    reward = 0.0
    # Vitals trajectory feedback.
    reward += 100.0 * (n_prev - n_next)

    # Escalation / de-escalation cost.
    next_soc = int(next_obs[7])
    prev_soc = int(prev_obs[7])
    change_in_soc = next_soc - prev_soc
    if change_in_soc > 0 and n_next == 0:
        reward -= 200.0 * change_in_soc
    if change_in_soc < 1 and n_next >= 2 and next_soc < _NUM_SOC - 1:
        soc_gap = (_NUM_SOC - 1) - next_soc
        reward -= 100.0 * soc_gap
    if change_in_soc != 0:
        reward -= 50.0

    # Severity-aware ICU overuse (reward3).
    if p["icu_overuse"] and next_soc == _ICU:
        reward -= max(0, 2 - n_next) * 50.0

    # SOC resource cost (reward4).
    if p["soc_cost"] is not None:
        reward -= float(p["soc_cost"][next_soc])

    # Treatment costs (from the executed action).
    antib, vent, vaso, _ = _decode_action(int(action_idx))
    c_antib, c_vent, c_vaso = p["treat"]
    if antib == 1: reward -= c_antib
    if vent == 1:  reward -= c_vent
    if vaso == 1:  reward -= c_vaso

    return float(reward)
