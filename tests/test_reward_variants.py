import numpy as np
import pytest
from sepsisSimDiabetes.reward_variants import compute_reward, REWARD_VARIANTS, num_abnormal, on_treatment

# obs: [hr, sysbp, percoxyg, glucose, antib, vaso, vent, soc]
NORMAL = np.array([1, 1, 1, 2, 0, 0, 0, 1])  # 0 abnormal, no treatment, AMBULATORY

def test_variant_keys():
    assert set(REWARD_VARIANTS) == {"reward0", "reward1", "reward2", "reward3", "reward4"}

def test_num_abnormal_skips_masked():
    o = np.array([0, -1, 1, 2, 0, 0, 0, 1])  # hr abnormal(0), sysbp masked(-1)
    assert num_abnormal(o) == 1

def test_on_treatment():
    assert not on_treatment(NORMAL)
    assert on_treatment(np.array([1,1,1,2,1,0,0,1]))  # antibiotic on

def test_death_terminal_uses_variant_scale():
    dead = np.array([0, 0, 0, 0, 0, 0, 0, 3])  # >=3 abnormal
    assert compute_reward(NORMAL, 0, dead, "reward0") == -10000
    assert compute_reward(NORMAL, 0, dead, "reward1") == -1000

def test_discharge_terminal():
    # next has 0 abnormal and no treatment -> +T regardless of action
    assert compute_reward(NORMAL, 0, NORMAL, "reward0") == 10000
    assert compute_reward(NORMAL, 0, NORMAL, "reward2") == 1000

def test_reward2_treatment_costs():
    # non-terminal step, no soc change, no vital change; action 19 = antib+soc3? decode below.
    # action_idx 27 = antib(1)*16 + vent(1)*8 + vaso(0)*4 + soc(3): antib+vent at ICU
    prev = np.array([0, 0, 1, 2, 0, 0, 0, 3])  # 2 abnormal, ICU
    nxt  = np.array([0, 0, 1, 2, 1, 0, 1, 3])  # 2 abnormal still, antib+vent on, ICU
    # reward2 treatment: antib -60, vent -120; no vital change (2->2); soc change 0
    r = compute_reward(prev, 27, nxt, "reward2")
    assert r == -60 - 120  # -180

def test_reward3_icu_overuse():
    # ICU with only 1 abnormal -> overuse penalty max(0,2-1)*50 = 50, treatment reward1 costs
    prev = np.array([0, 1, 1, 2, 0, 0, 0, 3])  # 1 abnormal, ICU
    nxt  = np.array([0, 1, 1, 2, 1, 0, 0, 3])  # 1 abnormal, antib on, ICU
    r = compute_reward(prev, 19, nxt, "reward3")  # action 19 = antib + soc3
    # vital change 0; soc change 0; icu-overuse -50; antib -10
    assert r == -50 - 10

def test_reward4_soc_resource_cost():
    prev = np.array([0, 1, 1, 2, 0, 0, 0, 3])  # 1 abnormal ICU
    nxt  = np.array([0, 1, 1, 2, 0, 0, 0, 3])  # unchanged, no treatment, ICU
    r = compute_reward(prev, 3, nxt, "reward4")  # action 3 = soc3, no treatment
    assert r == -50  # ICU resource cost; no treatment, no vital/soc change
