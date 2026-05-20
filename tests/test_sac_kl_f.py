"""Unit tests for SACKLFAgent and the feasibility mask."""
import numpy as np
import torch

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from phase2_sac.feasibility import FEASIBILITY_MASK
from triage_rl.config import SACKLFAgentConfig
from phase2_sac.agents.sac_kl_f import SACKLFAgent


def test_feasibility_mask_count():
    """Feasibility derivation: ASYNC/AMB allow (vent=0, vaso=0)*(antib in {0,1}) = 2 actions each;
    FAC/ICU allow all 8 treatment combos. Total = 2 + 2 + 8 + 8 = 20."""
    assert FEASIBILITY_MASK.shape == (32,)
    assert FEASIBILITY_MASK.dtype == np.bool_
    assert int(FEASIBILITY_MASK.sum()) == 20


def test_feasibility_mask_specific_actions():
    # action_idx = 16*antib + 8*vent + 4*vaso + soc
    # ASYNC (soc=0): vent=0 vaso=0 → idx 0 (no treatment), idx 16 (antib only). Feasible.
    assert FEASIBILITY_MASK[0] == True
    assert FEASIBILITY_MASK[16] == True
    # ASYNC + vent=1 (idx 8) → infeasible.
    assert FEASIBILITY_MASK[8] == False
    # ICU (soc=3) + all treatments (antib+vent+vaso) → idx 31 → feasible.
    assert FEASIBILITY_MASK[31] == True


def _make_agent(seed=0, beta=10.0):
    cfg = SACKLFAgentConfig(feasibility_beta=beta)
    return SACKLFAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                      config=cfg, seed=seed, device="cpu")


def _random_batch(rng, n=64):
    return {
        "obs": rng.standard_normal((n, State.NUM_STATE_VARS)).astype(np.float32),
        "action": rng.integers(0, Action.NUM_ACTIONS_TOTAL, size=n).astype(np.int64),
        "scaled_reward": rng.standard_normal(n).astype(np.float32) * 0.5,
        "next_obs": rng.standard_normal((n, State.NUM_STATE_VARS)).astype(np.float32),
        "terminated": rng.integers(0, 2, size=n).astype(np.bool_),
    }


def test_sac_kl_f_returns_infeasible_mass_metric():
    agent = _make_agent()
    rng = np.random.default_rng(0)
    metrics = agent.update(_random_batch(rng))
    assert "infeasible_mass" in metrics
    assert 0.0 <= metrics["infeasible_mass"] <= 1.0


def test_sac_kl_f_regularizer_drives_infeasible_mass_down():
    """With high β, the regularizer should push infeasible_mass down over training steps."""
    agent = _make_agent(beta=10.0)
    rng = np.random.default_rng(0)
    initial = agent.update(_random_batch(rng))["infeasible_mass"]
    for _ in range(49):
        agent.update(_random_batch(rng))
    final = agent.update(_random_batch(rng))["infeasible_mass"]
    assert final < initial, f"infeasible_mass did not decrease: initial={initial:.4f}, final={final:.4f}"
