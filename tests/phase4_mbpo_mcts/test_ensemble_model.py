"""Ensemble recurrent transition model tests."""
import numpy as np
import torch
from triage_rl.config import EnsembleModelConfig
from phase4_mbpo_mcts.agents.ensemble_model import EnsembleModel

OBS_DIMS = [3, 3, 2, 5, 2, 2, 2, 4]  # HR, SysBP, OXY, GLUC, ANTIB, VASO, VENT, SOC
N_ACTIONS = 32


def _make(cfg=None, seed=0):
    cfg = cfg or EnsembleModelConfig(n_members=3, hidden_dim=32)
    return EnsembleModel(obs_dims=OBS_DIMS, n_actions=N_ACTIONS, config=cfg, seed=seed, device="cpu")


def test_predict_shape():
    m = _make()
    # Batch B=4, sequence T=5
    obs_hist = torch.randint(0, 3, (4, 5, len(OBS_DIMS)))  # any small ints
    obs_hist = obs_hist.clamp(max=1)  # keep within bin ranges per var safely
    act_hist = torch.randint(0, N_ACTIONS, (4, 5))
    logits_list = m.predict(obs_hist, act_hist, member_idx=0)
    assert len(logits_list) == len(OBS_DIMS)
    for i, dim in enumerate(OBS_DIMS):
        assert logits_list[i].shape == (4, dim)


def test_members_initialized_differently():
    m = _make(seed=0)
    obs = torch.zeros(1, 1, len(OBS_DIMS), dtype=torch.long)
    act = torch.zeros(1, 1, dtype=torch.long)
    l0 = m.predict(obs, act, member_idx=0)[0]
    l1 = m.predict(obs, act, member_idx=1)[0]
    assert not torch.allclose(l0, l1)


def test_nll_decreases_on_fixed_minibatch():
    cfg = EnsembleModelConfig(n_members=1, hidden_dim=32, lr=1e-2)
    m = EnsembleModel(obs_dims=OBS_DIMS, n_actions=N_ACTIONS, config=cfg, seed=0, device="cpu")
    rng = np.random.default_rng(0)
    B, T = 8, 6
    obs_hist = torch.tensor(rng.integers(0, 2, (B, T, len(OBS_DIMS))), dtype=torch.long)
    act_hist = torch.tensor(rng.integers(0, N_ACTIONS, (B, T)), dtype=torch.long)
    target_next = torch.tensor(rng.integers(0, 2, (B, len(OBS_DIMS))), dtype=torch.long)
    initial_loss = m.compute_loss(obs_hist, act_hist, target_next, member_idx=0).item()
    for _ in range(50):
        loss = m.compute_loss(obs_hist, act_hist, target_next, member_idx=0)
        m.optimizers[0].zero_grad(); loss.backward(); m.optimizers[0].step()
    final_loss = m.compute_loss(obs_hist, act_hist, target_next, member_idx=0).item()
    assert final_loss < initial_loss * 0.8
