"""Unit tests for IQLKLFAgent."""
import numpy as np
import pytest
import torch

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.config import IQLKLFAgentConfig
from phase3_iql.agents.iql_kl_f import IQLKLFAgent


def _make_agent(seed=0, beta=10.0):
    cfg = IQLKLFAgentConfig(feasibility_beta=beta)
    return IQLKLFAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                       config=cfg, seed=seed, device="cpu")


def _random_batch(rng, n=64):
    return {
        "obs": rng.standard_normal((n, State.NUM_STATE_VARS)).astype(np.float32),
        "action": rng.integers(0, Action.NUM_ACTIONS_TOTAL, size=n).astype(np.int64),
        "scaled_reward": rng.standard_normal(n).astype(np.float32) * 0.5,
        "next_obs": rng.standard_normal((n, State.NUM_STATE_VARS)).astype(np.float32),
        "terminated": rng.integers(0, 2, size=n).astype(np.bool_),
    }


def test_iql_kl_f_returns_infeasible_mass():
    agent = _make_agent()
    rng = np.random.default_rng(0)
    metrics = agent.update(_random_batch(rng))
    assert "infeasible_mass" in metrics
    assert 0.0 <= metrics["infeasible_mass"] <= 1.0


def test_iql_kl_f_regularizer_drives_infeasible_mass_down():
    """With high beta_feas, the regularizer should push infeasible_mass down over training steps."""
    agent = _make_agent(beta=10.0)
    rng = np.random.default_rng(0)
    initial = agent.update(_random_batch(rng))["infeasible_mass"]
    for _ in range(49):
        agent.update(_random_batch(rng))
    final = agent.update(_random_batch(rng))["infeasible_mass"]
    assert final < initial, f"infeasible_mass did not decrease: initial={initial:.4f}, final={final:.4f}"
