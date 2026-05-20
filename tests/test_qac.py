"""Unit tests for QACAgent."""
import numpy as np
import pytest
import torch

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.config import QACAgentConfig
from phase2_qac.agents.qac import QACAgent


def _make_agent(seed=0):
    return QACAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                    config=QACAgentConfig(), seed=seed, device="cpu")


def _random_batch(rng, n=32):
    return {
        "obs": rng.standard_normal((n, State.NUM_STATE_VARS)).astype(np.float32),
        "action": rng.integers(0, Action.NUM_ACTIONS_TOTAL, size=n).astype(np.int64),
        "scaled_reward": rng.standard_normal(n).astype(np.float32) * 0.5,
        "next_obs": rng.standard_normal((n, State.NUM_STATE_VARS)).astype(np.float32),
        "terminated": rng.integers(0, 2, size=n).astype(np.bool_),
    }


def test_qac_act_in_range():
    agent = _make_agent()
    obs = np.zeros(State.NUM_STATE_VARS, dtype=np.float32)
    for _ in range(50):
        a = agent.act(obs, eval_mode=False)
        assert 0 <= a < Action.NUM_ACTIONS_TOTAL
        assert isinstance(a, int)


def test_qac_act_eval_mode_deterministic():
    agent = _make_agent()
    obs = np.zeros(State.NUM_STATE_VARS, dtype=np.float32)
    a1 = agent.act(obs, eval_mode=True)
    a2 = agent.act(obs, eval_mode=True)
    assert a1 == a2


def test_qac_update_returns_expected_keys():
    agent = _make_agent()
    rng = np.random.default_rng(0)
    metrics = agent.update(_random_batch(rng))
    for k in ("q_loss", "actor_loss", "alpha_loss", "alpha",
              "entropy", "q1_mean", "q2_mean", "target_drift_l2"):
        assert k in metrics, f"missing {k}"


def test_qac_twin_q_min_is_lower():
    """min(Q1, Q2) <= each."""
    agent = _make_agent()
    obs = torch.zeros((4, State.NUM_STATE_VARS), dtype=torch.float32)
    with torch.no_grad():
        q1 = agent.q1(obs)
        q2 = agent.q2(obs)
        qmin = torch.min(q1, q2)
        assert (qmin <= q1).all()
        assert (qmin <= q2).all()


def test_qac_update_moves_params():
    agent = _make_agent()
    rng = np.random.default_rng(0)
    before = [p.detach().clone() for p in agent.actor.parameters()]
    agent.update(_random_batch(rng))
    after = list(agent.actor.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_qac_save_load_roundtrip(tmp_path):
    a1 = _make_agent(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(3):
        a1.update(_random_batch(rng))
    path = tmp_path / "qac.pt"
    a1.save(path)
    a2 = _make_agent(seed=99)
    a2.load(path)
    obs = np.zeros(State.NUM_STATE_VARS, dtype=np.float32)
    assert a1.act(obs, eval_mode=True) == a2.act(obs, eval_mode=True)
    # actor params match
    for p1, p2 in zip(a1.actor.parameters(), a2.actor.parameters()):
        assert torch.allclose(p1, p2)
