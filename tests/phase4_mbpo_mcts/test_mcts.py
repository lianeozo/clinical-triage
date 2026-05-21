"""Tests for the AlphaZero-style MCTS planner.

The planner is a self-contained function ``run_mcts`` that takes callable
closures for the dynamics model and the policy/value net. These tests use
trivial lambdas as fakes, so the planner is exercised in isolation from
torch.
"""
import numpy as np

from triage_rl.config import MCTSConfig
from phase4_mbpo_mcts.agents.mcts import run_mcts


def _fake_model_step(history, action, member_idx):
    """Deterministic 1-step env.

    next_obs = [action], reward = 1.0 iff action==7 else 0.0, done after 1 step.
    """
    nxt = np.array([action], dtype=np.int64)
    r = 1.0 if action == 7 else 0.0
    return nxt, r, True


def _fake_pi_v(history):
    """Uniform prior, value=0."""
    return np.ones(32) / 32, 0.0


def test_mcts_finds_rewarding_action():
    cfg = MCTSConfig(
        n_simulations=200,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.0,
    )
    rng = np.random.default_rng(0)
    res = run_mcts(
        root_obs=np.zeros(1, dtype=np.int64),
        root_history=[],
        model_step_fn=_fake_model_step,
        pi_v_fn=_fake_pi_v,
        config=cfg,
        n_actions=32,
        n_ensemble_members=1,
        rng=rng,
    )
    # action 7 is the only rewarding action; should accumulate most visits
    best = max(res.visit_counts, key=res.visit_counts.get)
    assert best == 7
    others = sum(v for k, v in res.visit_counts.items() if k != 7)
    assert res.visit_counts[7] > others / 4


def test_mcts_root_dirichlet_noise_changes_priors():
    cfg = MCTSConfig(n_simulations=2, dirichlet_alpha=10.0, dirichlet_eps=0.99)
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(1)
    r1 = run_mcts(
        np.zeros(1, dtype=np.int64), [], _fake_model_step, _fake_pi_v,
        cfg, 32, 1, rng1,
    )
    r2 = run_mcts(
        np.zeros(1, dtype=np.int64), [], _fake_model_step, _fake_pi_v,
        cfg, 32, 1, rng2,
    )
    # Different rngs + heavy noise should produce different first-sim choices.
    assert r1.visit_counts != r2.visit_counts


def test_mcts_terminal_uses_real_reward_not_vpsi():
    """When the leaf is terminal, the leaf value must be the env's terminal
    reward, NOT V_psi(history). Here V_psi=-100 but action 0 yields reward
    1.0 with done=True, so MCTS should still prefer action 0.
    """
    def model_step(history, action, member_idx):
        return (
            np.array([action], dtype=np.int64),
            1.0 if action == 0 else 0.0,
            True,
        )

    def pi_v(history):
        return np.ones(32) / 32, -100.0

    cfg = MCTSConfig(n_simulations=100, dirichlet_eps=0.0)
    res = run_mcts(
        np.zeros(1, dtype=np.int64), [], model_step, pi_v,
        cfg, 32, 1, np.random.default_rng(0),
    )
    assert res.visit_counts[0] > res.visit_counts.get(1, 0)
    # positive Q at root despite negative V_psi
    assert max(res.root_q.values()) > 0
