"""Unit tests for SACKLPPOAgent and the PPO-reference loader."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.config import PPOAgentConfig, SACKLPPOAgentConfig
from phase1_ppo_dqn.agents.ppo import _ActorCritic
from phase2_sac.ppo_reference import find_final_checkpoint, load_ppo_actor
from phase2_sac.agents.sac_kl_ppo import SACKLPPOAgent


@pytest.fixture
def fake_ppo_run_dir(tmp_path):
    """Build a synthetic Phase-1 PPO run dir with one seed's _final.pt checkpoint."""
    ppo_cfg = PPOAgentConfig()
    model = _ActorCritic(State.NUM_STATE_VARS, ppo_cfg.hidden_dim,
                         Action.NUM_ACTIONS_TOTAL)
    run_dir = tmp_path / "fake_ppo_run"
    seed_dir = run_dir / "seed_0"
    (seed_dir / "checkpoints").mkdir(parents=True)
    ckpt_path = seed_dir / "checkpoints" / "step_500000_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    return run_dir


def test_find_final_checkpoint(fake_ppo_run_dir):
    p = find_final_checkpoint(fake_ppo_run_dir / "seed_0")
    assert p.name == "step_500000_final.pt"


def test_load_ppo_actor_freezes_params(fake_ppo_run_dir):
    model = load_ppo_actor(fake_ppo_run_dir, seed=0,
                           obs_dim=State.NUM_STATE_VARS,
                           n_actions=Action.NUM_ACTIONS_TOTAL,
                           device="cpu")
    assert all(not p.requires_grad for p in model.parameters())


def _random_batch(rng, n=32):
    return {
        "obs": rng.standard_normal((n, State.NUM_STATE_VARS)).astype(np.float32),
        "action": rng.integers(0, Action.NUM_ACTIONS_TOTAL, size=n).astype(np.int64),
        "scaled_reward": rng.standard_normal(n).astype(np.float32) * 0.5,
        "next_obs": rng.standard_normal((n, State.NUM_STATE_VARS)).astype(np.float32),
        "terminated": rng.integers(0, 2, size=n).astype(np.bool_),
    }


def test_sac_kl_ppo_kl_term_present_and_finite(fake_ppo_run_dir):
    cfg = SACKLPPOAgentConfig(kl_beta=0.5)
    agent = SACKLPPOAgent(obs_dim=State.NUM_STATE_VARS,
                       n_actions=Action.NUM_ACTIONS_TOTAL,
                       config=cfg, seed=0, device="cpu",
                       ppo_run_dir=fake_ppo_run_dir)
    rng = np.random.default_rng(0)
    metrics = agent.update(_random_batch(rng))
    assert "kl_to_ppo" in metrics
    assert np.isfinite(metrics["kl_to_ppo"])


def test_sac_kl_ppo_inherits_sac_metric_keys(fake_ppo_run_dir):
    cfg = SACKLPPOAgentConfig()
    agent = SACKLPPOAgent(obs_dim=State.NUM_STATE_VARS,
                       n_actions=Action.NUM_ACTIONS_TOTAL,
                       config=cfg, seed=0, device="cpu",
                       ppo_run_dir=fake_ppo_run_dir)
    rng = np.random.default_rng(0)
    metrics = agent.update(_random_batch(rng))
    for k in ("q_loss", "actor_loss", "alpha_loss", "alpha", "entropy",
              "q1_mean", "q2_mean", "target_drift_l2", "kl_to_ppo"):
        assert k in metrics, f"missing {k}"


def test_sac_kl_ppo_requires_ppo_run_dir():
    cfg = SACKLPPOAgentConfig()
    with pytest.raises(ValueError, match="ppo_run_dir"):
        SACKLPPOAgent(obs_dim=State.NUM_STATE_VARS,
                   n_actions=Action.NUM_ACTIONS_TOTAL,
                   config=cfg, seed=0, device="cpu",
                   ppo_run_dir=None)
