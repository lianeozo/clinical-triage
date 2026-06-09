import math

import numpy as np
import torch

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.agents.random import RandomAgent
from triage_rl.agents.noop import NoOpAgent
from triage_rl.config import DQNAgentConfig, PPOAgentConfig
from phase1_ppo_dqn.agents.dqn import DQNAgent
from phase1_ppo_dqn.agents.ppo import PPOAgent
from phase1_ppo_dqn.agents.fact_ppo import FactorizedPPOAgent
from phase2_sac.feasibility import FEASIBILITY_MASK


def test_random_agent_acts_in_range():
    agent = RandomAgent(seed=0)
    obs = np.zeros(State.NUM_STATE_VARS, dtype=np.float32)
    for _ in range(100):
        a = agent.act(obs)
        assert 0 <= a < Action.NUM_ACTIONS_TOTAL
        assert isinstance(a, int)


def test_random_agent_is_seeded():
    a1 = RandomAgent(seed=42)
    a2 = RandomAgent(seed=42)
    obs = np.zeros(State.NUM_STATE_VARS, dtype=np.float32)
    seq1 = [a1.act(obs) for _ in range(20)]
    seq2 = [a2.act(obs) for _ in range(20)]
    assert seq1 == seq2


def test_noop_agent_keeps_current_soc_no_treatment():
    agent = NoOpAgent()
    for soc in (State.ASYNC, State.AMBULATORY, State.FACILITY, State.ICU):
        obs = np.zeros(State.NUM_STATE_VARS, dtype=np.float32)
        obs[State.SOC_IDX] = soc
        a_idx = agent.act(obs)
        a = Action(action_idx=a_idx)
        assert a.soc == soc, f"NoOp should stay at SOC {soc}, got {a.soc}"
        assert a.antibiotic == 0
        assert a.ventilation == 0
        assert a.vasopressors == 0


def test_dqn_agent_acts_in_range():
    cfg = DQNAgentConfig()
    agent = DQNAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                     total_env_steps=1000, config=cfg, seed=0, device="cpu")
    obs = np.zeros(State.NUM_STATE_VARS, dtype=np.float32)
    for _ in range(50):
        a = agent.act(obs, eval_mode=False)
        assert 0 <= a < Action.NUM_ACTIONS_TOTAL
        assert isinstance(a, int)


def test_dqn_agent_update_returns_expected_keys(tmp_path):
    cfg = DQNAgentConfig()
    agent = DQNAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                     total_env_steps=1000, config=cfg, seed=0, device="cpu")
    batch = {
        "obs": np.zeros((32, State.NUM_STATE_VARS), dtype=np.float32),
        "action": np.zeros(32, dtype=np.int64),
        "scaled_reward": np.zeros(32, dtype=np.float32),
        "next_obs": np.zeros((32, State.NUM_STATE_VARS), dtype=np.float32),
        "terminated": np.zeros(32, dtype=np.bool_),
    }
    metrics = agent.update(batch)
    for k in ("loss", "q_mean", "q_max", "target_drift_l2"):
        assert k in metrics


def test_dqn_save_load_roundtrip(tmp_path):
    cfg = DQNAgentConfig()
    a1 = DQNAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                  total_env_steps=1000, config=cfg, seed=0, device="cpu")
    # Burn some steps so _steps_done is non-zero and worth round-tripping.
    obs = np.zeros(State.NUM_STATE_VARS, dtype=np.float32)
    for _ in range(5):
        a1.act(obs, eval_mode=False)
    path = tmp_path / "dqn.pt"
    a1.save(path)
    a2 = DQNAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                  total_env_steps=1000, config=cfg, seed=99, device="cpu")
    a2.load(path)
    # In eval mode both should produce the same greedy action (policy_net restored).
    assert a1.act(obs, eval_mode=True) == a2.act(obs, eval_mode=True)
    # _steps_done round-tripped.
    assert a2._steps_done == a1._steps_done
    # target_net parameters match.
    for p1, p2 in zip(a1.target_net.parameters(), a2.target_net.parameters()):
        assert torch.allclose(p1, p2)


def test_ppo_agent_acts_in_range():
    cfg = PPOAgentConfig()
    agent = PPOAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                     config=cfg, seed=0, device="cpu")
    obs = np.zeros(State.NUM_STATE_VARS, dtype=np.float32)
    for _ in range(50):
        a = agent.act(obs, eval_mode=False)
        assert 0 <= a < Action.NUM_ACTIONS_TOTAL
        assert isinstance(a, int)


def test_ppo_agent_get_logp_value():
    cfg = PPOAgentConfig()
    agent = PPOAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                     config=cfg, seed=0, device="cpu")
    obs = np.zeros((4, State.NUM_STATE_VARS), dtype=np.float32)
    actions = np.array([0, 1, 2, 3], dtype=np.int64)
    logp, value = agent.get_logp_value(obs, actions)
    assert logp.shape == (4,)
    assert value.shape == (4,)


def test_ppo_agent_update_returns_expected_keys():
    cfg = PPOAgentConfig()
    agent = PPOAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                     config=cfg, seed=0, device="cpu")
    n = 32
    batch = {
        "obs": np.zeros((n, State.NUM_STATE_VARS), dtype=np.float32),
        "action": np.zeros(n, dtype=np.int64),
        "old_logprob": np.zeros(n, dtype=np.float32),
        "advantage": np.zeros(n, dtype=np.float32),
        "return_": np.zeros(n, dtype=np.float32),
    }
    metrics = agent.update(batch)
    for k in ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac"):
        assert k in metrics


def test_ppo_update_moves_params():
    """A single PPO update with non-zero advantage must change at least one parameter."""
    cfg = PPOAgentConfig()
    agent = PPOAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                     config=cfg, seed=0, device="cpu")
    rng = np.random.default_rng(0)
    n = 64
    batch = {
        "obs": rng.standard_normal((n, State.NUM_STATE_VARS)).astype(np.float32),
        "action": rng.integers(0, Action.NUM_ACTIONS_TOTAL, size=n).astype(np.int64),
        "old_logprob": (-np.log(Action.NUM_ACTIONS_TOTAL) * np.ones(n)).astype(np.float32),
        "advantage": rng.standard_normal(n).astype(np.float32),
        "return_": rng.standard_normal(n).astype(np.float32),
    }
    before = [p.detach().clone() for p in agent.model.parameters()]
    agent.update(batch)
    after = list(agent.model.parameters())
    moved = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert moved, "PPO update should move at least one parameter"


def test_ppo_save_load_roundtrip(tmp_path):
    cfg = PPOAgentConfig()
    a1 = PPOAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                  config=cfg, seed=0, device="cpu")
    # Run one update so weights are non-default.
    rng = np.random.default_rng(0)
    n = 32
    batch = {
        "obs": rng.standard_normal((n, State.NUM_STATE_VARS)).astype(np.float32),
        "action": rng.integers(0, Action.NUM_ACTIONS_TOTAL, size=n).astype(np.int64),
        "old_logprob": (-np.log(Action.NUM_ACTIONS_TOTAL) * np.ones(n)).astype(np.float32),
        "advantage": rng.standard_normal(n).astype(np.float32),
        "return_": rng.standard_normal(n).astype(np.float32),
    }
    a1.update(batch)
    path = tmp_path / "ppo.pt"
    a1.save(path)
    a2 = PPOAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                  config=cfg, seed=99, device="cpu")
    a2.load(path)
    obs = np.zeros(State.NUM_STATE_VARS, dtype=np.float32)
    assert a1.act(obs, eval_mode=True) == a2.act(obs, eval_mode=True)
    for p1, p2 in zip(a1.model.parameters(), a2.model.parameters()):
        assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# FactPPO tests
# ---------------------------------------------------------------------------

def test_factppo_action_always_feasible():
    """Every action sampled by FactPPO must pass the global feasibility mask."""
    cfg = PPOAgentConfig()
    agent = FactorizedPPOAgent(
        obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
        config=cfg, seed=0, device="cpu",
    )
    rng = np.random.default_rng(7)
    for _ in range(50):
        obs = rng.standard_normal(State.NUM_STATE_VARS).astype(np.float32)
        a = agent.act(obs, eval_mode=False)
        assert FEASIBILITY_MASK[a], f"action {a} is infeasible"


def test_factppo_logprob_decomposition():
    """Joint log_prob from act_with_logp_value equals log π_soc + log π_treat manually."""
    cfg = PPOAgentConfig()
    agent = FactorizedPPOAgent(
        obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
        config=cfg, seed=42, device="cpu",
    )
    rng = np.random.default_rng(0)
    obs_np = rng.standard_normal(State.NUM_STATE_VARS).astype(np.float32)

    for _ in range(10):
        action_idx, logp_joint, _ = agent.act_with_logp_value(obs_np)
        soc = action_idx % 4
        treat = action_idx // 4

        x = torch.from_numpy(obs_np).unsqueeze(0)
        with torch.no_grad():
            soc_logits, treat_logits_all, _ = agent.model(x)
            from phase1_ppo_dqn.agents.fact_ppo import _apply_feasibility_mask
            from torch.distributions import Categorical as Cat
            logp_soc = Cat(logits=soc_logits).log_prob(torch.tensor([soc])).item()
            t_logits = treat_logits_all[0, soc]
            t_logits = _apply_feasibility_mask(t_logits.unsqueeze(0), torch.tensor([soc]), agent.model.treat_feasible).squeeze(0)
            logp_treat = Cat(logits=t_logits).log_prob(torch.tensor(treat)).item()

        assert abs(logp_joint - (logp_soc + logp_treat)) < 1e-5, (
            f"logp_joint={logp_joint:.6f} != logp_soc+logp_treat={logp_soc+logp_treat:.6f}"
        )


def test_factppo_update_finite():
    """A single FactPPO update on random data must produce finite metrics."""
    cfg = PPOAgentConfig()
    agent = FactorizedPPOAgent(
        obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
        config=cfg, seed=0, device="cpu",
    )
    rng = np.random.default_rng(1)
    n = 32
    feasible_actions = [i for i in range(Action.NUM_ACTIONS_TOTAL) if FEASIBILITY_MASK[i]]
    actions = rng.choice(feasible_actions, size=n).astype(np.int64)
    batch = {
        "obs": rng.standard_normal((n, State.NUM_STATE_VARS)).astype(np.float32),
        "action": actions,
        "old_logprob": rng.standard_normal(n).astype(np.float32),
        "advantage": rng.standard_normal(n).astype(np.float32),
        "return_": rng.standard_normal(n).astype(np.float32),
    }
    metrics = agent.update(batch)
    for key in ("pg_loss", "v_loss", "soc_entropy", "treat_entropy"):
        assert math.isfinite(metrics[key]), f"{key}={metrics[key]} is not finite"


def test_factppo_soc_dist_sums_to_one():
    """Softmax over SOC logits must sum to 1 for every observation in a batch."""
    cfg = PPOAgentConfig()
    agent = FactorizedPPOAgent(
        obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
        config=cfg, seed=3, device="cpu",
    )
    rng = np.random.default_rng(5)
    batch_obs = torch.from_numpy(
        rng.standard_normal((20, State.NUM_STATE_VARS)).astype(np.float32)
    )
    with torch.no_grad():
        soc_logits, _, _ = agent.model(batch_obs)
        soc_probs = soc_logits.softmax(dim=-1)
    row_sums = soc_probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(20), atol=1e-5), (
        f"SOC distribution does not sum to 1; max deviation={((row_sums - 1).abs().max()).item()}"
    )
