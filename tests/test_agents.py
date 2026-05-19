import numpy as np
import torch

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.agents.random import RandomAgent
from triage_rl.agents.noop import NoOpAgent
from triage_rl.config import DQNAgentConfig
from phase1_ppo_dqn.agents.dqn import DQNAgent


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
    path = tmp_path / "dqn.pt"
    a1.save(path)
    a2 = DQNAgent(obs_dim=State.NUM_STATE_VARS, n_actions=Action.NUM_ACTIONS_TOTAL,
                  total_env_steps=1000, config=cfg, seed=99, device="cpu")
    a2.load(path)
    obs = np.zeros(State.NUM_STATE_VARS, dtype=np.float32)
    # In eval mode both should produce the same greedy action.
    assert a1.act(obs, eval_mode=True) == a2.act(obs, eval_mode=True)
