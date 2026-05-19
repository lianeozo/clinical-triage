import numpy as np

from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.State import State
from triage_rl.agents.random import RandomAgent
from triage_rl.agents.noop import NoOpAgent


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
